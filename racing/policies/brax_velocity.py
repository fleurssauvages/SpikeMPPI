from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from .base import JointPolicy


def body_motion_numpy(robot, data) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(data.xmat[robot.root_body_id], dtype=np.float64).reshape(3, 3)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    return rot.T @ qvel[:3], rot.T @ qvel[3:6]


def rapid_observation_numpy(
    robot,
    data,
    command: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Native-MuJoCo counterpart of the Rapid-Locomotion MJX observation."""
    body_linear, body_angular = body_motion_numpy(robot, data)
    rot = np.asarray(data.xmat[robot.root_body_id], dtype=np.float64).reshape(3, 3)
    projected_gravity = rot.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    qpos0 = np.asarray(robot.model.qpos0, dtype=np.float64)
    return np.concatenate([
        body_linear,
        body_angular,
        projected_gravity,
        np.asarray(command, dtype=np.float64).reshape(3),
        qpos[7:] - qpos0[7:],
        qvel[6:],
        np.asarray(previous_action, dtype=np.float64).reshape(robot.nu),
    ]).astype(np.float32, copy=False)


def _legacy_observation(robot, data, command: np.ndarray, previous_action: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.asarray(data.qpos[2:], dtype=np.float32),
        np.asarray(data.qvel, dtype=np.float32),
        np.asarray(command, dtype=np.float32).reshape(3),
        np.asarray(previous_action, dtype=np.float32).reshape(robot.nu),
    ]).astype(np.float32, copy=False)


class RapidCommandEnvelope:
    """Query the learned joint (v_x, omega_z) command curriculum envelope."""

    def __init__(self, metadata: dict):
        curriculum = dict(metadata.get("curriculum", {}))
        cfg = dict(curriculum.get("config", {}))
        # Use only cells that actually passed the curriculum tracking test.
        # ``active_cells`` includes the newly unlocked shell, which is useful for
        # training but is not yet evidence that the policy can race there.
        cells = curriculum.get("certified_cells") or curriculum.get("active_cells")
        if cells is None:
            cmin = np.asarray(metadata.get("command_min", [-0.5, -1.0, -2.0]), dtype=np.float64)
            cmax = np.asarray(metadata.get("command_max", [3.0, 1.0, 2.0]), dtype=np.float64)
            self.cells = np.asarray([
                [max(0.0, cmax[0]), 0.0],
                [max(0.0, cmax[0]), cmax[2]],
                [max(0.0, cmax[0]), cmin[2]],
            ], dtype=np.float64)
            self.vy_min = float(cmin[1])
            self.vy_max = float(cmax[1])
            self.step_vx = 0.5
            self.step_wz = 0.5
        else:
            self.cells = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
            self.vy_min = float(cfg.get("vy_min", -0.6))
            self.vy_max = float(cfg.get("vy_max", 0.6))
            self.step_vx = float(cfg.get("grid_step_vx", 0.5))
            self.step_wz = float(cfg.get("grid_step_wz", 0.5))
        positive = self.cells[self.cells[:, 0] > 1e-6]
        self.max_forward = float(np.max(positive[:, 0])) if len(positive) else 1.0
        self.max_yaw = float(np.max(np.abs(self.cells[:, 1]))) if len(self.cells) else 1.0

    def fastest_speed(self, curvature: float, cap: float | None = None) -> float:
        kappa = float(curvature)
        cap_value = self.max_forward if cap is None else min(self.max_forward, max(0.0, float(cap)))
        candidates = self.cells[(self.cells[:, 0] > 0.0) & (self.cells[:, 0] <= cap_value + 1e-9)]
        if len(candidates) == 0:
            return max(0.2, cap_value)
        order = np.argsort(candidates[:, 0])[::-1]
        yaw_tol = max(0.30, 0.76 * self.step_wz)
        for idx in order:
            vx, wz_cell = candidates[idx]
            required_wz = vx * kappa
            if abs(wz_cell - required_wz) <= yaw_tol:
                return float(vx)
        if abs(kappa) < 1e-8:
            return float(np.max(candidates[:, 0]))
        return float(max(0.2, min(cap_value, self.max_yaw / abs(kappa))))


class BraxVelocityPolicy(JointPolicy):
    """Deterministic high-speed locomotion policy used as the MPPI nominal.

    The neural policy generates native actuator controls.  The track only
    generates velocity commands; MPPI subsequently refines the joint controls
    directly.
    """

    name = "brax_velocity"

    def __init__(self, checkpoint_dir: str | Path, *, race_speed: float | None = None):
        self.path = Path(checkpoint_dir)
        metadata_path = self.path / "metadata.json"
        params_path = self.path / "params.pkl"
        if not metadata_path.exists() or not params_path.exists():
            raise FileNotFoundError(
                f"Expected metadata.json and params.pkl in velocity-policy directory {self.path}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.robot_name = str(self.metadata["robot"])
        self.control_dt = float(self.metadata["control_dt"])
        self.race_speed = None if race_speed is None else float(race_speed)
        self.observation_version = str(self.metadata.get("observation_version", "legacy_v1"))
        self.command_envelope = RapidCommandEnvelope(self.metadata)
        self.heading_gain = float(self.metadata.get("race_heading_gain", 2.0))
        self.lookahead_distances = tuple(
            float(x) for x in self.metadata.get(
                "race_curvature_lookahead_m", [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
            )
        )

        try:
            import jax
            import jax.numpy as jnp
            from flax import linen
            from brax.io import model as brax_model
            from brax.training.acme import running_statistics
            from brax.training.agents.ppo import networks as ppo_networks
        except ImportError as exc:
            raise RuntimeError("Loading a trained velocity policy requires JAX, Flax and Brax") from exc

        self._jax = jax
        self._jnp = jnp
        params = brax_model.load_params(str(params_path))
        policy_layers = tuple(int(x) for x in self.metadata["policy_hidden_layer_sizes"])
        value_layers = tuple(int(x) for x in self.metadata["value_hidden_layer_sizes"])
        activation_name = str(self.metadata.get("activation", "swish")).lower()
        activation = linen.elu if activation_name == "elu" else linen.swish
        networks = ppo_networks.make_ppo_networks(
            observation_size=int(self.metadata["observation_size"]),
            action_size=int(self.metadata["action_size"]),
            policy_hidden_layer_sizes=policy_layers,
            value_hidden_layer_sizes=value_layers,
            preprocess_observations_fn=running_statistics.normalize,
            distribution_type="tanh_normal",
            activation=activation,
            init_noise_std=float(self.metadata.get("init_noise_std", 1.0)),
        )
        make_inference_fn = ppo_networks.make_inference_fn(networks)
        self._inference = jax.jit(make_inference_fn(params, deterministic=True))
        self._key = jax.random.PRNGKey(0)

    @property
    def learned_max_speed(self) -> float:
        return self.command_envelope.max_forward

    def reset(self, robot, data) -> None:
        del data
        if robot.name != self.robot_name:
            raise ValueError(
                f"Policy was trained for {self.robot_name!r}, but race robot is {robot.name!r}."
            )
        if int(self.metadata["action_size"]) != robot.nu:
            raise ValueError("Policy action dimension does not match MuJoCo model.nu")

    def _normalized_previous_action(self, robot, data) -> np.ndarray:
        low, high = robot.control_bounds()
        half = np.maximum(0.5 * (high - low), 1e-8)
        return np.clip(
            (np.asarray(data.ctrl, dtype=np.float64) - 0.5 * (low + high)) / half,
            -1.0,
            1.0,
        )

    def action_for_command(self, robot, data, command: np.ndarray) -> np.ndarray:
        low, high = robot.control_bounds()
        previous_action = self._normalized_previous_action(robot, data)
        if self.observation_version == "rapid_v2":
            obs = rapid_observation_numpy(robot, data, command, previous_action)
        else:
            obs = _legacy_observation(robot, data, command, previous_action)
        if obs.size != int(self.metadata["observation_size"]):
            raise ValueError(
                f"Policy observation mismatch: produced {obs.size}, expected {self.metadata['observation_size']}"
            )
        self._key, act_key = self._jax.random.split(self._key)
        action, _ = self._inference(self._jnp.asarray(obs), act_key)
        normalized = np.asarray(action, dtype=np.float64).reshape(robot.nu)
        ctrl = 0.5 * (low + high) + 0.5 * (high - low) * np.clip(normalized, -1.0, 1.0)
        return robot.clip_ctrl(ctrl)

    def _fast_track_command(self, robot, data, track, prior, current_s: float) -> np.ndarray:
        # Look ahead in curvature and select the fastest command lying inside
        # the learned joint (v_x, omega_z) curriculum envelope.
        speeds = [
            self.command_envelope.fastest_speed(
                float(track.curvature(float(current_s) + d)), cap=self.race_speed
            )
            for d in self.lookahead_distances
        ]
        speed = float(min(speeds)) if speeds else self.command_envelope.fastest_speed(0.0, self.race_speed)

        target_lookahead = min(2.5, max(0.35, 0.35 * speed + 0.25))
        target_s = float(current_s) + target_lookahead
        mean, _ = prior.sample(track, target_s)
        p = robot.xy(data)
        direction = np.asarray(mean, dtype=np.float64).reshape(2) - p
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            direction = np.asarray(track.tangent(current_s), dtype=np.float64)
            norm = max(float(np.linalg.norm(direction)), 1e-8)
        direction /= norm
        v_world = speed * direction

        yaw = robot.root_yaw(data)
        c, s = math.cos(yaw), math.sin(yaw)
        vx_body = c * v_world[0] + s * v_world[1]
        vy_body = -s * v_world[0] + c * v_world[1]
        vy_body = float(np.clip(vy_body, self.command_envelope.vy_min, self.command_envelope.vy_max))
        desired_heading = math.atan2(direction[1], direction[0])
        heading_error = math.atan2(math.sin(desired_heading - yaw), math.cos(desired_heading - yaw))
        curvature = float(track.curvature(float(current_s) + min(0.5, target_lookahead)))
        wz_feedforward = speed * curvature
        wz = np.clip(
            wz_feedforward + self.heading_gain * heading_error,
            -self.command_envelope.max_yaw,
            self.command_envelope.max_yaw,
        )
        return np.asarray([vx_body, vy_body, wz], dtype=np.float32)

    def action(self, robot, data, *, track, prior, current_s: float) -> np.ndarray:
        command = self._fast_track_command(robot, data, track, prior, current_s)
        return self.action_for_command(robot, data, command)


__all__ = [
    "BraxVelocityPolicy",
    "RapidCommandEnvelope",
    "rapid_observation_numpy",
    "body_motion_numpy",
]
