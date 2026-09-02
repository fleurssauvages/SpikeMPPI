from __future__ import annotations

"""MJX/Warp high-speed command-conditioned locomotion environments.

The reward, 50 Hz control loop, 10 s command hold, push disturbances and
(v_x, yaw-rate) curriculum follow the morphology-independent parts of
Margolis et al., *Rapid Locomotion via Reinforcement Learning*.

Ant keeps the Margolis-style reward.  Spinner, snake, crawler and biped keep the same
PPO learner and command curriculum but add morphology-specific reward shaping
adapted from published locomotion objectives; see morphology_rewards.py.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from racing.robots.classic import classic_xml_path, find_classic_model
from .rapid_locomotion import (
    RapidCurriculumConfig,
    RapidDomainRandomizationConfig,
    RapidRewardConfig,
)
from .morphology_rewards import morphology_reward_profile


@dataclass(frozen=True)
class VelocityTrainingSpec:
    robot: str
    healthy_height_fraction: float
    min_root_up: float
    ctrl_dt: float


SPECS: dict[str, VelocityTrainingSpec] = {
    "ant": VelocityTrainingSpec(
        robot="ant",
        healthy_height_fraction=0.45,
        min_root_up=0.15,
        ctrl_dt=0.02,  # 50 Hz, matching the paper controller rate.
    ),
    "spinner": VelocityTrainingSpec(
        robot="spinner",
        healthy_height_fraction=0.30,
        min_root_up=0.00,
        ctrl_dt=0.02,
    ),
    "snake": VelocityTrainingSpec(
        robot="snake",
        healthy_height_fraction=0.20,
        min_root_up=-0.20,
        ctrl_dt=0.02,
    ),
    "crawler": VelocityTrainingSpec(
        robot="crawler",
        healthy_height_fraction=0.25,
        min_root_up=0.00,
        ctrl_dt=0.02,
    ),
    "biped": VelocityTrainingSpec(
        robot="biped",
        healthy_height_fraction=0.50,
        # Koseki et al. terminate after about 80 degrees of body tilt; cos(80deg) ~= 0.17.
        min_root_up=0.17,
        ctrl_dt=0.02,
    ),
}


def training_spec(robot_name: str) -> VelocityTrainingSpec:
    key = str(robot_name).strip().lower().replace("-", "_")
    if key not in SPECS:
        raise ValueError(
            f"High-speed policy training is currently implemented for {', '.join(SPECS)}; got {robot_name!r}."
        )
    return SPECS[key]


def make_velocity_env(
    robot_name: str,
    *,
    impl: str = "warp",
    command_cells: Optional[np.ndarray] = None,
    command_mask: Optional[np.ndarray] = None,
    curriculum_config: RapidCurriculumConfig | None = None,
    reward_config: RapidRewardConfig | None = None,
    domain_config: RapidDomainRandomizationConfig | None = None,
    ctrl_dt: Optional[float] = None,
    episode_length: int = 1000,
    command_hold_s: Optional[float] = None,
    grid_jitter: bool = True,
    enable_pushes: bool = True,
    naconmax: int = 512,
    njmax: int = 512,
):
    """Construct a Rapid-Locomotion-style MJX environment.

    ``command_cells`` is an ``(N,2)`` array of [v_x, omega_z] grid centers.
    When ``command_mask`` is supplied, only mask-true cells are sampled.  A
    fixed-size cell table plus a changing mask keeps JAX array shapes stable
    across host-side curriculum phases.  The lateral command is sampled
    separately, as in the paper.
    """
    try:
        import jax
        import jax.numpy as jp
        from ml_collections import config_dict
        import mujoco
        from mujoco import mjx
        from mujoco_playground._src import mjx_env
    except ImportError as exc:  # pragma: no cover - user's accelerator stack
        raise RuntimeError(
            "Policy training requires JAX, Brax, mujoco-mjx, and mujoco-playground."
        ) from exc

    info = find_classic_model(robot_name)
    spec = training_spec(info.name)
    curriculum = curriculum_config or RapidCurriculumConfig()
    rewards = reward_config or RapidRewardConfig()
    domain = domain_config or RapidDomainRandomizationConfig()
    reward_profile = morphology_reward_profile(info.name)
    xml_path = classic_xml_path(info)

    if command_cells is None:
        vx = np.arange(
            curriculum.initial_vx_min,
            curriculum.initial_vx_max + 0.5 * curriculum.grid_step_vx,
            curriculum.grid_step_vx,
            dtype=np.float32,
        )
        wz = np.arange(
            curriculum.initial_wz_min,
            curriculum.initial_wz_max + 0.5 * curriculum.grid_step_wz,
            curriculum.grid_step_wz,
            dtype=np.float32,
        )
        command_cells = np.asarray([(x, z) for x in vx for z in wz], dtype=np.float32)
    command_cells = np.asarray(command_cells, dtype=np.float32).reshape(-1, 2)
    if len(command_cells) == 0:
        raise ValueError("command_cells cannot be empty")
    if command_mask is None:
        command_mask = np.ones((len(command_cells),), dtype=bool)
    command_mask = np.asarray(command_mask, dtype=bool).reshape(-1)
    if command_mask.shape != (len(command_cells),):
        raise ValueError(
            f"command_mask must have shape ({len(command_cells)},), got {command_mask.shape}"
        )
    if not np.any(command_mask):
        raise ValueError("command_mask must enable at least one command cell")

    class VelocityTrackingEnv(mjx_env.MjxEnv):
        def __init__(self):
            mj_model = mujoco.MjModel.from_xml_path(str(xml_path))

            # MuJoCo-Warp does not support PGS.  Keep source XMLs portable by
            # switching any PGS model to Newton only for Warp training.
            if (
                str(impl).strip().lower() == "warp"
                and int(mj_model.opt.solver)
                == int(mujoco.mjtSolver.mjSOL_PGS)
            ):
                print("Warp does not support PGS; switching solver to Newton.")
                mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

            sim_dt = float(mj_model.opt.timestep)
            use_ctrl_dt = float(ctrl_dt if ctrl_dt is not None else spec.ctrl_dt)
            n_substeps = max(1, int(round(use_ctrl_dt / sim_dt)))
            use_ctrl_dt = n_substeps * sim_dt
            config = config_dict.create(
                ctrl_dt=use_ctrl_dt,
                sim_dt=sim_dt,
                episode_length=int(episode_length),
                action_repeat=1,
                impl=str(impl),
                naconmax=max(0, int(naconmax)),
                njmax=max(1, int(njmax)),
            )
            super().__init__(config)
            self._xml_path = str(xml_path)
            self._mj_model = mj_model
            self._mj_model.opt.timestep = self.sim_dt
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

            root_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_BODY, info.root_body_hint
            )
            if root_id < 0:
                for j in range(self._mj_model.njnt):
                    if int(self._mj_model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                        root_id = int(self._mj_model.jnt_bodyid[j])
                        break
            if root_id < 0:
                raise ValueError(f"{robot_name} does not expose a free-root locomotion body")
            self._root_body_id = int(root_id)

            qpos0 = np.asarray(self._mj_model.qpos0, dtype=np.float32)
            self._qpos0 = jp.asarray(qpos0)
            root_free_joint = -1
            for j in range(self._mj_model.njnt):
                if (
                    int(self._mj_model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
                    and int(self._mj_model.jnt_bodyid[j]) == self._root_body_id
                ):
                    root_free_joint = int(j)
                    break
            if root_free_joint < 0:
                raise ValueError(f"{robot_name} root body does not own a free joint")
            self._root_qpos_start = int(self._mj_model.jnt_qposadr[root_free_joint])
            self._root_qvel_start = int(self._mj_model.jnt_dofadr[root_free_joint])
            self._joint_qpos_start = self._root_qpos_start + 7
            self._joint_qvel_start = self._root_qvel_start + 6
            self._initial_root_height = float(qpos0[self._root_qpos_start + 2])

            # Map actuator order to joint coordinates.  Snake includes passive
            # wheel axles; regularizers should operate only on the eight learned
            # actuator DoFs, not on those passive coordinates.
            actuator_joint_ids = np.asarray(self._mj_model.actuator_trnid[:, 0], dtype=np.int32)
            if np.any(actuator_joint_ids < 0):
                raise ValueError(f"{robot_name} requires joint-transmission actuators")
            self._actuated_qpos_indices = jp.asarray(
                np.asarray([self._mj_model.jnt_qposadr[j] for j in actuator_joint_ids], dtype=np.int32)
            )
            self._actuated_qvel_indices = jp.asarray(
                np.asarray([self._mj_model.jnt_dofadr[j] for j in actuator_joint_ids], dtype=np.int32)
            )

            limited = np.asarray(self._mj_model.actuator_ctrllimited, dtype=bool)
            ranges = np.asarray(self._mj_model.actuator_ctrlrange, dtype=np.float32)
            low = np.full(self._mj_model.nu, -1.0, dtype=np.float32)
            high = np.full(self._mj_model.nu, 1.0, dtype=np.float32)
            if ranges.shape == (self._mj_model.nu, 2):
                low[limited] = ranges[limited, 0]
                high[limited] = ranges[limited, 1]
            self._ctrl_center = jp.asarray(0.5 * (low + high))
            self._ctrl_half_range = jp.asarray(0.5 * (high - low))

            self._command_cells = jp.asarray(command_cells)
            self._command_mask = jp.asarray(command_mask)
            self._vy_min = float(curriculum.vy_min)
            self._vy_max = float(curriculum.vy_max)
            self._jitter_vx = 0.5 * float(curriculum.grid_step_vx) if grid_jitter else 0.0
            self._jitter_wz = 0.5 * float(curriculum.grid_step_wz) if grid_jitter else 0.0
            hold_s = float(command_hold_s if command_hold_s is not None else curriculum.command_hold_s)
            self._hold_steps = max(1, int(round(hold_s / self.dt)))
            self._push_period = max(1, int(round(float(domain.push_interval_s) / self.dt)))
            self._enable_pushes = bool(enable_pushes)
            self._max_push_velocity_xy = float(domain.max_push_velocity_xy)

        @property
        def xml_path(self) -> str:
            return self._xml_path

        @property
        def action_size(self) -> int:
            return int(self._mj_model.nu)

        @property
        def mj_model(self):
            return self._mj_model

        @property
        def mjx_model(self):
            return self._mjx_model

        @property
        def root_body_id(self) -> int:
            return self._root_body_id

        def _sample_command(self, rng):
            rng_idx, rng_vy, rng_jitter = jax.random.split(rng, 3)
            # Uniform categorical over currently-active cells.  Keeping the
            # candidate table fixed-size avoids command-array shape changes as
            # the host curriculum expands.
            logits = jp.where(self._command_mask, 0.0, -1.0e9)
            idx = jax.random.categorical(rng_idx, logits)
            cell = self._command_cells[idx]
            vy = jax.random.uniform(rng_vy, (), minval=self._vy_min, maxval=self._vy_max)
            jitter = jax.random.uniform(
                rng_jitter,
                (2,),
                minval=jp.asarray([-self._jitter_vx, -self._jitter_wz]),
                maxval=jp.asarray([self._jitter_vx, self._jitter_wz]),
            )
            return jp.asarray([cell[0] + jitter[0], vy, cell[1] + jitter[1]])

        def _normalized_to_ctrl(self, action):
            action = jp.clip(action, -1.0, 1.0)
            return self._ctrl_center + self._ctrl_half_range * action

        def _root_rotation(self, data):
            return data.xmat[self._root_body_id]

        def _body_motion(self, data):
            rot = self._root_rotation(data)
            root_vel = data.qvel[self._root_qvel_start:self._root_qvel_start + 6]
            body_linear = rot.T @ root_vel[:3]
            body_angular = rot.T @ root_vel[3:6]
            return body_linear, body_angular

        def _projected_gravity(self, data):
            rot = self._root_rotation(data)
            return rot.T @ jp.asarray([0.0, 0.0, -1.0])

        def _get_obs(self, data, command, previous_action):
            body_linear, body_angular = self._body_motion(data)
            joint_pos = data.qpos[self._joint_qpos_start:] - self._qpos0[self._joint_qpos_start:]
            joint_vel = data.qvel[self._joint_qvel_start:]
            return jp.concatenate([
                body_linear,
                body_angular,
                self._projected_gravity(data),
                command,
                joint_pos,
                joint_vel,
                previous_action,
            ])

        def _health(self, data):
            up = data.xmat[self._root_body_id, 2, 2]
            min_height = float(spec.healthy_height_fraction) * self._initial_root_height
            healthy = (data.xpos[self._root_body_id, 2] > min_height) & (
                up > float(spec.min_root_up)
            )
            finite = jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all()
            return healthy & finite

        def reset(self, rng):
            rng, rng_q, rng_v, rng_cmd = jax.random.split(rng, 4)
            qpos = self._qpos0
            if self._mj_model.nq > self._joint_qpos_start:
                qpos = qpos.at[self._joint_qpos_start:].add(
                    0.05 * jax.random.uniform(
                        rng_q,
                        (self._mj_model.nq - self._joint_qpos_start,),
                        minval=-1.0,
                        maxval=1.0,
                    )
                )
            qpos = qpos.at[self._root_qpos_start + 2].add(0.01 * jax.random.normal(rng_q))
            # The released rapid-locomotion code initializes base velocity in
            # roughly [-0.5, 0.5]; use the same scale for all generalized vels.
            qvel = jax.random.uniform(
                rng_v, (self._mj_model.nv,), minval=-0.5, maxval=0.5
            )
            data = mjx_env.make_data(
                self.mj_model,
                qpos=qpos,
                qvel=qvel,
                ctrl=self._ctrl_center,
                impl=self.mjx_model.impl.value,
                naconmax=self._config.naconmax,
                njmax=self._config.njmax,
            )
            data = mjx.forward(self.mjx_model, data)
            command = self._sample_command(rng_cmd)
            previous_action = jp.zeros(self.action_size)
            info_dict = {
                "rng": rng,
                "command": command,
                "previous_action": previous_action,
                "previous_joint_velocity": data.qvel[self._joint_qvel_start:],
                "command_steps": jp.asarray(self._hold_steps, dtype=jp.int32),
                "step_count": jp.asarray(0, dtype=jp.int32),
                "time_out": jp.asarray(0.0, dtype=jp.float32),
            }
            metrics = {
                "tracking_lin_vel_per_step": jp.zeros(()),
                "tracking_ang_vel_per_step": jp.zeros(()),
                "lin_vel_z_penalty_per_step": jp.zeros(()),
                "ang_vel_xy_penalty_per_step": jp.zeros(()),
                "orientation_penalty_per_step": jp.zeros(()),
                "torque_penalty_per_step": jp.zeros(()),
                "dof_acc_penalty_per_step": jp.zeros(()),
                "action_rate_penalty_per_step": jp.zeros(()),
                "morph_spin_progress_per_step": jp.zeros(()),
                "morph_spin_speed_excess_per_step": jp.zeros(()),
                "morph_lateral_slip_per_step": jp.zeros(()),
                "morph_mechanical_power_per_step": jp.zeros(()),
                "morph_spatial_curvature_per_step": jp.zeros(()),
                "morph_upright_support_per_step": jp.zeros(()),
                "morph_height_deviation_per_step": jp.zeros(()),
            }
            obs = self._get_obs(data, command, previous_action)
            return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info_dict)

        def step(self, state, action):
            action = jp.clip(action, -1.0, 1.0)
            ctrl = self._normalized_to_ctrl(action)
            data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

            rng, rng_cmd, rng_push = jax.random.split(state.info["rng"], 3)
            step_count = state.info["step_count"] + 1
            if self._enable_pushes:
                do_push = (step_count % self._push_period) == 0
                push_xy = jax.random.uniform(
                    rng_push,
                    (2,),
                    minval=-self._max_push_velocity_xy,
                    maxval=self._max_push_velocity_xy,
                )
                qvel = data.qvel.at[:2].add(jp.where(do_push, push_xy, jp.zeros(2)))
                data = data.replace(qvel=qvel)

            body_linear, body_angular = self._body_motion(data)
            command = state.info["command"]
            lin_error2 = jp.sum((body_linear[:2] - command[:2]) ** 2)
            yaw_error2 = (body_angular[2] - command[2]) ** 2
            r_lin = jp.exp(-lin_error2 / float(rewards.tracking_sigma))
            r_yaw = jp.exp(-yaw_error2 / float(rewards.tracking_sigma))

            lin_vel_z_cost = body_linear[2] ** 2
            ang_vel_xy_cost = jp.sum(body_angular[:2] ** 2)
            projected_gravity = self._projected_gravity(data)
            orientation_cost = jp.sum(projected_gravity[:2] ** 2)
            # MuJoCo motor actuators expose actuator_force.  This
            # is the closest native-MuJoCo counterpart to the paper's torque term.
            torque_cost = jp.sum(data.actuator_force ** 2)
            joint_vel = data.qvel[self._joint_qvel_start:]
            dof_acc = (joint_vel - state.info["previous_joint_velocity"]) / self.dt
            dof_acc_cost = jp.sum(dof_acc ** 2)
            action_rate_cost = jp.sum((action - state.info["previous_action"]) ** 2)

            # Morphology-specific terms.  The branch is static at trace time, so
            # it adds no dynamic Python control flow inside the JIT.
            actuated_pos = data.qpos[self._actuated_qpos_indices]
            actuated_vel = data.qvel[self._actuated_qvel_indices]
            spin_progress = jp.zeros(())
            spin_speed_excess = jp.zeros(())
            lateral_slip_cost = body_linear[1] ** 2
            mechanical_power_cost = jp.mean(jp.abs(data.actuator_force * actuated_vel))
            spatial_curvature_cost = jp.zeros(())
            upright_support = jp.zeros(())
            height_deviation_cost = jp.zeros(())
            if spec.robot == "spinner":
                spin_vel = actuated_vel[jp.asarray([0, 2, 4, 6])]
                spin_activity = jp.tanh(jp.mean(jp.abs(spin_vel)) / 4.0)
                spin_progress = r_lin * spin_activity
                excess = jp.maximum(jp.abs(spin_vel) - float(reward_profile.spin_speed_limit), 0.0)
                spin_speed_excess = jp.mean(excess ** 2)
            elif spec.robot == "snake":
                # Baysal & Altas explicitly combine reference-speed accuracy
                # with mechanical power.  The passive wheel axles make lateral
                # undulation physically useful without forcing a hand-coded gait.
                pass
            elif spec.robot == "crawler":
                # Mishra et al. regularize strain gradients along a crawler.
                # Apply the same discrete second-difference idea to each row of
                # four paddles, encouraging a propagating rather than jerky gait.
                left = actuated_pos[:4]
                right = actuated_pos[4:]
                left_second = left[2:] - 2.0 * left[1:-1] + left[:-2]
                right_second = right[2:] - 2.0 * right[1:-1] + right[:-2]
                spatial_curvature_cost = 0.5 * (
                    jp.mean(left_second ** 2) + jp.mean(right_second ** 2)
                )
            elif spec.robot == "biped":
                # Koseki et al. use alive/support terms to keep a passive-dynamic
                # biped upright while rewarding forward locomotion.  Modern biped
                # RL also regularizes base height/orientation and joint power.
                root_up = data.xmat[self._root_body_id, 2, 2]
                root_height = data.xpos[self._root_body_id, 2]
                height_error = root_height - float(self._initial_root_height)
                height_deviation_cost = height_error ** 2
                upright_support = (
                    jp.clip(root_up, 0.0, 1.0)
                    * jp.exp(-(height_error / 0.12) ** 2)
                )

            reward = self.dt * (
                float(rewards.tracking_lin_vel) * r_lin
                + float(rewards.tracking_ang_vel) * r_yaw
                + float(rewards.lin_vel_z) * lin_vel_z_cost
                + float(rewards.ang_vel_xy) * ang_vel_xy_cost
                + float(rewards.orientation) * orientation_cost
                + float(rewards.torques) * torque_cost
                + float(rewards.dof_acc) * dof_acc_cost
                + float(rewards.action_rate) * action_rate_cost
                + float(reward_profile.spin_progress) * spin_progress
                + float(reward_profile.spin_speed_penalty) * spin_speed_excess
                + float(reward_profile.lateral_slip) * lateral_slip_cost
                + float(reward_profile.mechanical_power) * mechanical_power_cost
                + float(reward_profile.spatial_curvature) * spatial_curvature_cost
                + float(reward_profile.upright_support) * upright_support
                + float(reward_profile.height_deviation) * height_deviation_cost
            )
            if rewards.only_positive_rewards:
                reward = jp.maximum(reward, 0.0)

            healthy = self._health(data)
            done = (~healthy).astype(jp.float32)

            remaining = state.info["command_steps"] - 1
            should_resample = remaining <= 0
            sampled_command = self._sample_command(rng_cmd)
            next_command = jp.where(should_resample, sampled_command, command)
            next_remaining = jp.where(
                should_resample,
                jp.asarray(self._hold_steps, dtype=jp.int32),
                remaining,
            )

            info_dict = dict(state.info)
            info_dict.update({
                "rng": rng,
                "command": next_command,
                "previous_action": action,
                "previous_joint_velocity": joint_vel,
                "command_steps": next_remaining,
                "step_count": step_count,
            })
            metrics = dict(state.metrics)
            metrics.update({
                "tracking_lin_vel_per_step": r_lin,
                "tracking_ang_vel_per_step": r_yaw,
                "lin_vel_z_penalty_per_step": lin_vel_z_cost,
                "ang_vel_xy_penalty_per_step": ang_vel_xy_cost,
                "orientation_penalty_per_step": orientation_cost,
                "torque_penalty_per_step": torque_cost,
                "dof_acc_penalty_per_step": dof_acc_cost,
                "action_rate_penalty_per_step": action_rate_cost,
                "morph_spin_progress_per_step": spin_progress,
                "morph_spin_speed_excess_per_step": spin_speed_excess,
                "morph_lateral_slip_per_step": lateral_slip_cost,
                "morph_mechanical_power_per_step": mechanical_power_cost,
                "morph_spatial_curvature_per_step": spatial_curvature_cost,
                "morph_upright_support_per_step": upright_support,
                "morph_height_deviation_per_step": height_deviation_cost,
            })
            obs = self._get_obs(data, next_command, action)
            return mjx_env.State(data, obs, reward, done, metrics, info_dict)

    return VelocityTrackingEnv()


def make_domain_randomizer(
    *,
    friction_range: tuple[float, float] = (0.05, 4.0),
    motor_strength_range: tuple[float, float] = (0.90, 1.10),
):
    """Return a Brax/MJX per-world randomizer for safe batchable fields.

    The paper also randomizes payload mass, COM and restitution.  Those fields
    can require recomputing derived MuJoCo constants, so this fast MJX path
    deliberately randomizes the two directly batchable quantities that matter
    most to our simulator-only racing experiments: contact friction and motor
    strength.  The race-time online identifier separately handles mass/model
    mismatch.
    """
    try:
        import jax
        import jax.numpy as jp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Domain randomization requires JAX") from exc

    f_lo, f_hi = map(float, friction_range)
    m_lo, m_hi = map(float, motor_strength_range)

    def randomize(model, rng):
        @jax.vmap
        def sample_one(key):
            key_f, key_m = jax.random.split(key)
            friction_scale = jax.random.uniform(key_f, (), minval=f_lo, maxval=f_hi)
            motor_scale = jax.random.uniform(key_m, (), minval=m_lo, maxval=m_hi)
            friction = model.geom_friction * friction_scale
            gain = model.actuator_gainprm * motor_scale
            return friction, gain

        friction, gain = sample_one(rng)
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({"geom_friction": 0, "actuator_gainprm": 0})
        randomized = model.tree_replace({
            "geom_friction": jp.asarray(friction),
            "actuator_gainprm": jp.asarray(gain),
        })
        return randomized, in_axes

    return randomize


__all__ = [
    "VelocityTrainingSpec",
    "SPECS",
    "training_spec",
    "make_velocity_env",
    "make_domain_randomizer",
]
