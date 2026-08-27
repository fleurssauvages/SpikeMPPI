from __future__ import annotations

"""MJX/Warp high-speed command-conditioned locomotion environments.

The reward, 50 Hz control loop, 10 s command hold, push disturbances and
(v_x, yaw-rate) curriculum follow the morphology-independent parts of
Margolis et al., *Rapid Locomotion via Reinforcement Learning*.

The classic MuJoCo Ant/Humanoid use direct actuator controls rather than the
Mini Cheetah's joint-position/PD interface, so foot-air-time and collision
terms that depend on the Mini-Cheetah contact topology are intentionally not
copied verbatim.
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
    "humanoid": VelocityTrainingSpec(
        robot="humanoid",
        healthy_height_fraction=0.55,
        min_root_up=0.25,
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

    ``command_cells`` is an ``(N,2)`` array of active [v_x, omega_z] grid
    centers.  The lateral command is sampled separately, as in the paper.
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

    class VelocityTrackingEnv(mjx_env.MjxEnv):
        def __init__(self):
            mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
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
            self._initial_root_height = float(qpos0[2])
            # Classic Ant/Humanoid both use a leading free joint.
            self._joint_qpos_start = 7
            self._joint_qvel_start = 6

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
            idx = jax.random.randint(rng_idx, (), 0, self._command_cells.shape[0])
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
            body_linear = rot.T @ data.qvel[:3]
            body_angular = rot.T @ data.qvel[3:6]
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
            qpos = qpos.at[2].add(0.01 * jax.random.normal(rng_q))
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
                "torque_penalty_per_step": jp.zeros(()),
                "dof_acc_penalty_per_step": jp.zeros(()),
                "action_rate_penalty_per_step": jp.zeros(()),
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
            # Classic Ant/Humanoid motor actuators expose actuator_force.  This
            # is the closest native-MuJoCo counterpart to the paper's torque term.
            torque_cost = jp.sum(data.actuator_force ** 2)
            joint_vel = data.qvel[self._joint_qvel_start:]
            dof_acc = (joint_vel - state.info["previous_joint_velocity"]) / self.dt
            dof_acc_cost = jp.sum(dof_acc ** 2)
            action_rate_cost = jp.sum((action - state.info["previous_action"]) ** 2)

            reward = self.dt * (
                float(rewards.tracking_lin_vel) * r_lin
                + float(rewards.tracking_ang_vel) * r_yaw
                + float(rewards.lin_vel_z) * lin_vel_z_cost
                + float(rewards.ang_vel_xy) * ang_vel_xy_cost
                + float(rewards.torques) * torque_cost
                + float(rewards.dof_acc) * dof_acc_cost
                + float(rewards.action_rate) * action_rate_cost
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
                "torque_penalty_per_step": torque_cost,
                "dof_acc_penalty_per_step": dof_acc_cost,
                "action_rate_penalty_per_step": action_rate_cost,
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
