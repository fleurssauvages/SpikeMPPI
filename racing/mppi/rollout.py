from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from typing import Sequence
import numpy as np


@dataclass
class RolloutCostConfig:
    hard_collision_clearance: float = 0.02
    fall_height_fraction: float = 0.45
    min_root_up: float = 0.15
    upright_weight: float = 0.05
    control_deviation_weight: float = 1e-4




class NativeRolloutBatcher:
    """Fast batched open-loop rollouts using MuJoCo's native C++ rollout module.

    This backend removes the Python loop over MPPI samples and keeps a persistent
    native thread pool.  For the classic Ant/Humanoid models, whose root body is
    attached to world by a free joint, rollout costs are computed directly from
    the returned qpos trajectories in vectorized NumPy.  Unsupported root layouts
    fall back to the legacy Python evaluator.
    """

    def __init__(self, robot, *, workers: int = 0, batch_hint: int = 128) -> None:
        from mujoco import rollout as mj_rollout

        self.robot = robot
        self.mj = robot.mujoco
        self.model = robot.model
        cpu = max(1, int(os.cpu_count() or 1))
        requested = int(workers)
        if requested <= 0:
            requested = min(cpu, max(1, int(batch_hint)))
        self.nthread = max(1, requested)
        # nthread=0 executes on the calling thread and avoids thread-pool overhead.
        runner_threads = self.nthread if self.nthread > 1 else 0
        self.runner = mj_rollout.Rollout(nthread=runner_threads)
        self.data = (
            [self.mj.MjData(self.model) for _ in range(self.nthread)]
            if self.nthread > 1
            else self.mj.MjData(self.model)
        )
        self.state_spec = self.mj.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = int(self.mj.mj_stateSize(self.model, self.state_spec))
        self._state_scratch = self.mj.MjData(self.model)
        self._root_qadr = self._find_world_free_root_qadr()
        self.supports_vectorized_cost = self._root_qadr is not None
        self._ctrl_scale = np.maximum(robot.control_scale(), 1e-6)
        self._ctrl_low, self._ctrl_high = robot.control_bounds()

    def close(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
            self.runner = None

    def __del__(self):
        self.close()

    def _find_world_free_root_qadr(self) -> int | None:
        m = self.model
        root = int(self.robot.root_body_id)
        # Direct qpos extraction is exact only when the free root is attached to world.
        if int(m.body_parentid[root]) != 0:
            return None
        free_type = int(self.mj.mjtJoint.mjJNT_FREE)
        for j in range(int(m.njnt)):
            if int(m.jnt_bodyid[j]) == root and int(m.jnt_type[j]) == free_type:
                return int(m.jnt_qposadr[j])
        return None

    def snapshot_to_state(self, snapshot) -> np.ndarray:
        d = self._state_scratch
        d.time = float(snapshot.time)
        d.qpos[:] = snapshot.qpos
        d.qvel[:] = snapshot.qvel
        if self.model.na:
            d.act[:] = snapshot.act
        out = np.empty(self.nstate, dtype=np.float64)
        self.mj.mj_getState(self.model, d, out, self.state_spec)
        return out

    def snapshots_to_states(self, snapshots: Sequence) -> np.ndarray:
        return np.stack([self.snapshot_to_state(s) for s in snapshots], axis=0)

    def _root_from_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._root_qadr is None:
            raise RuntimeError("root pose is not directly available from qpos")
        # mjSTATE_FULLPHYSICS is ordered as time, qpos, qvel, act, ...
        q = 1 + int(self._root_qadr)
        pos = state[..., q:q + 3]
        quat = state[..., q + 3:q + 7]
        # MuJoCo quaternion convention is [w, x, y, z]. R_zz = 1 - 2(x^2+y^2).
        up = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
        return pos[..., :2], pos[..., 2], up

    def rollout_states(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        state, _ = self.runner.rollout(
            self.model,
            self.data,
            np.asarray(initial_state, dtype=np.float64),
            np.ascontiguousarray(controls, dtype=np.float64),
        )
        return state

    def evaluate(
        self,
        start_snapshot,
        control_batch: np.ndarray,
        track,
        current_s: float,
        *,
        control_substeps: int,
        nominal_controls: np.ndarray,
        cost_cfg: RolloutCostConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.supports_vectorized_cost:
            raise RuntimeError("vectorized rollout cost is unsupported for this root layout")
        batch = np.asarray(control_batch, dtype=np.float64)
        n, h, _ = batch.shape
        substeps = max(1, int(control_substeps))
        # Repeat each control over the underlying physics steps in native code.
        expanded = np.repeat(batch, substeps, axis=1)
        initial = self.snapshot_to_state(start_snapshot)[None, :]
        states = self.rollout_states(initial, expanded)
        sampled = states[:, substeps - 1::substeps, :]
        positions, height, up = self._root_from_state(sampled)

        flat_s, flat_d2 = track.project(positions.reshape(-1, 2))
        progress_s = np.asarray(flat_s, dtype=np.float64).reshape(n, h)
        d2 = np.asarray(flat_d2, dtype=np.float64).reshape(n, h)
        allowed = max(0.0, 0.5 * float(track.road_width) - float(cost_cfg.hard_collision_clearance))
        failure = d2 > allowed * allowed
        failure |= height < float(cost_cfg.fall_height_fraction) * max(self.robot.initial_root_height, 1e-6)
        failure |= up < float(cost_cfg.min_root_up)

        # MuJoCo rollout fills the tail after divergence with an unchanged state.
        times = sampled[..., 0]
        prev_t = np.concatenate(
            [np.full((n, 1), float(start_snapshot.time), dtype=np.float64), times[:, :-1]], axis=1
        )
        failure |= times <= prev_t + 1e-15

        prev_s = np.concatenate(
            [np.full((n, 1), float(current_s), dtype=np.float64), progress_s[:, :-1]], axis=1
        )
        ds = progress_s - prev_s
        half = 0.5 * float(track.length)
        ds = np.where(ds > half, ds - float(track.length), ds)
        ds = np.where(ds < -half, ds + float(track.length), ds)

        alive = np.logical_and.accumulate(~failure, axis=1)
        cumulative_alive = np.cumsum(np.where(alive, ds, 0.0), axis=1)
        terminal_progress = cumulative_alive[:, -1]
        failed = np.any(failure, axis=1)

        costs = np.full(n, math.inf, dtype=np.float64)
        finite = ~failed
        if np.any(finite):
            cumulative = np.cumsum(ds[finite], axis=1)
            progress_cost = -np.sum(cumulative, axis=1) / max(1, h)
            upright_cost = float(cost_cfg.upright_weight) * np.sum((1.0 - up[finite]) ** 2, axis=1)
            du = batch[finite] - np.asarray(nominal_controls, dtype=np.float64)[None, :, :]
            control_cost = float(cost_cfg.control_deviation_weight) * np.sum(
                np.mean((du / self._ctrl_scale[None, None, :]) ** 2, axis=2), axis=1
            )
            costs[finite] = progress_cost + upright_cost + control_cost
        return positions, costs, terminal_progress, failed

    def estimate_joint_task_jacobians(
        self,
        snapshots: Sequence,
        nominal_controls: np.ndarray,
        *,
        control_substeps: int,
        lookahead_steps: int,
        epsilon_fraction: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.supports_vectorized_cost:
            raise RuntimeError("vectorized Jacobians are unsupported for this root layout")
        u_nom = np.asarray(nominal_controls, dtype=np.float64)
        h, nu = u_nom.shape
        jac = np.zeros((h, 2, nu), dtype=np.float64)
        endpoints = np.empty((h, 2), dtype=np.float64)
        scale = np.maximum(self.robot.control_scale(fraction=1.0), 1e-6)
        substeps = max(1, int(control_substeps))
        L = max(1, int(lookahead_steps))
        initial_all = self.snapshots_to_states(snapshots)

        # Near the horizon the legacy implementation uses shorter futures. Group
        # by that length so every native batch remains exactly equivalent.
        for ell in range(1, L + 1):
            tids = [t for t in range(h) if min(L, h - t) == ell]
            if not tids:
                continue
            init_rows = []
            control_rows = []
            meta = []
            for t in tids:
                future = u_nom[t:t + ell]
                init_rows.append(initial_all[t])
                control_rows.append(future)
                meta.append((t, -1, 1.0))
                for j in range(nu):
                    eps = max(1e-7, float(epsilon_fraction) * float(scale[j]))
                    plus = future.copy()
                    plus[0, j] += eps
                    plus[0] = np.clip(plus[0], self._ctrl_low, self._ctrl_high)
                    denom = float(plus[0, j] - future[0, j])
                    init_rows.append(initial_all[t])
                    control_rows.append(plus)
                    meta.append((t, j, denom))

            init_batch = np.asarray(init_rows, dtype=np.float64)
            controls = np.repeat(np.asarray(control_rows, dtype=np.float64), substeps, axis=1)
            out = self.rollout_states(init_batch, controls)
            xy, _, _ = self._root_from_state(out[:, -1, :])
            base_by_t = {}
            for k, (t, j, denom) in enumerate(meta):
                if j < 0:
                    endpoints[t] = xy[k]
                    base_by_t[t] = xy[k]
                elif abs(denom) > 1e-12:
                    jac[t, :, j] = (xy[k] - base_by_t[t]) / denom
        return jac, endpoints


@dataclass
class NominalRollout:
    controls: np.ndarray
    snapshots: list
    positions: np.ndarray
    progress_s: np.ndarray
    cumulative_progress: np.ndarray


def rollout_policy_nominal(
    robot,
    start_snapshot,
    policy,
    track,
    prior,
    current_s: float,
    *,
    horizon: int,
    control_substeps: int,
) -> NominalRollout:
    """Roll out the robot's default policy in native MuJoCo to seed the nominal."""
    d = robot.new_data(start_snapshot)
    controls = np.empty((horizon, robot.nu), dtype=np.float64)
    snapshots = []
    positions = np.empty((horizon, 2), dtype=np.float64)
    progress_s = np.empty(horizon, dtype=np.float64)
    cumulative = np.empty(horizon, dtype=np.float64)
    s_prev = float(current_s)
    cum = 0.0
    for t in range(horizon):
        snapshots.append(robot.snapshot(d))
        s_now, _ = track.project(robot.xy(d))
        u = np.asarray(policy.action(robot, d, track=track, prior=prior, current_s=float(s_now)), dtype=np.float64)
        controls[t] = robot.clip_ctrl(u)
        robot.step_control(controls[t], substeps=control_substeps, data=d)
        p = robot.xy(d)
        s_new, _ = track.project(p)
        cum += track.signed_progress_delta(float(s_new), s_prev)
        s_prev = float(s_new)
        positions[t] = p
        progress_s[t] = float(s_new)
        cumulative[t] = cum
    return NominalRollout(controls, snapshots, positions, progress_s, cumulative)


def rollout_controls(
    robot,
    start_snapshot,
    controls: np.ndarray,
    track,
    current_s: float,
    *,
    control_substeps: int,
    nominal_controls: np.ndarray | None = None,
    cost_cfg: RolloutCostConfig | None = None,
) -> tuple[np.ndarray, float, float, bool]:
    """Evaluate one direct joint-control sequence entirely in native MuJoCo."""
    cfg = cost_cfg or RolloutCostConfig()
    d = robot.new_data(start_snapshot)
    controls = np.asarray(controls, dtype=np.float64)
    nominal_controls = controls if nominal_controls is None else np.asarray(nominal_controls, dtype=np.float64)
    positions = np.empty((len(controls), 2), dtype=np.float64)
    s_prev = float(current_s)
    cumulative = 0.0
    prefix_sum = 0.0
    control_cost = 0.0
    upright_cost = 0.0
    allowed = max(0.0, 0.5 * float(track.road_width) - float(cfg.hard_collision_clearance))
    off_track = False
    ctrl_scale = np.maximum(robot.control_scale(), 1e-6) if robot.nu else None

    for t, u in enumerate(controls):
        robot.step_control(u, substeps=control_substeps, data=d)
        p = robot.xy(d)
        positions[t] = p
        s_new, d2 = track.project(p)
        if float(d2) > allowed * allowed:
            off_track = True
            break
        if robot.root_height(d) < cfg.fall_height_fraction * max(robot.initial_root_height, 1e-6):
            off_track = True
            break
        up = robot.root_up(d)
        if up < cfg.min_root_up:
            off_track = True
            break
        ds = track.signed_progress_delta(float(s_new), s_prev)
        cumulative += ds
        s_prev = float(s_new)
        prefix_sum += cumulative
        upright_cost += float(cfg.upright_weight) * (1.0 - up) ** 2
        if robot.nu:
            du = u - nominal_controls[min(t, len(nominal_controls) - 1)]
            control_cost += float(cfg.control_deviation_weight) * float(np.mean((du / ctrl_scale) ** 2))

    if off_track:
        return positions, math.inf, cumulative, True
    progress_cost = -prefix_sum / max(1, len(controls))
    return positions, float(progress_cost + upright_cost + control_cost), cumulative, False


def evaluate_control_batch(
    robot,
    start_snapshot,
    control_batch: np.ndarray,
    track,
    current_s: float,
    *,
    control_substeps: int,
    nominal_controls: np.ndarray,
    cost_cfg: RolloutCostConfig,
    workers: int = 0,
    native_batcher: NativeRolloutBatcher | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if native_batcher is not None and native_batcher.supports_vectorized_cost:
        return native_batcher.evaluate(
            start_snapshot, control_batch, track, current_s,
            control_substeps=control_substeps,
            nominal_controls=nominal_controls,
            cost_cfg=cost_cfg,
        )
    batch = np.asarray(control_batch, dtype=np.float64)
    n, h, _ = batch.shape
    positions = np.empty((n, h, 2), dtype=np.float64)
    costs = np.empty(n, dtype=np.float64)
    progress = np.empty(n, dtype=np.float64)
    failed = np.zeros(n, dtype=bool)

    def one(i: int):
        pos, cost, prog, fail = rollout_controls(
            robot,
            start_snapshot,
            batch[i],
            track,
            current_s,
            control_substeps=control_substeps,
            nominal_controls=nominal_controls,
            cost_cfg=cost_cfg,
        )
        return i, pos, cost, prog, fail

    if int(workers) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            iterator = pool.map(one, range(n))
            for i, pos, cost, prog, fail in iterator:
                positions[i] = pos
                costs[i] = cost
                progress[i] = prog
                failed[i] = fail
    else:
        for i in range(n):
            _, pos, cost, prog, fail = one(i)
            positions[i] = pos
            costs[i] = cost
            progress[i] = prog
            failed[i] = fail
    return positions, costs, progress, failed


def _endpoint_from_snapshot(robot, snapshot, controls: np.ndarray, control_substeps: int) -> np.ndarray:
    d = robot.new_data(snapshot)
    for u in controls:
        robot.step_control(u, substeps=control_substeps, data=d)
    return robot.xy(d)


def estimate_joint_task_jacobians(
    robot,
    snapshots: Sequence,
    nominal_controls: np.ndarray,
    *,
    control_substeps: int,
    lookahead_steps: int,
    epsilon_fraction: float = 1e-3,
    centered: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct MuJoCo finite-difference J_t = d p_xy(t+L) / d ctrl_t.

    This is model agnostic and includes contacts, actuator dynamics and the full
    robot morphology. It replaces the two-control A/B projection in the source
    controller with a joint-dimensional raw local sensitivity.
    """
    u_nom = np.asarray(nominal_controls, dtype=np.float64)
    h, nu = u_nom.shape
    jac = np.zeros((h, 2, nu), dtype=np.float64)
    endpoints = np.empty((h, 2), dtype=np.float64)
    scale = np.maximum(robot.control_scale(fraction=1.0), 1e-6)
    L = max(1, int(lookahead_steps))

    for t in range(h):
        end = min(h, t + L)
        future = u_nom[t:end]
        base = _endpoint_from_snapshot(robot, snapshots[t], future, control_substeps)
        endpoints[t] = base
        for j in range(nu):
            eps = max(1e-7, float(epsilon_fraction) * float(scale[j]))
            plus = future.copy()
            plus[0, j] += eps
            plus[0] = robot.clip_ctrl(plus[0])
            p_plus = _endpoint_from_snapshot(robot, snapshots[t], plus, control_substeps)
            if centered:
                minus = future.copy()
                minus[0, j] -= eps
                minus[0] = robot.clip_ctrl(minus[0])
                p_minus = _endpoint_from_snapshot(robot, snapshots[t], minus, control_substeps)
                denom = float(plus[0, j] - minus[0, j])
                if abs(denom) > 1e-12:
                    jac[t, :, j] = (p_plus - p_minus) / denom
            else:
                denom = float(plus[0, j] - future[0, j])
                if abs(denom) > 1e-12:
                    jac[t, :, j] = (p_plus - base) / denom
    return jac, endpoints


def refine_policy_nominal(
    robot,
    start_snapshot,
    policy_rollout: NominalRollout,
    track,
    prior,
    *,
    control_substeps: int,
    lookahead_steps: int,
    iterations: int = 1,
    damping: float = 1e-4,
    step_size: float = 0.35,
    max_control_step_fraction: float = 0.15,
    epsilon_fraction: float = 1e-3,
    native_batcher: NativeRolloutBatcher | None = None,
) -> tuple[NominalRollout, np.ndarray, np.ndarray]:
    """Policy-seeded iLQR-like task-space refinement.

    A full contact-rich iLQR implementation that works identically for every
    robot actuator model is brittle. This uses the same local principle:
    linearize MuJoCo around the policy rollout, solve a damped least-squares
    task correction, update the joint sequence, and re-rollout. The resulting
    sensitivity matrices are also the exact matrices consumed by SPG.
    """
    controls = np.asarray(policy_rollout.controls, dtype=np.float64).copy()
    current = policy_rollout
    jac = np.zeros((len(controls), 2, robot.nu), dtype=np.float64)
    endpoints = current.positions.copy()
    max_step = max_control_step_fraction * np.maximum(robot.control_scale(fraction=1.0), 1e-6)

    def _estimate(snaps, ctrls):
        if native_batcher is not None and native_batcher.supports_vectorized_cost:
            return native_batcher.estimate_joint_task_jacobians(
                snaps, ctrls,
                control_substeps=control_substeps,
                lookahead_steps=lookahead_steps,
                epsilon_fraction=epsilon_fraction,
            )
        return estimate_joint_task_jacobians(
            robot, snaps, ctrls,
            control_substeps=control_substeps,
            lookahead_steps=lookahead_steps,
            epsilon_fraction=epsilon_fraction,
        )

    for _ in range(max(0, int(iterations))):
        jac, endpoints = _estimate(current.snapshots, controls)
        for t in range(len(controls)):
            s_target, _ = track.project(endpoints[t])
            mean, _ = prior.sample(track, float(s_target))
            error = np.asarray(mean, dtype=np.float64) - endpoints[t]
            j = jac[t]
            pinv = j.T @ np.linalg.inv(j @ j.T + float(damping) * np.eye(2))
            du = float(step_size) * (pinv @ error)
            du = np.clip(du, -max_step, max_step)
            controls[t] = robot.clip_ctrl(controls[t] + du)

        # Re-rollout the corrected open-loop sequence, retaining snapshots for
        # the next linearization pass.
        d = robot.new_data(start_snapshot)
        snapshots = []
        positions = np.empty_like(current.positions)
        progress_s = np.empty_like(current.progress_s)
        cumulative = np.empty_like(current.cumulative_progress)
        s_prev = float(track.project(robot.xy(d))[0])
        cum = 0.0
        for t, u in enumerate(controls):
            snapshots.append(robot.snapshot(d))
            robot.step_control(u, substeps=control_substeps, data=d)
            p = robot.xy(d)
            s_new, _ = track.project(p)
            cum += track.signed_progress_delta(float(s_new), s_prev)
            s_prev = float(s_new)
            positions[t] = p
            progress_s[t] = float(s_new)
            cumulative[t] = cum
        current = NominalRollout(controls.copy(), snapshots, positions, progress_s, cumulative)

    # Ensure sensitivities correspond to the final nominal.
    jac, endpoints = _estimate(current.snapshots, current.controls)
    return current, jac, endpoints
