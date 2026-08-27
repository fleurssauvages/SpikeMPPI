from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np


@dataclass
class RolloutCostConfig:
    hard_collision_clearance: float = 0.02
    fall_height_fraction: float = 0.45
    min_root_up: float = 0.15
    upright_weight: float = 0.05
    control_deviation_weight: float = 1e-4


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
            scale = np.maximum(robot.control_scale(), 1e-6)
            control_cost += float(cfg.control_deviation_weight) * float(np.mean((du / scale) ** 2))

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    for _ in range(max(0, int(iterations))):
        jac, endpoints = estimate_joint_task_jacobians(
            robot,
            current.snapshots,
            controls,
            control_substeps=control_substeps,
            lookahead_steps=lookahead_steps,
            epsilon_fraction=epsilon_fraction,
        )
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
    jac, endpoints = estimate_joint_task_jacobians(
        robot,
        current.snapshots,
        current.controls,
        control_substeps=control_substeps,
        lookahead_steps=lookahead_steps,
        epsilon_fraction=epsilon_fraction,
    )
    return current, jac, endpoints
