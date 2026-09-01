from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import math
import time
import numpy as np

from .lbps import optimize_lbps_temperature, weighted_control_sequence
from .rollout import (
    RolloutCostConfig,
    NativeRolloutBatcher,
    evaluate_control_batch,
    rollout_policy_nominal,
    rollout_control_nominal,
    refine_policy_nominal,
)
from .spg import build_spg_factors, sample_joint_noise
from .fast_kernels import NUMBA_AVAILABLE, spg_dense_project_and_smooth, lbps_optimize_fast


class ControllerVariant(str, Enum):
    POLICY_NOMINAL = "policy_nominal"
    STANDARD_MPPI = "standard_mppi"
    SENSITIVITY_PROJECTED_GAUSSIAN_MPPI = "sensitivity_projected_gaussian_prior_mppi"


@dataclass
class ControllerConfig:
    control_dt: float = 0.02
    horizon: int = 50
    num_rollouts: int = 32
    lambda_temperature: float = 1.0
    adaptive_temperature_lbps: bool = True
    lbps_delta: float = 0.9
    lbps_optimizer_iterations: int = 32
    temporal_noise_smoothing: float = 0.25

    # Direct joint-space proposal.
    joint_noise_fraction: float = 0.08
    unlimited_control_span: float = 1.0

    # Policy-seeded local iLQR-style nominal construction.
    nominal_refine_iterations: int = 0
    nominal_refine_damping: float = 1e-4
    nominal_refine_step_size: float = 0.35
    nominal_max_step_fraction: float = 0.15
    sensitivity_epsilon_fraction: float = 1e-3

    # SPG projection, now J in R^(2 x model.nu).
    spg_lookahead_steps: int = 3
    spg_pseudoinverse_damping: float = 1e-6
    spg_covariance_jitter: float = 1e-8
    spg_null_std_scale: float = 0.15
    spg_mix: float = 1.0
    # Receding-horizon SPG sensitivities shift naturally with the plan.  A value
    # of 1 preserves the original full finite-difference refresh every tick.
    # Values >1 reuse the shifted Jacobian between full refreshes; an optional
    # prefix can still be refreshed every tick for near-term accuracy.
    spg_jacobian_refresh_interval: int = 4
    spg_jacobian_refresh_prefix: int = 2

    # Racing rollout constraints/cost.
    hard_collision_clearance: float = 0.02
    fall_height_fraction: float = 0.45
    min_root_up: float = 0.15
    upright_weight: float = 0.05
    control_deviation_weight: float = 1e-4

    # Dense task-transfer shaping for push_box. Ignored for ordinary racing.
    # The box term is primary; robot motion and robot-box approach only remove
    # the zero-signal plateau before the first contact.
    box_progress_weight: float = 1.0
    robot_progress_weight: float = 0.35
    robot_box_approach_weight: float = 1.00
    # Keep pushing in the quasi-planar regime. Ballistic/tipped box candidates
    # are not useful solutions to a ground-pushing task.
    box_max_lift: float = 0.12
    box_min_up: float = 0.75
    rollout_workers: int = 16
    # Prefer the allocation-light fused evaluator when it has been built, while
    # remaining runnable on installations that only provide mujoco.rollout.
    rollout_backend: str = "auto"
    rollout_chunk_size: int = 0

    # Standard receding-horizon warm start. After the first update, shift the
    # previous optimized sequence instead of doing H synchronous policy calls.
    warm_start: bool = True

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.num_rollouts = max(1, int(self.num_rollouts))
        self.lbps_optimizer_iterations = max(8, int(self.lbps_optimizer_iterations))
        self.spg_lookahead_steps = max(1, int(self.spg_lookahead_steps))
        self.spg_jacobian_refresh_interval = max(0, int(self.spg_jacobian_refresh_interval))
        self.spg_jacobian_refresh_prefix = max(0, int(self.spg_jacobian_refresh_prefix))
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if self.lambda_temperature <= 0.0:
            raise ValueError("lambda_temperature must be positive")
        if not 0.0 < self.lbps_delta < 1.0:
            raise ValueError("lbps_delta must lie in (0, 1)")
        if not 0.0 <= self.temporal_noise_smoothing < 1.0:
            raise ValueError("temporal_noise_smoothing must lie in [0, 1)")
        if not 0.0 <= self.spg_mix <= 1.0:
            raise ValueError("spg_mix must lie in [0, 1]")
        if self.spg_null_std_scale < 0.0:
            raise ValueError("spg_null_std_scale must be nonnegative")
        if self.spg_pseudoinverse_damping < 0.0 or self.spg_covariance_jitter < 0.0:
            raise ValueError("SPG damping and covariance jitter must be nonnegative")
        if self.sensitivity_epsilon_fraction <= 0.0:
            raise ValueError("sensitivity_epsilon_fraction must be positive")
        if self.box_progress_weight < 0.0 or self.robot_progress_weight < 0.0 or self.robot_box_approach_weight < 0.0:
            raise ValueError("push-task reward weights must be nonnegative")
        if self.box_max_lift < 0.0:
            raise ValueError("box_max_lift must be nonnegative")
        if not -1.0 <= self.box_min_up <= 1.0:
            raise ValueError("box_min_up must lie in [-1, 1]")
        self.rollout_chunk_size = max(0, int(self.rollout_chunk_size))
        self.rollout_backend = str(self.rollout_backend).strip().lower()
        if self.rollout_backend not in {"auto", "fused", "native", "python"}:
            raise ValueError("rollout_backend must be 'auto', 'fused', 'native' or 'python'")


class JointMPPIController:
    """Native-MuJoCo, direct-joint LBPS-MPPI with SPG.

    This follows the uploaded controller's structure, but the control vector is
    no longer fixed to two vehicle controls. For every classic MuJoCo robot:

        u_t == MjData.ctrl in R^(model.nu)

    The nominal is seeded by that robot's locomotion policy, locally refined around
    the policy rollout, and SPG maps the 2-D trajectory prior through the raw
    MuJoCo joint-to-planar-position sensitivity.
    """

    def __init__(
        self,
        robot,
        track,
        prior,
        policy,
        cfg: ControllerConfig,
        *,
        variant: ControllerVariant | str = ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI,
        seed: int = 1,
    ) -> None:
        self.robot = robot
        self.track = track
        self.prior = prior
        self.policy = policy
        self.cfg = cfg
        self.variant = ControllerVariant(variant)
        self.rng = np.random.default_rng(int(seed))
        ratio = float(cfg.control_dt) / max(float(robot.physics_dt), 1e-12)
        self.control_substeps = max(1, int(round(ratio)))
        actual = self.control_substeps * float(robot.physics_dt)
        if abs(actual - float(cfg.control_dt)) > 0.25 * float(robot.physics_dt):
            raise ValueError(
                f"control_dt={cfg.control_dt:g} is not close to an integer multiple of "
                f"MuJoCo timestep={robot.physics_dt:g}; nearest is {actual:g}."
            )
        self.cost_cfg = RolloutCostConfig(
            hard_collision_clearance=cfg.hard_collision_clearance,
            fall_height_fraction=cfg.fall_height_fraction,
            min_root_up=cfg.min_root_up,
            upright_weight=cfg.upright_weight,
            control_deviation_weight=cfg.control_deviation_weight,
            box_progress_weight=cfg.box_progress_weight,
            robot_progress_weight=cfg.robot_progress_weight,
            robot_box_approach_weight=cfg.robot_box_approach_weight,
            box_max_lift=cfg.box_max_lift,
            box_min_up=cfg.box_min_up,
        )
        self.native_batcher = None
        self._previous_plan: np.ndarray | None = None
        self._previous_jacobian: np.ndarray | None = None
        self._spg_updates = 0
        self._last_spg_refresh_mode = "full"
        self._ctrl_low, self._ctrl_high = robot.control_bounds(cfg.unlimited_control_span)
        self._joint_std = robot.control_scale(
            fraction=cfg.joint_noise_fraction,
            unlimited_span=cfg.unlimited_control_span,
        )
        self._spg_fast_workspace: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._standard_workspace: tuple[np.ndarray, np.ndarray] | None = None
        if cfg.rollout_backend in {"auto", "native", "fused"}:
            try:
                self.native_batcher = NativeRolloutBatcher(
                    robot, workers=cfg.rollout_workers, batch_hint=cfg.num_rollouts,
                    chunk_size=cfg.rollout_chunk_size,
                    fused=(cfg.rollout_backend in {"auto", "fused"}),
                )
            except Exception:
                if cfg.rollout_backend == "fused":
                    # Fused mode is explicit: never silently benchmark the slower
                    # stock path when the extension is missing or incompatible.
                    raise
                # Auto mode prefers fused but falls back to the stock persistent
                # native pool. Native mode keeps the legacy Python evaluator as
                # its compatibility fallback.
                if cfg.rollout_backend == "auto":
                    try:
                        self.native_batcher = NativeRolloutBatcher(
                            robot, workers=cfg.rollout_workers,
                            batch_hint=cfg.num_rollouts,
                            chunk_size=cfg.rollout_chunk_size,
                            fused=False,
                        )
                    except Exception:
                        self.native_batcher = None
                else:
                    self.native_batcher = None

    @property
    def rollout_backend_name(self) -> str:
        if self.native_batcher is not None and self.native_batcher.supports_vectorized_cost:
            suffix = f"/chunk{self.native_batcher.chunk_size}" if self.native_batcher.chunk_size > 0 else ""
            prefix = "fused" if self.native_batcher.uses_fused else "native"
            return f"{prefix}/{self.native_batcher.nthread}t{suffix}"
        return "python"

    def sync_planning_model(self) -> None:
        """Synchronize an explicitly changed planner model into fused workers."""
        if self.native_batcher is not None:
            self.native_batcher.sync_fused_model()

    def close(self) -> None:
        """Release native rollout worker threads explicitly."""
        if self.native_batcher is not None:
            self.native_batcher.close()
            self.native_batcher = None

    def _build_nominal(self, data, current_s: float):
        """Build a policy-seeded nominal, computing sensitivities only when needed.

        Standard MPPI with ``nominal_refine_iterations == 0`` is deliberately
        cheap: it uses the default policy rollout directly and does *not* finite
        difference every joint.  Sensitivity construction is reserved for SPG
        or an explicitly requested nominal refinement.
        """
        t0 = time.perf_counter()
        start = self.robot.snapshot(data)
        used_warm_start = bool(
            self.cfg.warm_start
            and self._previous_plan is not None
            and self._previous_plan.shape == (self.cfg.horizon, self.robot.nu)
        )
        if used_warm_start:
            shifted = np.empty_like(self._previous_plan)
            shifted[:-1] = self._previous_plan[1:]
            shifted[-1] = self._previous_plan[-1]
            policy_rollout = rollout_control_nominal(
                self.robot,
                start,
                shifted,
                self.track,
                current_s,
                control_substeps=self.control_substeps,
                native_batcher=self.native_batcher,
            )
        else:
            policy_rollout = rollout_policy_nominal(
                self.robot,
                start,
                self.policy,
                self.track,
                self.prior,
                current_s,
                horizon=self.cfg.horizon,
                control_substeps=self.control_substeps,
            )
        t_policy = time.perf_counter()

        need_spg = self.variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI
        need_refine = int(self.cfg.nominal_refine_iterations) > 0

        if not need_spg and not need_refine:
            refined = policy_rollout
            jac = None
            endpoints = policy_rollout.positions.copy()
        elif need_spg and not need_refine and (
            self.native_batcher is not None
            and self.native_batcher.supports_vectorized_cost
            and policy_rollout.native_initial_states is not None
            and policy_rollout.native_states is not None
        ):
            # Fast exact-dynamics SPG path.  The nominal states already contain
            # the unperturbed lookahead endpoints, so finite differences only
            # simulate the perturbed controls.
            refined = policy_rollout
            endpoints = self.native_batcher.nominal_lookahead_endpoints(
                policy_rollout.native_states, self.cfg.spg_lookahead_steps
            )
            interval = int(self.cfg.spg_jacobian_refresh_interval)
            can_shift = bool(
                used_warm_start
                and interval != 1
                and self._previous_jacobian is not None
                and self._previous_jacobian.shape == (self.cfg.horizon, 2, self.robot.nu)
            )
            # interval=1: original full refresh every tick.
            # interval=0: full refresh only on initialization, then incremental.
            # interval>1: incremental ticks with a periodic full correction.
            full_refresh = (not can_shift) or (
                interval > 1 and self._spg_updates % interval == 0
            )
            if full_refresh:
                jac, endpoints = self.native_batcher.estimate_joint_task_jacobians(
                    None, refined.controls,
                    control_substeps=self.control_substeps,
                    lookahead_steps=self.cfg.spg_lookahead_steps,
                    epsilon_fraction=self.cfg.sensitivity_epsilon_fraction,
                    initial_states=refined.native_initial_states,
                    nominal_states=refined.native_states,
                )
                self._last_spg_refresh_mode = "full"
            else:
                # Standard receding-horizon reuse: J[t+1] from the previous
                # solution is the natural seed for J[t] now.  This changes only
                # the proposal covariance; every candidate is still evaluated
                # with the configured MuJoCo planning model and integrator.
                jac = np.empty_like(self._previous_jacobian)
                jac[:-1] = self._previous_jacobian[1:]
                jac[-1] = self._previous_jacobian[-1]
                prefix = min(self.cfg.horizon, int(self.cfg.spg_jacobian_refresh_prefix))
                head = np.arange(prefix, dtype=np.int64)
                # A shifted plan has one genuinely new row at the horizon tail;
                # refresh that row as well so stale information never accumulates
                # indefinitely.  The head rows get priority because they shape
                # the controls that will actually be applied next.
                if self.cfg.horizon > prefix:
                    ids = np.concatenate((head, np.asarray([self.cfg.horizon - 1], dtype=np.int64)))
                else:
                    ids = head
                fresh, endpoints = self.native_batcher.estimate_joint_task_jacobians(
                    None, refined.controls,
                    control_substeps=self.control_substeps,
                    lookahead_steps=self.cfg.spg_lookahead_steps,
                    epsilon_fraction=self.cfg.sensitivity_epsilon_fraction,
                    initial_states=refined.native_initial_states,
                    nominal_states=refined.native_states,
                    time_indices=ids,
                )
                if ids.size:
                    jac[ids] = fresh[ids]
                self._last_spg_refresh_mode = (
                    f"shift+head{prefix}+tail" if self.cfg.horizon > prefix
                    else f"shift+head{prefix}"
                )
            self._previous_jacobian = np.asarray(jac, dtype=np.float64).copy()
        else:
            refined, jac, endpoints = refine_policy_nominal(
                self.robot,
                start,
                policy_rollout,
                self.track,
                self.prior,
                control_substeps=self.control_substeps,
                lookahead_steps=self.cfg.spg_lookahead_steps,
                iterations=int(self.cfg.nominal_refine_iterations),
                damping=self.cfg.nominal_refine_damping,
                step_size=self.cfg.nominal_refine_step_size,
                max_control_step_fraction=self.cfg.nominal_max_step_fraction,
                epsilon_fraction=self.cfg.sensitivity_epsilon_fraction,
                native_batcher=self.native_batcher,
            )
            if need_spg and jac is not None:
                self._previous_jacobian = np.asarray(jac, dtype=np.float64).copy()
                self._last_spg_refresh_mode = "full"
        t_sensitivity = time.perf_counter()

        endpoint_s, _ = self.track.project(endpoints)
        prior_mean, prior_cov = self.prior.sample(self.track, endpoint_s)
        t_prior = time.perf_counter()
        nominal_timing_ms = {
            "policy": 0.0 if used_warm_start else 1e3 * (t_policy - t0),
            "warm_start": 1e3 * (t_policy - t0) if used_warm_start else 0.0,
            "sensitivity": 1e3 * (t_sensitivity - t_policy),
            "prior": 1e3 * (t_prior - t_sensitivity),
        }
        return (
            start, policy_rollout, refined, jac, endpoints,
            np.asarray(prior_mean), np.asarray(prior_cov), nominal_timing_ms,
        )

    def _sample_standard(self, nominal: np.ndarray) -> np.ndarray:
        n, h, nu = self.cfg.num_rollouts, self.cfg.horizon, self.robot.nu
        std = self._joint_std
        if (
            self._standard_workspace is None
            or self._standard_workspace[0].shape != (n, h, nu)
        ):
            self._standard_workspace = (
                np.empty((n, h, nu), dtype=np.float64),
                np.empty((n, h, nu), dtype=np.float64),
            )
        noise, controls = self._standard_workspace
        self.rng.standard_normal(noise.shape, out=noise)
        np.multiply(noise, std[None, None, :], out=noise)
        rho = float(self.cfg.temporal_noise_smoothing)
        beta = math.sqrt(max(0.0, 1.0 - rho * rho))
        for t in range(1, h):
            noise[:, t] = rho * noise[:, t - 1] + beta * noise[:, t]
        np.add(nominal[None, :, :], noise, out=controls)
        np.clip(controls, self._ctrl_low, self._ctrl_high, out=controls)
        controls[0] = nominal
        return controls

    def _sample_spg(self, nominal, jac, endpoints, prior_mean, prior_cov):
        factors = build_spg_factors(
            jac,
            prior_mean,
            prior_cov,
            endpoints,
            damping=self.cfg.spg_pseudoinverse_damping,
            covariance_jitter=self.cfg.spg_covariance_jitter,
        )
        default_std = self._joint_std

        # The racing configuration uses pure SPG (mix=1).  Fuse the dense
        # null-space projection and AR(1) temporal smoothing in Numba while
        # leaving NumPy's RNG untouched.  This preserves the proposal
        # distribution and avoids several HxN temporary arrays/einsum passes.
        use_fast = bool(NUMBA_AVAILABLE and float(self.cfg.spg_mix) == 1.0)
        if use_fast:
            n, h, nu = self.cfg.num_rollouts, self.cfg.horizon, self.robot.nu
            shapes_ok = (
                self._spg_fast_workspace is not None
                and self._spg_fast_workspace[0].shape == (n, h, 2)
                and self._spg_fast_workspace[1].shape == (n, h, nu)
            )
            if not shapes_ok:
                self._spg_fast_workspace = (
                    np.empty((n, h, 2), dtype=np.float64),
                    np.empty((n, h, nu), dtype=np.float64),
                    np.empty((n, h, nu), dtype=np.float64),
                    np.empty((n, h, nu), dtype=np.float64),
                )
            z_task, z_null, noise, controls = self._spg_fast_workspace
            self.rng.standard_normal(z_task.shape, out=z_task)
            self.rng.standard_normal(z_null.shape, out=z_null)
            spg_dense_project_and_smooth(
                factors.task_factor, factors.null_projector, z_task, z_null, default_std,
                float(self.cfg.temporal_noise_smoothing),
                float(self.cfg.spg_null_std_scale), noise,
            )
            np.add(nominal[None, :, :], noise, out=controls)
            np.clip(controls, self._ctrl_low, self._ctrl_high, out=controls)
            controls[0] = nominal
            return controls, factors

        noise = sample_joint_noise(
            self.rng,
            factors,
            n=self.cfg.num_rollouts,
            default_std=default_std,
            temporal_smoothing=self.cfg.temporal_noise_smoothing,
            null_std_scale=self.cfg.spg_null_std_scale,
            spg_mix=self.cfg.spg_mix,
        )
        controls = nominal[None, :, :] + noise
        np.clip(controls, self._ctrl_low, self._ctrl_high, out=controls)
        controls[0] = nominal
        return controls, factors

    def step(self, data, current_s: float) -> tuple[np.ndarray, dict[str, Any]]:
        t_total = time.perf_counter()

        # A nominal-policy benchmark is closed-loop: only the action applied at
        # the current real state is needed.  Building an H-step simulated policy
        # rollout here used to perform H JAX inferences + H MuJoCo propagations
        # every 20 ms, making `policy_nominal` much slower computationally than
        # the nominal used by warm-started MPPI.  One inference per tick is both
        # faster and the faithful way to execute the pretrained running policy.
        if self.variant == ControllerVariant.POLICY_NOMINAL:
            t_policy0 = time.perf_counter()
            robot_s, _ = self.track.project(self.robot.xy(data))
            ctrl = np.asarray(
                self.policy.action(
                    self.robot, data, track=self.track, prior=self.prior,
                    current_s=float(robot_s),
                ),
                dtype=np.float64,
            )
            ctrl = self.robot.clip_ctrl(ctrl)
            t_policy1 = time.perf_counter()
            task_xy = np.asarray(self.robot.task_xy(data), dtype=np.float64).reshape(1, 2)
            one = ctrl.reshape(1, -1)
            elapsed = 1e3 * (t_policy1 - t_policy0)
            return ctrl.copy(), {
                "nominal": one.copy(),
                "policy_nominal": one.copy(),
                "nominal_positions": task_xy,
                "prior_mean": np.empty((0, 2), dtype=np.float64),
                "spatial_covariance": np.empty((0, 2, 2), dtype=np.float64),
                "joint_task_jacobians": None,
                "temperature": math.nan,
                "ess": 1.0,
                "finite_rollouts": 1,
                "rollout_backend": "policy-closed-loop",
                "spg_refresh_mode": "none",
                "timing_ms": {
                    "nominal": elapsed,
                    "policy": elapsed,
                    "warm_start": 0.0,
                    "sensitivity": 0.0,
                    "prior": 0.0,
                    "sampling": 0.0,
                    "rollouts": 0.0,
                    "rollout_physics": 0.0,
                    "rollout_cost": 0.0,
                    "rollout_fused": 0.0,
                    "update": 0.0,
                    "total": 1e3 * (time.perf_counter() - t_total),
                },
            }

        start, policy_nom, refined, jac, endpoints, prior_mean, prior_cov, nominal_parts = self._build_nominal(data, current_s)
        t_nominal = time.perf_counter()
        nominal = refined.controls

        factors = None
        if self.variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            controls, factors = self._sample_spg(nominal, jac, endpoints, prior_mean, prior_cov)
        else:
            controls = self._sample_standard(nominal)
        t_sample = time.perf_counter()

        positions, costs, terminal_progress, failed = evaluate_control_batch(
            self.robot,
            start,
            controls,
            self.track,
            current_s,
            control_substeps=self.control_substeps,
            nominal_controls=nominal,
            cost_cfg=self.cost_cfg,
            workers=self.cfg.rollout_workers,
            native_batcher=self.native_batcher,
        )
        t_rollout = time.perf_counter()
        rollout_physics_ms = (
            float(self.native_batcher.last_rollout_physics_ms)
            if self.native_batcher is not None else 0.0
        )
        rollout_cost_ms = (
            float(self.native_batcher.last_rollout_cost_ms)
            if self.native_batcher is not None else 0.0
        )
        rollout_fused_ms = (
            float(self.native_batcher.last_rollout_fused_ms)
            if self.native_batcher is not None else 0.0
        )

        if self.cfg.adaptive_temperature_lbps:
            if NUMBA_AVAILABLE:
                (
                    temperature, _alpha, ess, lbps_score, finite_count,
                    _reward_norm, _expected_return,
                ) = lbps_optimize_fast(
                    np.asarray(costs, dtype=np.float64),
                    float(self.cfg.lbps_delta),
                    float(self.cfg.lambda_temperature),
                    int(self.cfg.lbps_optimizer_iterations),
                )
            else:
                lbps = optimize_lbps_temperature(
                    costs,
                    delta=self.cfg.lbps_delta,
                    fallback_temperature=self.cfg.lambda_temperature,
                    iterations=self.cfg.lbps_optimizer_iterations,
                )
                temperature = lbps.temperature
                ess = lbps.ess
                finite_count = lbps.finite_count
                lbps_score = lbps.score
        else:
            temperature = self.cfg.lambda_temperature
            finite_count = int(np.count_nonzero(np.isfinite(costs)))
            finite = np.isfinite(costs)
            if finite_count:
                rho = np.min(costs[finite])
                w = np.exp(-(costs[finite] - rho) / temperature)
                ess = float(np.sum(w) ** 2 / max(np.sum(w * w), 1e-300))
            else:
                ess = 0.0
            lbps_score = math.nan

        candidate = weighted_control_sequence(costs, controls, temperature)
        np.clip(candidate, self._ctrl_low, self._ctrl_high, out=candidate)
        if self.cfg.warm_start:
            self._previous_plan = np.asarray(candidate, dtype=np.float64).copy()
        if self.variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            self._spg_updates += 1
        best = int(np.argmin(costs)) if len(costs) else 0
        t_update = time.perf_counter()
        info = {
            "planned_control_sequence": candidate,
            "policy_nominal": policy_nom.controls,
            "nominal": nominal,
            "nominal_positions": refined.positions,
            "prior_mean": prior_mean,
            "spatial_covariance": prior_cov,
            "joint_task_jacobians": jac,
            "spg_factors": factors,
            "temperature": float(temperature),
            "ess": float(ess),
            "lbps_score": float(lbps_score),
            "finite_rollouts": int(finite_count),
            "collision_rollouts": int(np.count_nonzero(failed)),
            "best_rollout": positions[best].copy(),
            "best_cost": float(costs[best]),
            "best_terminal_progress": float(terminal_progress[best]),
            "rollout_backend": self.rollout_backend_name,
            "spg_refresh_mode": self._last_spg_refresh_mode,
            "timing_ms": {
                "nominal": 1e3 * (t_nominal - t_total),
                **nominal_parts,
                "sampling": 1e3 * (t_sample - t_nominal),
                "rollouts": 1e3 * (t_rollout - t_sample),
                "rollout_physics": rollout_physics_ms,
                "rollout_cost": rollout_cost_ms,
                "rollout_fused": rollout_fused_ms,
                "update": 1e3 * (t_update - t_rollout),
                "total": 1e3 * (t_update - t_total),
            },
        }
        return candidate[0].copy(), info
