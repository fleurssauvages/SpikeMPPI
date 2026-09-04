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
from .fast_kernels import NUMBA_AVAILABLE, lbps_optimize_fast


class ControllerVariant(str, Enum):
    NOMINAL = "nominal"
    MPPI = "mppi"


class SamplingOption(str, Enum):
    """Candidate-generation strategy used by the MPPI controller."""

    STANDARD = "standard"
    GUIDED = "guided"
    DIAG_LOWRANK = "diag-lowrank"
    SPLINE = "spline"
    ICEM = "icem"


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

    # Alternative sampling proposals.  These only change how candidate
    # controls are generated; rollout physics, objective, LBPS and the MPPI
    # weighted update remain identical to standard MPPI.
    guided_rank: int = 6
    guided_fraction: float = 0.50
    diag_lowrank_rate: float = 0.08
    diag_lowrank_min: float = 0.25
    diag_lowrank_max: float = 4.0
    spline_modes: int = 6
    icem_elites: int = 4

    # Policy-seeded local iLQR-style nominal construction.
    nominal_refine_iterations: int = 0
    nominal_refine_damping: float = 1e-4
    nominal_refine_step_size: float = 0.35
    nominal_max_step_fraction: float = 0.15
    sensitivity_epsilon_fraction: float = 1e-3


    # Racing rollout constraints/cost.  A real fall requires BOTH an inverted
    # torso (root_up < min_root_up) and torso-ground contact. fall_height_fraction
    # is retained only for the state-only rollout fallback, where contact pairs
    # are unavailable and low root height is used as a conservative proxy.
    hard_collision_clearance: float = 0.02
    # Exact race/fused fall rule: root_up < min_root_up AND torso touches ground.
    # fall_height_fraction is used only by the state-only fallback as a contact proxy.
    fall_height_fraction: float = 0.45
    min_root_up: float = 0.0
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
    rollout_chunk_size: int = 0

    # Standard receding-horizon warm start. After the first update, shift the
    # previous optimized sequence instead of doing H synchronous policy calls.
    warm_start: bool = True

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.num_rollouts = max(1, int(self.num_rollouts))
        self.lbps_optimizer_iterations = max(8, int(self.lbps_optimizer_iterations))
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if self.lambda_temperature <= 0.0:
            raise ValueError("lambda_temperature must be positive")
        if not 0.0 < self.lbps_delta < 1.0:
            raise ValueError("lbps_delta must lie in (0, 1)")
        if not 0.0 <= self.temporal_noise_smoothing < 1.0:
            raise ValueError("temporal_noise_smoothing must lie in [0, 1)")
        if self.sensitivity_epsilon_fraction <= 0.0:
            raise ValueError("sensitivity_epsilon_fraction must be positive")
        self.guided_rank = max(1, int(self.guided_rank))
        if not 0.0 <= self.guided_fraction < 1.0:
            raise ValueError("guided_fraction must lie in [0, 1)")
        if not 0.0 < self.diag_lowrank_rate <= 1.0:
            raise ValueError("diag_lowrank_rate must lie in (0, 1]")
        if not 0.0 < self.diag_lowrank_min <= 1.0:
            raise ValueError("diag_lowrank_min must lie in (0, 1]")
        if self.diag_lowrank_max < 1.0:
            raise ValueError("diag_lowrank_max must be >= 1")
        if self.diag_lowrank_max < self.diag_lowrank_min:
            raise ValueError("diag_lowrank_max must be >= diag_lowrank_min")
        self.spline_modes = max(4, int(self.spline_modes))
        self.icem_elites = max(0, int(self.icem_elites))
        if self.box_progress_weight < 0.0 or self.robot_progress_weight < 0.0 or self.robot_box_approach_weight < 0.0:
            raise ValueError("push-task reward weights must be nonnegative")
        if self.box_max_lift < 0.0:
            raise ValueError("box_max_lift must be nonnegative")
        if not -1.0 <= self.box_min_up <= 1.0:
            raise ValueError("box_min_up must lie in [-1, 1]")
        self.rollout_chunk_size = max(0, int(self.rollout_chunk_size))


class JointMPPIController:
    """Native-MuJoCo policy-seeded direct-joint MPPI controller.

    This follows the uploaded controller's structure, but the control vector is
    no longer fixed to two vehicle controls. For every classic MuJoCo robot:

        u_t == MjData.ctrl in R^(model.nu)

    The nominal is seeded by that robot's locomotion policy, locally refined around
    the policy rollout, and MPPI explores directly in the full MuJoCo actuator space.
    """

    def __init__(
        self,
        robot,
        track,
        prior,
        policy,
        cfg: ControllerConfig,
        *,
        variant: ControllerVariant | str = ControllerVariant.MPPI,
        sampling: SamplingOption | str = SamplingOption.STANDARD,
        seed: int = 1,
    ) -> None:
        self.robot = robot
        self.track = track
        self.prior = prior
        self.policy = policy
        self.cfg = cfg
        self.variant = ControllerVariant(variant)
        self.sampling = SamplingOption(sampling)
        if self.variant == ControllerVariant.NOMINAL and self.sampling != SamplingOption.STANDARD:
            raise ValueError("non-standard --sampling options require --variant mppi")
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
        self.rollout_batcher = None
        # The same persistent native batcher is reused for candidate evaluation
        # and warm-start nominal propagation.
        self.nominal_batcher = None
        # Backwards-compatible alias used by candidate rollout helper arguments.
        self.native_batcher = None
        self._previous_plan: np.ndarray | None = None
        self._ctrl_low, self._ctrl_high = robot.control_bounds(cfg.unlimited_control_span)
        self._joint_std = robot.control_scale(
            fraction=cfg.joint_noise_fraction,
            unlimited_span=cfg.unlimited_control_span,
        )
        self._standard_workspace: tuple[np.ndarray, np.ndarray] | None = None
        self._direction_history: list[np.ndarray] = []
        self._last_proposal_rank = 0
        self._diag_variance = np.ones((cfg.horizon, robot.nu), dtype=np.float64)
        self._icem_elites: np.ndarray | None = None
        self._spline_basis = self._make_bspline_basis(cfg.horizon, cfg.spline_modes)
        # Online MPPI is CPU-only. Prefer the allocation-light fused C++
        # evaluator, then stock mujoco.rollout, then the Python fallback.
        try:
            self.rollout_batcher = NativeRolloutBatcher(
                robot, workers=cfg.rollout_workers, batch_hint=cfg.num_rollouts,
                chunk_size=cfg.rollout_chunk_size, fused=True,
            )
        except Exception:
            try:
                self.rollout_batcher = NativeRolloutBatcher(
                    robot, workers=cfg.rollout_workers, batch_hint=cfg.num_rollouts,
                    chunk_size=cfg.rollout_chunk_size, fused=False,
                )
            except Exception:
                self.rollout_batcher = None
        self.nominal_batcher = self.rollout_batcher
        self.native_batcher = self.rollout_batcher

    @property
    def rollout_backend_name(self) -> str:
        batcher = self.rollout_batcher
        if batcher is None:
            return "cpu/python"
        if batcher.supports_vectorized_cost:
            suffix = f"/chunk{batcher.chunk_size}" if batcher.chunk_size > 0 else ""
            prefix = "cpu/fused" if batcher.uses_fused else "cpu/native"
            return f"{prefix}/{batcher.nthread}t{suffix}"
        return "cpu/python"

    @property
    def sampling_description(self) -> str:
        if self.sampling == SamplingOption.GUIDED:
            return (
                f"guided low-rank: history_rank={self.cfg.guided_rank}, "
                f"guided_fraction={self.cfg.guided_fraction:g}, baseline=standard MPPI"
            )
        if self.sampling == SamplingOption.DIAG_LOWRANK:
            return (
                f"diagonal + low-rank: history_rank={self.cfg.guided_rank}, "
                f"guided_fraction={self.cfg.guided_fraction:g}, "
                f"diag_rate={self.cfg.diag_lowrank_rate:g}"
            )
        if self.sampling == SamplingOption.SPLINE:
            return f"spline latent sampling: cubic B-spline modes={self._spline_basis.shape[1]} per joint"
        if self.sampling == SamplingOption.ICEM:
            return f"iCEM-style elite reuse: shifted_elites={self.cfg.icem_elites}"
        return "standard MPPI Gaussian sampling"

    @property
    def proposal_description(self) -> str:
        """Backward-compatible alias for the former proposal terminology."""
        return self.sampling_description

    def sync_planning_model(self) -> None:
        """Synchronize explicit planner-model changes into active backends."""
        seen = set()
        for batcher in (self.rollout_batcher, self.nominal_batcher):
            if batcher is None or id(batcher) in seen:
                continue
            seen.add(id(batcher))
            if hasattr(batcher, "sync_model"):
                batcher.sync_model()
            elif hasattr(batcher, "sync_fused_model"):
                batcher.sync_fused_model()

    def close(self) -> None:
        """Release rollout resources explicitly."""
        seen = set()
        for batcher in (self.rollout_batcher, self.nominal_batcher):
            if batcher is None or id(batcher) in seen:
                continue
            seen.add(id(batcher))
            batcher.close()
        self.rollout_batcher = None
        self.nominal_batcher = None
        self.native_batcher = None

    def _build_nominal(self, data, current_s: float):
        """Build the policy-seeded control nominal used by MPPI.

        With warm start enabled, the previous optimized plan is shifted and
        re-simulated. Otherwise the pretrained policy generates the H-step
        nominal. Optional nominal refinement remains available. MPPI candidate
        generation is selected independently through ``SamplingOption``.
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
                native_batcher=self.nominal_batcher,
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

        refined = policy_rollout
        if int(self.cfg.nominal_refine_iterations) > 0:
            refined, _jac, _endpoints = refine_policy_nominal(
                self.robot,
                start,
                policy_rollout,
                self.track,
                self.prior,
                control_substeps=self.control_substeps,
                lookahead_steps=3,
                iterations=int(self.cfg.nominal_refine_iterations),
                damping=self.cfg.nominal_refine_damping,
                step_size=self.cfg.nominal_refine_step_size,
                max_control_step_fraction=self.cfg.nominal_max_step_fraction,
                epsilon_fraction=self.cfg.sensitivity_epsilon_fraction,
                native_batcher=self.nominal_batcher,
            )
        t_refine = time.perf_counter()

        endpoints = np.asarray(refined.positions, dtype=np.float64)
        endpoint_s, _ = self.track.project(endpoints)
        prior_mean, prior_cov = self.prior.sample(self.track, endpoint_s)
        t_prior = time.perf_counter()

        nominal_timing_ms = {
            "policy": 0.0 if used_warm_start else 1e3 * (t_policy - t0),
            "warm_start": 1e3 * (t_policy - t0) if used_warm_start else 0.0,
            "sensitivity": 1e3 * (t_refine - t_policy),
            "prior": 1e3 * (t_prior - t_refine),
        }
        return (
            start,
            policy_rollout,
            refined,
            None,
            endpoints,
            np.asarray(prior_mean),
            np.asarray(prior_cov),
            nominal_timing_ms,
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

    @staticmethod
    def _make_bspline_basis(horizon: int, modes: int, degree: int = 3) -> np.ndarray:
        """Return a row-normalized clamped uniform B-spline basis.

        Row normalization keeps unit latent coefficient variance at roughly
        unit marginal variance in every control timestep, so ``joint_noise``
        retains the same scale interpretation as standard MPPI.
        """
        h = max(1, int(horizon))
        m = max(degree + 1, min(int(modes), h))
        n_internal = m - degree - 1
        if n_internal > 0:
            interior = np.linspace(0.0, 1.0, n_internal + 2, dtype=np.float64)[1:-1]
        else:
            interior = np.empty(0, dtype=np.float64)
        knots = np.concatenate((
            np.zeros(degree + 1, dtype=np.float64),
            interior,
            np.ones(degree + 1, dtype=np.float64),
        ))
        x = np.linspace(0.0, 1.0, h, dtype=np.float64)
        # Cox-de Boor recursion. There are m basis functions.
        basis = np.zeros((h, m), dtype=np.float64)
        for i in range(m):
            left, right = knots[i], knots[i + 1]
            basis[:, i] = ((x >= left) & (x < right)).astype(np.float64)
        basis[-1, -1] = 1.0
        for p in range(1, degree + 1):
            nxt = np.zeros_like(basis)
            for i in range(m):
                left_den = knots[i + p] - knots[i]
                if left_den > 0.0:
                    nxt[:, i] += ((x - knots[i]) / left_den) * basis[:, i]
                if i + 1 < m:
                    right_den = knots[i + p + 1] - knots[i + 1]
                    if right_den > 0.0:
                        nxt[:, i] += ((knots[i + p + 1] - x) / right_den) * basis[:, i + 1]
            basis = nxt
        basis[-1, :] = 0.0
        basis[-1, -1] = 1.0
        row_norm = np.sqrt(np.sum(basis * basis, axis=1, keepdims=True))
        basis /= np.maximum(row_norm, 1e-12)
        return basis

    def _temporal_filter_inplace(self, noise: np.ndarray) -> None:
        rho = float(self.cfg.temporal_noise_smoothing)
        beta = math.sqrt(max(0.0, 1.0 - rho * rho))
        for t in range(1, noise.shape[1]):
            noise[:, t] = rho * noise[:, t - 1] + beta * noise[:, t]

    def _history_basis(self) -> np.ndarray | None:
        """Orthonormal low-rank basis in normalized full-sequence space."""
        if not self._direction_history:
            return None
        d = self.cfg.horizon * self.robot.nu
        cols = []
        for direction in self._direction_history[-self.cfg.guided_rank:]:
            v = np.asarray(direction, dtype=np.float64).reshape(d)
            nrm = float(np.linalg.norm(v))
            if np.isfinite(nrm) and nrm > 1e-8:
                cols.append(v / nrm)
        if not cols:
            return None
        matrix = np.column_stack(cols)
        try:
            q, _ = np.linalg.qr(matrix, mode="reduced")
        except np.linalg.LinAlgError:
            return None
        if q.size == 0:
            return None
        return np.asarray(q[:, : min(q.shape[1], self.cfg.guided_rank)], dtype=np.float64)

    def _sample_guided(self, nominal: np.ndarray, *, adaptive_diagonal: bool) -> np.ndarray:
        n, h, nu = self.cfg.num_rollouts, self.cfg.horizon, self.robot.nu
        z = self.rng.standard_normal((n, h, nu))
        if adaptive_diagonal:
            z *= np.sqrt(np.maximum(self._diag_variance, 1e-12))[None, :, :]
        self._temporal_filter_inplace(z)

        q = self._history_basis()
        self._last_proposal_rank = 0 if q is None else int(q.shape[1])
        alpha = float(self.cfg.guided_fraction) if q is not None else 0.0
        if alpha > 0.0 and q is not None:
            # Blend an ordinary full-space draw with a low-rank guided draw,
            # then renormalize the expected trace back to D.  This is much less
            # aggressive than assigning alpha*D/r variance to each guided
            # direction (which is pathological for D=H*nu=400 and small r).
            d, r = h * nu, q.shape[1]
            z *= math.sqrt(max(0.0, 1.0 - alpha))
            coeff = self.rng.standard_normal((n, r))
            z.reshape(n, d)[:] += math.sqrt(alpha) * (coeff @ q.T)
            expected_trace = (1.0 - alpha) * d + alpha * r
            z *= math.sqrt(d / max(expected_trace, 1e-12))

        noise = z * self._joint_std[None, None, :]
        controls = nominal[None, :, :] + noise
        np.clip(controls, self._ctrl_low, self._ctrl_high, out=controls)
        controls[0] = nominal
        return controls

    def _sample_spline(self, nominal: np.ndarray) -> np.ndarray:
        n, h, nu = self.cfg.num_rollouts, self.cfg.horizon, self.robot.nu
        m = self._spline_basis.shape[1]
        coeff = self.rng.standard_normal((n, m, nu))
        z = np.einsum("tm,nmu->ntu", self._spline_basis, coeff, optimize=True)
        controls = nominal[None, :, :] + z * self._joint_std[None, None, :]
        np.clip(controls, self._ctrl_low, self._ctrl_high, out=controls)
        controls[0] = nominal
        return controls

    def _sample_icem(self, nominal: np.ndarray) -> np.ndarray:
        controls = self._sample_standard(nominal)
        if self._icem_elites is None or self.cfg.icem_elites <= 0:
            return controls
        k = min(int(self.cfg.icem_elites), controls.shape[0] - 1, self._icem_elites.shape[0])
        if k <= 0:
            return controls
        shifted = np.empty_like(self._icem_elites[:k])
        shifted[:, :-1] = self._icem_elites[:k, 1:]
        shifted[:, -1] = self._icem_elites[:k, -1]
        np.clip(shifted, self._ctrl_low, self._ctrl_high, out=shifted)
        controls[1 : 1 + k] = shifted
        return controls

    @staticmethod
    def _normalized_weights(costs: np.ndarray, temperature: float) -> np.ndarray:
        c = np.asarray(costs, dtype=np.float64).reshape(-1)
        finite = np.isfinite(c)
        w = np.zeros_like(c)
        if not np.any(finite):
            if len(w):
                w[:] = 1.0 / len(w)
            return w
        rho = float(np.min(c[finite]))
        w[finite] = np.exp(np.clip(-(c[finite] - rho) / max(float(temperature), 1e-300), -745.0, 0.0))
        total = float(np.sum(w))
        if total <= 1e-12:
            w[finite] = 1.0 / np.count_nonzero(finite)
        else:
            w /= total
        return w

    def _shift_horizon_array(self, x: np.ndarray, *, tail) -> np.ndarray:
        y = np.empty_like(x)
        y[:-1] = x[1:]
        y[-1] = tail
        return y

    def _update_direction_memory(self, candidate: np.ndarray, nominal: np.ndarray) -> None:
        if self.sampling not in {SamplingOption.GUIDED, SamplingOption.DIAG_LOWRANK}:
            return
        denom = np.maximum(self._joint_std, 1e-12)
        direction = (np.asarray(candidate) - np.asarray(nominal)) / denom[None, :]
        if not np.all(np.isfinite(direction)):
            return
        if self.cfg.warm_start:
            shifted_history = []
            for old in self._direction_history:
                old_h = np.asarray(old, dtype=np.float64).reshape(self.cfg.horizon, self.robot.nu)
                old_shift = self._shift_horizon_array(old_h, tail=np.zeros(self.robot.nu))
                shifted_history.append(old_shift.reshape(-1))
            direction = self._shift_horizon_array(direction, tail=np.zeros(self.robot.nu))
            self._direction_history = shifted_history
        else:
            self._direction_history = []
        self._direction_history.append(direction.reshape(-1).copy())
        if len(self._direction_history) > self.cfg.guided_rank:
            self._direction_history = self._direction_history[-self.cfg.guided_rank :]

    def _update_adaptive_diagonal(
        self,
        controls: np.ndarray,
        nominal: np.ndarray,
        costs: np.ndarray,
        temperature: float,
        ess: float,
    ) -> None:
        if self.sampling != SamplingOption.DIAG_LOWRANK:
            return
        w = self._normalized_weights(costs, temperature)
        denom = np.maximum(self._joint_std, 1e-12)
        delta = (np.asarray(controls) - np.asarray(nominal)[None, :, :]) / denom[None, None, :]
        mean = np.einsum("n,nhu->hu", w, delta)
        centered = delta - mean[None, :, :]
        empirical = np.einsum("n,nhu->hu", w, centered * centered)

        # With small populations, only trust the empirical variance when the
        # MPPI weights retain a reasonable ESS. Otherwise shrink to standard
        # MPPI variance instead of collapsing around a few samples.
        target_ess = max(4.0, min(16.0, 0.5 * self.cfg.num_rollouts))
        confidence = float(np.clip((float(ess) - 1.0) / max(target_ess - 1.0, 1.0), 0.0, 1.0))
        target = (1.0 - confidence) + confidence * empirical
        target = np.clip(target, self.cfg.diag_lowrank_min, self.cfg.diag_lowrank_max)
        mean_target = float(np.mean(target))
        if mean_target > 1e-12:
            target /= mean_target
        rate = float(self.cfg.diag_lowrank_rate)
        updated = (1.0 - rate) * self._diag_variance + rate * target
        updated = np.clip(updated, self.cfg.diag_lowrank_min, self.cfg.diag_lowrank_max)
        mean_updated = float(np.mean(updated))
        if mean_updated > 1e-12:
            updated /= mean_updated
        if self.cfg.warm_start:
            updated = self._shift_horizon_array(updated, tail=np.ones(self.robot.nu))
        self._diag_variance = updated

    def _update_icem_elites(self, controls: np.ndarray, costs: np.ndarray) -> None:
        if self.sampling != SamplingOption.ICEM or self.cfg.icem_elites <= 0:
            return
        finite = np.flatnonzero(np.isfinite(costs))
        if finite.size == 0:
            return
        order = finite[np.argsort(np.asarray(costs)[finite])]
        k = min(int(self.cfg.icem_elites), len(order))
        self._icem_elites = np.asarray(controls[order[:k]], dtype=np.float64).copy()

    def step(self, data, current_s: float) -> tuple[np.ndarray, dict[str, Any]]:
        t_total = time.perf_counter()

        # A nominal-policy benchmark is closed-loop: only the action applied at
        # the current real state is needed.  Building an H-step simulated policy
        # rollout here used to perform H JAX inferences + H MuJoCo propagations
        # every 20 ms, making `nominal` much slower computationally than
        # the nominal used by warm-started MPPI.  One inference per tick is both
        # faster and the faithful way to execute the pretrained running policy.
        if self.variant == ControllerVariant.NOMINAL:
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
                "temperature": math.nan,
                "ess": 1.0,
                "finite_rollouts": 1,
                "rollout_backend": "policy-closed-loop",
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

        start, policy_nom, refined, sensitivity, endpoints, prior_mean, prior_cov, nominal_parts = self._build_nominal(data, current_s)
        t_nominal = time.perf_counter()
        nominal = refined.controls

        if self.sampling == SamplingOption.GUIDED:
            controls = self._sample_guided(nominal, adaptive_diagonal=False)
        elif self.sampling == SamplingOption.DIAG_LOWRANK:
            controls = self._sample_guided(nominal, adaptive_diagonal=True)
        elif self.sampling == SamplingOption.SPLINE:
            controls = self._sample_spline(nominal)
        elif self.sampling == SamplingOption.ICEM:
            controls = self._sample_icem(nominal)
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
        self._update_adaptive_diagonal(controls, nominal, costs, temperature, ess)
        self._update_direction_memory(candidate, nominal)
        self._update_icem_elites(controls, costs)
        if self.cfg.warm_start:
            self._previous_plan = np.asarray(candidate, dtype=np.float64).copy()
        best = int(np.argmin(costs)) if len(costs) else 0
        if getattr(self.native_batcher, "returns_best_only", False):
            best_rollout = np.asarray(positions, dtype=np.float64).copy()
        else:
            best_rollout = positions[best].copy()
        t_update = time.perf_counter()
        info = {
            "planned_control_sequence": candidate,
            "policy_nominal": policy_nom.controls,
            "nominal": nominal,
            "nominal_positions": refined.positions,
            "prior_mean": prior_mean,
            "spatial_covariance": prior_cov,
            "temperature": float(temperature),
            "ess": float(ess),
            "lbps_score": float(lbps_score),
            "finite_rollouts": int(finite_count),
            "collision_rollouts": int(np.count_nonzero(failed)),
            "best_rollout": best_rollout,
            "best_cost": float(costs[best]),
            "best_terminal_progress": float(terminal_progress[best]),
            "rollout_backend": self.rollout_backend_name,
            "sampling_option": self.sampling.value,
            "proposal": self.sampling.value,  # backward-compatible info key
            "proposal_rank": int(self._last_proposal_rank),
            "diag_variance_min": float(np.min(self._diag_variance)),
            "diag_variance_max": float(np.max(self._diag_variance)),
            "icem_elites": 0 if self._icem_elites is None else int(len(self._icem_elites)),
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
