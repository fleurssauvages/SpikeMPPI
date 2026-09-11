from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import math
import time
import numpy as np

from .lbps import optimize_lbps_temperature
from .rollout import (
    RolloutCostConfig,
    NominalRollout,
    NativeRolloutBatcher,
    evaluate_control_batch,
    rollout_policy_nominal,
    rollout_control_nominal,
    refine_policy_nominal,
)
from .fast_kernels import NUMBA_AVAILABLE, lbps_optimize_fast
from .spike_kernels import StaticSpikeSampler, resolve_spike_sampler


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
    SPIKE = "spike"


BIO_EVENT_SAMPLING_OPTIONS = frozenset({SamplingOption.SPIKE})

@dataclass
class ControllerConfig:
    control_dt: float = 0.02
    horizon: int = 75
    num_rollouts: int = 32
    lambda_temperature: float = 1.0
    adaptive_temperature_lbps: bool = True
    lbps_delta: float = 0.95
    lbps_optimizer_iterations: int = 32
    temporal_noise_smoothing: float = 0.25

    # Direct joint-space proposal.
    joint_noise_fraction: float = 0.25
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

    # Shared marked-Poisson/twitch physiology. ``spike`` is always the fixed
    # direct-joint baseline: one independent neuron per actuator, identity wiring,
    # and no online learning.
    spike_rate_hz: float = 16.0
    spike_recruitment_levels: int = 6
    spike_sampler: str = "auto"


    spike_twitch_rise_s: float = 0.016
    spike_twitch_decay_s: float = 0.064
    spike_twitch_duration_s: float = 0.200

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
    # Nominal geometry is diagnostic-only unless nominal refinement is enabled.
    # Keep API compatibility; race.py disables it when the overlay is not used.
    nominal_diagnostics: bool = True

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
        if self.spike_sampler not in {"auto", "numpy", "numba"}:
            raise ValueError("spike_sampler must be auto, numpy, or numba")
        if not np.isfinite(self.spike_rate_hz) or self.spike_rate_hz <= 0:
            raise ValueError("spike_rate_hz must be finite and positive")
        self.spike_recruitment_levels = max(1, int(self.spike_recruitment_levels))
        if self.spike_twitch_rise_s <= 0.0:
            raise ValueError("spike_twitch_rise_s must be positive")
        if self.spike_twitch_decay_s <= self.spike_twitch_rise_s:
            raise ValueError("spike_twitch_decay_s must be greater than spike_twitch_rise_s")
        if self.spike_twitch_duration_s <= 0.0:
            raise ValueError("spike_twitch_duration_s must be positive")
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
        self._weight_workspace = np.empty(cfg.num_rollouts, dtype=np.float64)
        self._ctrl_low, self._ctrl_high = robot.control_bounds(cfg.unlimited_control_span)
        # Actuator-scale normalization used only by benchmark/control-quality
        # diagnostics. Keep this independent of --joint-noise so normalized
        # residual/smoothness metrics remain comparable across samplers and
        # exploration-scale sweeps. This matches NativeRolloutBatcher.
        self._ctrl_scale = np.maximum(
            np.asarray(robot.control_scale(), dtype=np.float64), 1e-6
        )
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
        self._spike_sampler_backend = (
            resolve_spike_sampler(cfg.spike_sampler)
            if self.sampling == SamplingOption.SPIKE
            else "unused"
        )
        self._spike_levels, self._spike_level_prior = self._make_recruitment_marks(
            cfg.spike_recruitment_levels
        )
        self._spike_twitch = self._make_spike_twitch_kernel(
            cfg.control_dt,
            cfg.spike_twitch_rise_s,
            cfg.spike_twitch_decay_s,
            cfg.spike_twitch_duration_s,
        )
        h = int(cfg.horizon)
        # ``spike`` is intentionally the former spike-joint baseline: exactly one
        # independent channel per physical actuator and identity decoding.
        self._spike_synergies = np.eye(robot.nu, dtype=np.float64)
        m = int(robot.nu)
        spike_shape = (h, m)
        self._spike_total_rate_hz = float(robot.nu) * float(cfg.spike_rate_hz)
        per_channel_rate = self._spike_total_rate_hz / float(m)
        half_base = 0.5 * per_channel_rate
        self._spike_pos_rate_map = np.full(spike_shape, half_base, dtype=np.float64)
        self._spike_neg_rate_map = np.full(spike_shape, half_base, dtype=np.float64)
        self._spike_mark_prob_map = np.broadcast_to(
            self._spike_level_prior, spike_shape + (len(self._spike_levels),)
        ).copy()
        for array in (self._spike_pos_rate_map, self._spike_neg_rate_map,
                      self._spike_mark_prob_map):
            array.flags.writeable = False
        self._last_spike_event_count_mean = 0.0
        k2_prefix = np.cumsum(self._spike_twitch ** 2)
        remaining = np.minimum(np.arange(h, 0, -1), len(k2_prefix))
        self._spike_variance_weights = k2_prefix[remaining - 1] / float(h)
        self._spike_global_noise_scale = self._compute_spike_global_noise_scale()
        self._spike_generator = None
        if self.sampling == SamplingOption.SPIKE:
            self._spike_generator = StaticSpikeSampler(
                self._spike_pos_rate_map, self._spike_neg_rate_map,
                self._spike_mark_prob_map, self._spike_levels,
                cfg.control_dt, cfg.num_rollouts, self._spike_twitch,
                backend=self._spike_sampler_backend,
            )

        self._fixed_spike_diagnostics = self._make_fixed_spike_diagnostics()

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
        if self.sampling == SamplingOption.SPIKE:
            return (
                "Spike-MPPI: fixed direct-joint signed marked-Poisson events + causal twitch; "
                f"neurons={self.robot.nu}, base_rate={self.cfg.spike_rate_hz:g}Hz, "
                "identity wiring, no learning, unchanged nominal, "
                f"implementation={self._spike_sampler_backend}"
            )
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
            if not self.cfg.nominal_diagnostics and self.cfg.nominal_refine_iterations <= 0:
                # Only the shifted controls enter sampling and rollout costs.
                # Re-simulating H nominal steps here cannot change them. Omit
                # that serial physics pass when no caller needs its geometry.
                # Empty diagnostics are explicit, not stale/approximate paths.
                shifted = self.robot.clip_ctrl(shifted)
                policy_rollout = NominalRollout(
                    shifted, [], np.empty((0, 2)), np.empty(0), np.empty(0)
                )
            else:
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
        if len(endpoints):
            endpoint_s, _ = self.track.project(endpoints)
            prior_mean, prior_cov = self.prior.sample(self.track, endpoint_s)
        else:
            prior_mean = np.empty((0, 2), dtype=np.float64)
            prior_cov = np.empty((0, 2, 2), dtype=np.float64)
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
        return controls

    def _sample_spline(self, nominal: np.ndarray) -> np.ndarray:
        n, h, nu = self.cfg.num_rollouts, self.cfg.horizon, self.robot.nu
        m = self._spline_basis.shape[1]
        coeff = self.rng.standard_normal((n, m, nu))
        z = np.einsum("tm,nmu->ntu", self._spline_basis, coeff, optimize=True)
        controls = nominal[None, :, :] + z * self._joint_std[None, None, :]
        np.clip(controls, self._ctrl_low, self._ctrl_high, out=controls)
        return controls

    @staticmethod
    def _make_spike_twitch_kernel(
        control_dt: float,
        rise_s: float,
        decay_s: float,
        duration_s: float,
    ) -> np.ndarray:
        """Return a causal, positive difference-of-exponentials twitch kernel."""
        steps = max(1, int(math.ceil(float(duration_s) / float(control_dt))))
        # Evaluate at the center/end of each discrete control bin rather than
        # exactly t=0; otherwise a difference of exponentials would have zero
        # effect on the control sample in which the event occurs.
        t = (np.arange(steps, dtype=np.float64) + 1.0) * float(control_dt)
        kernel = np.exp(-t / float(decay_s)) - np.exp(-t / float(rise_s))
        kernel = np.maximum(kernel, 0.0)
        peak = float(np.max(kernel)) if kernel.size else 0.0
        if peak <= 1e-12:
            return np.ones(1, dtype=np.float64)
        kernel /= peak
        return kernel

    @staticmethod
    def _make_recruitment_marks(
        levels: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ordered normalized recruitment marks and a uniform prior."""
        levels = max(1, int(levels))

        amplitudes = (
            np.arange(1, levels + 1, dtype=np.float64)
            / float(levels)
        )

        prior = np.full(
            levels,
            1.0 / float(levels),
            dtype=np.float64,
        )

        return amplitudes, prior

    def _compute_spike_global_noise_scale(self) -> float:
        """Match only the *global* uniform-law motor power to the 8-neuron identity baseline.

        There is intentionally no per-joint variance matching.  Every sparse
        primitive has unit row energy, so the fixed population event budget
        determines total motor power.  This single scalar converts the common
        twitch/event law to the configured joint-noise scale while preserving
        natural anisotropy from primitive wiring and learned spike statistics.
        """
        total_rate = np.asarray(
            self._spike_pos_rate_map + self._spike_neg_rate_map, dtype=np.float64
        )
        mark_prob = np.asarray(self._spike_mark_prob_map, dtype=np.float64)
        mark_sq = self._spike_levels * self._spike_levels
        expected_mark_sq = np.sum(mark_prob * mark_sq[None, None, :], axis=-1)
        impulse_variance = float(self.cfg.control_dt) * total_rate * expected_mark_sq
        channel_variance = self._spike_variance_weights @ impulse_variance
        total_motor_variance = float(
            np.sum(channel_variance * np.sum(self._spike_synergies ** 2, axis=1))
        )
        mean_motor_variance = total_motor_variance / float(max(self.robot.nu, 1))
        return 1.0 / math.sqrt(max(mean_motor_variance, 1e-12))

    def _make_fixed_spike_diagnostics(self) -> dict[str, float]:
        """Diagnostics for the fixed direct-joint Spike baseline."""
        m = int(self.robot.nu)
        levels = len(self._spike_levels)
        total_rate = float(self._spike_total_rate_hz)
        per_rate = total_rate / max(m, 1)
        out = {
            "spike_rate_hz_mean": per_rate, "spike_rate_hz_std": 0.0,
            "spike_rate_hz_min": per_rate, "spike_rate_hz_max": per_rate,
            "spike_pos_rate_hz_min": per_rate / 2, "spike_pos_rate_hz_max": per_rate / 2,
            "spike_neg_rate_hz_min": per_rate / 2, "spike_neg_rate_hz_max": per_rate / 2,
            "spike_sign_prob_min": 0.5, "spike_sign_prob_max": 0.5,
            "spike_sign_entropy": 1.0,
            "spike_recruitment_entropy": float(levels > 1),
            "spike_mark_entropy_mean": float(levels > 1),
            "spike_mark_prob_min": 1.0 / levels, "spike_mark_prob_max": 1.0 / levels,
            "spike_mean_recruitment": float(np.mean(self._spike_levels)),
            "spike_effective_synergies": float(m) if self.sampling == SamplingOption.SPIKE else math.nan,
            "spike_synergy_entropy": float(m > 1) if self.sampling == SamplingOption.SPIKE else math.nan,
            "spike_expected_events": self.cfg.control_dt * self.cfg.horizon * total_rate,
            "spike_total_rate_hz": total_rate,
            "spike_event_budget_fixed": 1.0,
            "spike_firing_fixed": float(self.sampling == SamplingOption.SPIKE),
        }
        if self.sampling not in BIO_EVENT_SAMPLING_OPTIONS:
            return {key: math.nan for key in out}
        return out

    def _spike_diagnostics(self) -> dict[str, float]:
        return self._fixed_spike_diagnostics

    def _sample_spike(self, nominal: np.ndarray) -> np.ndarray:
        """Fixed spike-joint baseline: no contextual or online adaptation."""
        if self._spike_generator is None:
            raise RuntimeError("Spike generator is unavailable")
        n = self.cfg.num_rollouts
        z, event_total = self._spike_generator.sample_projected_identity(self.rng)
        z *= self._spike_global_noise_scale
        z *= self._joint_std[None, None, :]
        # ``z`` is borrowed sampler workspace and is fully overwritten on the
        # next draw. Reuse it as the candidate-control batch rather than
        # allocating another N x H x nu array every 20 ms.
        z += nominal[None, :, :]
        np.clip(z, self._ctrl_low, self._ctrl_high, out=z)
        self._last_spike_event_count_mean = float(event_total) / n
        return z

    def _sample_icem(self, nominal: np.ndarray) -> np.ndarray:
        controls = self._sample_standard(nominal)
        if self._icem_elites is None or self.cfg.icem_elites <= 0:
            return controls
        k = min(int(self.cfg.icem_elites), controls.shape[0], self._icem_elites.shape[0])
        if k <= 0:
            return controls
        shifted = np.empty_like(self._icem_elites[:k])
        shifted[:, :-1] = self._icem_elites[:k, 1:]
        shifted[:, -1] = self._icem_elites[:k, -1]
        np.clip(shifted, self._ctrl_low, self._ctrl_high, out=shifted)
        controls[:k] = shifted
        return controls

    def _normalized_weights(self, costs: np.ndarray, temperature: float) -> np.ndarray:
        """Return normalized MPPI weights in a persistent N-element buffer.

        The same weights drive the control update, diagnostics, and (for the
        diagonal-low-rank sampler) covariance adaptation. Computing the
        exponential weights once avoids two or three duplicate passes per tick.
        The returned array is borrowed until the next controller step.
        """
        c = np.asarray(costs, dtype=np.float64).reshape(-1)
        if self._weight_workspace.shape != c.shape:
            self._weight_workspace = np.empty_like(c)
        w = self._weight_workspace
        finite = np.isfinite(c)
        w.fill(0.0)
        finite_count = int(np.count_nonzero(finite))
        if finite_count == 0:
            if len(w):
                w.fill(1.0 / len(w))
            return w
        rho = float(np.min(c[finite]))
        w[finite] = np.exp(
            np.clip(
                -(c[finite] - rho) / max(float(temperature), 1e-300),
                -745.0, 0.0,
            )
        )
        total = float(np.sum(w))
        if total <= 1e-12:
            w[finite] = 1.0 / finite_count
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
        weights: np.ndarray,
        ess: float,
    ) -> None:
        if self.sampling != SamplingOption.DIAG_LOWRANK:
            return
        w = weights
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
        elif self.sampling == SamplingOption.SPIKE:
            controls = self._sample_spike(nominal)
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

        # Compute MPPI weights once and reuse them for the weighted sequence,
        # diagnostics, and any sampler-specific adaptation.
        diag_weights = self._normalized_weights(costs, temperature)
        candidate = np.einsum("n,nhu->hu", diag_weights, controls, optimize=False)
        np.clip(candidate, self._ctrl_low, self._ctrl_high, out=candidate)
        if self.sampling == SamplingOption.DIAG_LOWRANK:
            self._update_adaptive_diagonal(controls, nominal, diag_weights, ess)
        if self.sampling in {SamplingOption.GUIDED, SamplingOption.DIAG_LOWRANK}:
            self._update_direction_memory(candidate, nominal)
        if self.sampling == SamplingOption.ICEM:
            self._update_icem_elites(controls, costs)

        if self.cfg.warm_start:
            self._previous_plan = np.asarray(candidate, dtype=np.float64).copy()
        best = int(np.argmin(costs)) if len(costs) else 0
        if getattr(self.native_batcher, "returns_best_only", False):
            best_rollout = np.asarray(positions, dtype=np.float64).copy()
        else:
            best_rollout = positions[best].copy()
        t_update = time.perf_counter()

        # Optimization-quality diagnostics use only the sampled rollout
        # population.  The nominal sequence is the proposal center but is not
        # evaluated as an extra rollout, so nominal-cost deltas are undefined.
        finite_cost = np.isfinite(costs)
        nominal_cost = math.nan
        costs_arr = np.asarray(costs, dtype=np.float64)
        if np.any(finite_cost):
            weighted_rollout_cost = float(
                np.sum(diag_weights[finite_cost] * costs_arr[finite_cost])
            )
            best_finite_cost = float(np.min(costs_arr[finite_cost]))
        else:
            weighted_rollout_cost = math.nan
            best_finite_cost = math.nan

        if np.isfinite(nominal_cost) and np.isfinite(weighted_rollout_cost):
            nominal_weighted_improvement = nominal_cost - weighted_rollout_cost
            nominal_weighted_improvement_rel = nominal_weighted_improvement / max(
                abs(nominal_cost), 1e-12
            )
        else:
            nominal_weighted_improvement = math.nan
            nominal_weighted_improvement_rel = math.nan

        if np.isfinite(nominal_cost) and np.isfinite(best_finite_cost):
            nominal_best_improvement = nominal_cost - best_finite_cost
            nominal_best_improvement_rel = nominal_best_improvement / max(
                abs(nominal_cost), 1e-12
            )
        else:
            nominal_best_improvement = math.nan
            nominal_best_improvement_rel = math.nan

        applied_delta = np.asarray(candidate[0] - nominal[0], dtype=np.float64)
        applied_residual_l2 = float(np.linalg.norm(applied_delta))
        applied_residual_rms_norm = float(
            np.sqrt(np.mean((applied_delta / np.maximum(self._ctrl_scale, 1e-12)) ** 2))
        )
        ctrl0 = np.asarray(candidate[0], dtype=np.float64)
        sat_tol = 1e-6 * np.maximum(self._ctrl_high - self._ctrl_low, 1.0)
        applied_saturation_fraction = float(
            np.mean((ctrl0 <= self._ctrl_low + sat_tol) | (ctrl0 >= self._ctrl_high - sat_tol))
        )
        spike_diag = self._spike_diagnostics()

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
            "diag_variance_min": (
                float(np.min(self._diag_variance))
                if self.sampling == SamplingOption.DIAG_LOWRANK else 1.0
            ),
            "diag_variance_max": (
                float(np.max(self._diag_variance))
                if self.sampling == SamplingOption.DIAG_LOWRANK else 1.0
            ),
            "icem_elites": 0 if self._icem_elites is None else int(len(self._icem_elites)),
            "spike_sampler_backend": self._spike_sampler_backend,
            "nominal_geometry_evaluated": bool(len(refined.positions)),
            "spike_synergies": int(self._spike_synergies.shape[0]),
            "spike_events_mean": float(self._last_spike_event_count_mean),
            **spike_diag,
            "nominal_cost": nominal_cost,
            "weighted_rollout_cost": weighted_rollout_cost,
            "best_finite_cost": best_finite_cost,
            "nominal_weighted_improvement": float(nominal_weighted_improvement),
            "nominal_weighted_improvement_rel": float(nominal_weighted_improvement_rel),
            "nominal_best_improvement": float(nominal_best_improvement),
            "nominal_best_improvement_rel": float(nominal_best_improvement_rel),
            "applied_residual_l2": applied_residual_l2,
            "applied_residual_rms_norm": applied_residual_rms_norm,
            "applied_saturation_fraction": applied_saturation_fraction,
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
        # Previous "total" ended before diagnostics, underreporting latency.
        action = candidate[0].copy()
        t_done = time.perf_counter()
        info["timing_ms"]["diagnostics"] = 1e3 * (t_done - t_update)
        info["timing_ms"]["total"] = 1e3 * (t_done - t_total)
        return action, info
