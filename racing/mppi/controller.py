from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import math
import numpy as np

from .lbps import optimize_lbps_temperature, weighted_control_sequence
from .rollout import (
    RolloutCostConfig,
    evaluate_control_batch,
    rollout_policy_nominal,
    refine_policy_nominal,
)
from .spg import build_spg_factors, sample_joint_noise


class ControllerVariant(str, Enum):
    POLICY_NOMINAL = "policy_nominal"
    STANDARD_MPPI = "standard_mppi"
    SENSITIVITY_PROJECTED_GAUSSIAN_MPPI = "sensitivity_projected_gaussian_prior_mppi"


@dataclass
class ControllerConfig:
    control_dt: float = 0.02
    horizon: int = 15
    num_rollouts: int = 128
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

    # Racing rollout constraints/cost.
    hard_collision_clearance: float = 0.02
    fall_height_fraction: float = 0.45
    min_root_up: float = 0.15
    upright_weight: float = 0.05
    control_deviation_weight: float = 1e-4
    rollout_workers: int = 0

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.num_rollouts = max(1, int(self.num_rollouts))
        self.lbps_optimizer_iterations = max(8, int(self.lbps_optimizer_iterations))
        self.spg_lookahead_steps = max(1, int(self.spg_lookahead_steps))
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
        )

    def _build_nominal(self, data, current_s: float):
        """Build a policy-seeded nominal, computing sensitivities only when needed.

        Standard MPPI with ``nominal_refine_iterations == 0`` is deliberately
        cheap: it uses the default policy rollout directly and does *not* finite
        difference every joint.  Sensitivity construction is reserved for SPG
        or an explicitly requested nominal refinement.
        """
        start = self.robot.snapshot(data)
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

        need_spg = self.variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI
        need_refine = int(self.cfg.nominal_refine_iterations) > 0

        if not need_spg and not need_refine:
            refined = policy_rollout
            jac = None
            endpoints = policy_rollout.positions.copy()
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
            )

        endpoint_s, _ = self.track.project(endpoints)
        prior_mean, prior_cov = self.prior.sample(self.track, endpoint_s)
        return (
            start, policy_rollout, refined, jac, endpoints,
            np.asarray(prior_mean), np.asarray(prior_cov),
        )

    def _sample_standard(self, nominal: np.ndarray) -> np.ndarray:
        n, h, nu = self.cfg.num_rollouts, self.cfg.horizon, self.robot.nu
        std = self.robot.control_scale(
            fraction=self.cfg.joint_noise_fraction,
            unlimited_span=self.cfg.unlimited_control_span,
        )
        noise = self.rng.standard_normal((n, h, nu)) * std[None, None, :]
        rho = float(self.cfg.temporal_noise_smoothing)
        beta = math.sqrt(max(0.0, 1.0 - rho * rho))
        for t in range(1, h):
            noise[:, t] = rho * noise[:, t - 1] + beta * noise[:, t]
        controls = nominal[None, :, :] + noise
        controls = self.robot.clip_ctrl(controls, self.cfg.unlimited_control_span)
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
        default_std = self.robot.control_scale(
            fraction=self.cfg.joint_noise_fraction,
            unlimited_span=self.cfg.unlimited_control_span,
        )
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
        controls = self.robot.clip_ctrl(controls, self.cfg.unlimited_control_span)
        controls[0] = nominal
        return controls, factors

    def step(self, data, current_s: float) -> tuple[np.ndarray, dict[str, Any]]:
        start, policy_nom, refined, jac, endpoints, prior_mean, prior_cov = self._build_nominal(data, current_s)
        nominal = refined.controls

        if self.variant == ControllerVariant.POLICY_NOMINAL:
            return nominal[0].copy(), {
                "nominal": nominal,
                "policy_nominal": policy_nom.controls,
                "nominal_positions": refined.positions,
                "prior_mean": prior_mean,
                "spatial_covariance": prior_cov,
                "joint_task_jacobians": jac,
                "temperature": math.nan,
                "ess": 1.0,
                "finite_rollouts": 1,
            }

        factors = None
        if self.variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            controls, factors = self._sample_spg(nominal, jac, endpoints, prior_mean, prior_cov)
        else:
            controls = self._sample_standard(nominal)

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
        )

        if self.cfg.adaptive_temperature_lbps:
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
        candidate = self.robot.clip_ctrl(candidate, self.cfg.unlimited_control_span)
        best = int(np.argmin(costs)) if len(costs) else 0
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
        }
        return candidate[0].copy(), info
