from __future__ import annotations

"""Paper-style settings for *Rapid Locomotion via Reinforcement Learning*.

This module ports the parts of Margolis et al. that are morphology independent:

* a joint (v_x, yaw-rate) Grid Adaptive Curriculum,
* the velocity-tracking reward scales used by the released locomotion code,
* the PPO hyperparameters reported in the paper, and
* the paper's dynamics-randomization ranges as metadata.

The original paper targets the MIT Mini Cheetah with position targets and a PD
loop.  This project retains those settings for Ant and reuses the same PPO
configuration for the synthetic spinner/snake/crawler/biped suite.  Morphology-specific
reward additions live in morphology_rewards.py.
"""

from dataclasses import asdict, dataclass, replace
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class RapidRewardConfig:
    tracking_sigma: float = 0.25
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = 0.0
    torques: float = -1.0e-5
    dof_acc: float = -2.5e-7
    action_rate: float = -0.01
    only_positive_rewards: bool = True


@dataclass(frozen=True)
class RapidPPOConfig:
    # Table 4 in Margolis et al. (IJRR 2024 / RSS 2022).
    num_envs: int = 4096
    total_timesteps: int = 400_000_000
    discounting: float = 0.99
    gae_lambda: float = 0.95
    unroll_length: int = 21
    num_updates_per_batch: int = 5  # PPO epochs per rollout
    num_minibatches: int = 4
    batch_size: int = 1024  # 4 * 1024 = 4096 samples/minibatch group
    entropy_cost: float = 0.01
    clipping_epsilon: float = 0.2
    learning_rate: float = 1.0e-3
    vf_loss_coefficient: float = 1.0
    normalize_observations: bool = True
    max_grad_norm: float = 1.0
    policy_layers: tuple[int, ...] = (512, 256, 128)
    value_layers: tuple[int, ...] = (512, 256, 128)
    init_noise_std: float = 1.0


@dataclass(frozen=True)
class RapidDomainRandomizationConfig:
    # Table 2 of the paper.  The MJX trainer currently applies friction and
    # motor-strength randomization directly and push disturbances in-state.
    friction_min: float = 0.05
    friction_max: float = 4.0
    restitution_min: float = 0.0
    restitution_max: float = 1.0
    payload_mass_min_kg: float = -1.0
    payload_mass_max_kg: float = 3.0
    com_offset_min_m: float = -0.10
    com_offset_max_m: float = 0.10
    motor_strength_min: float = 0.90
    motor_strength_max: float = 1.10
    push_interval_s: float = 15.0
    max_push_velocity_xy: float = 1.0


@dataclass(frozen=True)
class RapidCurriculumConfig:
    # The paper starts with +/-1 and represents the command distribution on a
    # 0.5 m/s x 0.5 rad/s grid.  The published heatmaps use +/-6 axes.
    vx_min: float = -6.0
    vx_max: float = 6.0
    wz_min: float = -6.0
    wz_max: float = 6.0
    grid_step_vx: float = 0.5
    grid_step_wz: float = 0.5
    initial_vx_min: float = -1.0
    initial_vx_max: float = 1.0
    initial_wz_min: float = -1.0
    initial_wz_max: float = 1.0
    vy_min: float = -0.6
    vy_max: float = 0.6
    command_hold_s: float = 10.0
    forward_success_threshold: float = 0.80
    yaw_success_threshold: float = 0.50
    # We adapt the shared grid between PPO phases.  A 20M phase yields twenty
    # curriculum updates over a 400M-step run, enough to traverse the grid.
    phase_timesteps: int = 20_000_000
    frontier_eval_seconds: float = 4.0


class GridAdaptiveCurriculum:
    """Discrete joint distribution over forward velocity and yaw rate.

    The original implementation updates a shared command grid from per-bin
    tracking reward.  In our JAX/MJX port the shared host-side mask is updated
    between PPO phases, which avoids mutable global state inside a jitted/vmapped
    environment while retaining the key joint-distribution behavior.
    """

    def __init__(
        self,
        config: RapidCurriculumConfig | None = None,
        active: np.ndarray | None = None,
        certified: np.ndarray | None = None,
    ):
        self.config = config or RapidCurriculumConfig()
        c = self.config
        self.vx_values = _inclusive_axis(c.vx_min, c.vx_max, c.grid_step_vx)
        self.wz_values = _inclusive_axis(c.wz_min, c.wz_max, c.grid_step_wz)
        shape = (len(self.vx_values), len(self.wz_values))
        if active is None:
            vx_ok = (self.vx_values >= c.initial_vx_min - 1e-9) & (self.vx_values <= c.initial_vx_max + 1e-9)
            wz_ok = (self.wz_values >= c.initial_wz_min - 1e-9) & (self.wz_values <= c.initial_wz_max + 1e-9)
            self.active = vx_ok[:, None] & wz_ok[None, :]
        else:
            arr = np.asarray(active, dtype=bool)
            if arr.shape != shape:
                raise ValueError(f"active curriculum mask must have shape {shape}, got {arr.shape}")
            self.active = arr.copy()
        if certified is None:
            self.certified = np.zeros(shape, dtype=bool)
        else:
            arr = np.asarray(certified, dtype=bool)
            if arr.shape != shape:
                raise ValueError(f"certified curriculum mask must have shape {shape}, got {arr.shape}")
            self.certified = arr.copy()

    def active_cells(self) -> np.ndarray:
        idx = np.argwhere(self.active)
        if len(idx) == 0:
            return np.asarray([[0.0, 0.0]], dtype=np.float32)
        return np.column_stack((self.vx_values[idx[:, 0]], self.wz_values[idx[:, 1]])).astype(np.float32)

    def frontier_indices(self) -> list[tuple[int, int]]:
        """Return active, uncertified cells that define the learned envelope.

        Cells adjacent to inactive cells are the normal curriculum frontier.
        Uncertified cells on the global grid edge are also included so the last
        unlocked shell is actually tested before a fixed grid is considered
        exhausted.  Failed cells remain candidates and can be retried after the
        next PPO phase.
        """
        frontier: list[tuple[int, int]] = []
        ni_max, nj_max = self.active.shape[0] - 1, self.active.shape[1] - 1
        for i_raw, j_raw in np.argwhere(self.active & ~self.certified):
            i, j = int(i_raw), int(j_raw)
            touches_inactive = any(not self.active[ni, nj] for ni, nj in self._neighbors(i, j))
            on_grid_edge = i in (0, ni_max) or j in (0, nj_max)
            if touches_inactive or on_grid_edge:
                frontier.append((i, j))
        return frontier

    def frontier_cells(self) -> np.ndarray:
        inds = self.frontier_indices()
        return np.asarray([[self.vx_values[i], self.wz_values[j]] for i, j in inds], dtype=np.float32)

    def expand_from_successes(self, successful_indices: Iterable[tuple[int, int]]) -> int:
        before = int(np.count_nonzero(self.active))
        updated = self.active.copy()
        for i, j in successful_indices:
            i, j = int(i), int(j)
            self.certified[i, j] = True
            updated[i, j] = True
            for ni, nj in self._neighbors(i, j):
                updated[ni, nj] = True
        self.active = updated
        return int(np.count_nonzero(self.active)) - before

    def extend_forward_from_successes(
        self,
        successful_indices: Iterable[tuple[int, int]],
        *,
        max_forward_speed: float | None = None,
    ) -> int:
        """Append one +v_x curriculum row after the straight outer edge passes.

        This preserves the original grid spacing and one-neighbor-at-a-time
        curriculum shape, but removes the published +/-6 m/s plotting limit as
        a training ceiling.  Only the positive/forward edge grows because that
        is the racing direction.
        """
        edge_i = len(self.vx_values) - 1
        near_zero = 0.51 * float(self.config.grid_step_wz)
        # Inspect the certified mask rather than only this phase's successes so
        # --resume can continue past an already-certified old grid ceiling.
        passed_edge_js = [
            int(j)
            for j in range(len(self.wz_values))
            if self.certified[edge_i, j] and abs(float(self.wz_values[j])) <= near_zero
        ]
        if not passed_edge_js:
            return 0

        next_vx = float(self.vx_values[-1] + float(self.config.grid_step_vx))
        if max_forward_speed is not None and next_vx > float(max_forward_speed) + 1e-9:
            return 0

        old_active = self.active
        old_certified = self.certified
        self.vx_values = np.concatenate([self.vx_values, np.asarray([next_vx], dtype=np.float64)])
        self.active = np.zeros((old_active.shape[0] + 1, old_active.shape[1]), dtype=bool)
        self.certified = np.zeros_like(self.active)
        self.active[:-1, :] = old_active
        self.certified[:-1, :] = old_certified
        self.active[-1, passed_edge_js] = True
        self.config = replace(self.config, vx_max=next_vx)
        return len(passed_edge_js)

    def certified_cells(self, *, fallback_to_active: bool = False) -> np.ndarray:
        idx = np.argwhere(self.certified)
        if len(idx) == 0:
            if fallback_to_active:
                return self.active_cells()
            return np.zeros((0, 2), dtype=np.float32)
        return np.column_stack((self.vx_values[idx[:, 0]], self.wz_values[idx[:, 1]])).astype(np.float32)

    def index_of_cell(self, vx: float, wz: float) -> tuple[int, int]:
        i = int(np.argmin(np.abs(self.vx_values - float(vx))))
        j = int(np.argmin(np.abs(self.wz_values - float(wz))))
        return i, j

    def certified_forward_speed(self) -> float:
        cells = self.certified_cells(fallback_to_active=False)
        if len(cells) == 0:
            return 0.0
        near_zero_yaw = np.abs(cells[:, 1]) <= 0.51 * self.config.grid_step_wz
        candidates = cells[near_zero_yaw, 0]
        if candidates.size:
            return float(max(0.0, np.max(candidates)))
        return float(max(0.0, np.max(cells[:, 0])))

    def learned_forward_speed(self) -> float:
        if np.any(self.certified):
            return self.certified_forward_speed()
        cells = self.active_cells()
        near_zero_yaw = np.abs(cells[:, 1]) <= 0.51 * self.config.grid_step_wz
        candidates = cells[near_zero_yaw, 0]
        if candidates.size:
            return float(max(0.0, np.max(candidates)))
        return float(max(0.0, np.max(cells[:, 0])))

    def learned_yaw_rate(self) -> float:
        cells = self.certified_cells(fallback_to_active=True)
        return float(np.max(np.abs(cells[:, 1]))) if len(cells) else 0.0

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "vx_values": self.vx_values.tolist(),
            "wz_values": self.wz_values.tolist(),
            "active": self.active.astype(np.uint8).tolist(),
            "active_cells": self.active_cells().tolist(),
            "certified": self.certified.astype(np.uint8).tolist(),
            "certified_cells": self.certified_cells().tolist(),
            "learned_forward_speed": self.learned_forward_speed(),
            "learned_yaw_rate": self.learned_yaw_rate(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "GridAdaptiveCurriculum":
        cfg = RapidCurriculumConfig(**dict(value.get("config", {})))
        active = np.asarray(value["active"], dtype=bool)
        certified_value = value.get("certified")
        certified = None if certified_value is None else np.asarray(certified_value, dtype=bool)
        return cls(cfg, active=active, certified=certified)

    def _neighbors(self, i: int, j: int):
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < self.active.shape[0] and 0 <= nj < self.active.shape[1]:
                yield ni, nj


def _inclusive_axis(lo: float, hi: float, step: float) -> np.ndarray:
    count = int(round((float(hi) - float(lo)) / float(step)))
    return np.linspace(float(lo), float(hi), count + 1, dtype=np.float64)


__all__ = [
    "RapidRewardConfig",
    "RapidPPOConfig",
    "RapidDomainRandomizationConfig",
    "RapidCurriculumConfig",
    "GridAdaptiveCurriculum",
]
