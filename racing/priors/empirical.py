from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .base import SpatialPrior


@dataclass
class EmpiricalPrior(SpatialPrior):
    """Embodiment-independent Frenet prior distilled from completed laps.

    Only the learned lateral mean and normal dispersion are transferred.
    Tangential variance remains fixed, preventing source-robot timing/speed
    from being transferred into the target MPPI proposal.
    """

    track_length: float
    lateral_mean: np.ndarray
    normal_variance: np.ndarray
    source_laps: int
    tangent_std: float
    normal_floor_std: float

    def __post_init__(self) -> None:
        self.lateral_mean = np.asarray(self.lateral_mean, dtype=np.float64).reshape(-1)
        self.normal_variance = np.asarray(self.normal_variance, dtype=np.float64).reshape(-1)
        if len(self.lateral_mean) != len(self.normal_variance) or len(self.lateral_mean) < 8:
            raise ValueError("EmpiricalPrior banks must have equal length >= 8")

    @property
    def samples(self) -> int:
        return len(self.lateral_mean)

    def _interp(self, bank: np.ndarray, s: np.ndarray) -> np.ndarray:
        u = np.mod(s, self.track_length) / self.track_length * self.samples
        i0 = np.floor(u).astype(np.int64) % self.samples
        w = u - np.floor(u)
        i1 = (i0 + 1) % self.samples
        return (1.0 - w) * bank[i0] + w * bank[i1]

    def sample(self, track, s):
        scalar = np.ndim(s) == 0
        arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        center = np.asarray(track.sample(arr), dtype=np.float64)
        tangent = np.asarray(track.tangent(arr), dtype=np.float64)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        offset = self._interp(self.lateral_mean, arr)
        nvar = self._interp(self.normal_variance, arr)
        mean = center + offset[:, None] * normal
        cov = (
            self.tangent_std**2 * tangent[:, :, None] * tangent[:, None, :]
            + nvar[:, None, None] * normal[:, :, None] * normal[:, None, :]
        )
        if scalar:
            return mean[0], cov[0]
        return mean, cov

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            track_length=self.track_length,
            lateral_mean=self.lateral_mean,
            normal_variance=self.normal_variance,
            source_laps=self.source_laps,
            tangent_std=self.tangent_std,
            normal_floor_std=self.normal_floor_std,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EmpiricalPrior":
        with np.load(path) as d:
            return cls(
                track_length=float(d["track_length"]),
                lateral_mean=d["lateral_mean"],
                normal_variance=d["normal_variance"],
                source_laps=int(d["source_laps"]),
                tangent_std=float(d["tangent_std"]),
                normal_floor_std=float(d["normal_floor_std"]),
            )


def distill_empirical_prior(
    track,
    xy: np.ndarray,
    cumulative_progress: np.ndarray,
    completed_laps: int,
    *,
    samples: int = 512,
    normal_floor_std: float = 0.15,
    tangent_std: float | None = None,
) -> EmpiricalPrior:
    xy = np.asarray(xy, dtype=np.float64)
    progress = np.asarray(cumulative_progress, dtype=np.float64).reshape(-1)
    if xy.shape != (len(progress), 2):
        raise ValueError("xy must have shape [T,2] matching progress")
    if completed_laps < 1:
        raise ValueError("At least one completed lap is required")
    samples = max(16, int(samples))
    tangent_std = 0.5 * track.road_width if tangent_std is None else float(tangent_std)
    normal_floor_std = float(normal_floor_std)

    pmono = np.maximum.accumulate(progress)
    unique_p, unique_idx = np.unique(pmono, return_index=True)
    xy = xy[unique_idx]
    max_laps = min(int(completed_laps), int(np.floor(unique_p[-1] / track.length + 1e-9)))
    if max_laps < 1:
        raise ValueError("Progress history does not contain a complete lap")

    s_grid = np.arange(samples, dtype=np.float64) * track.length / samples
    center = np.asarray(track.sample(s_grid), dtype=np.float64)
    normal = np.asarray(track.normal(s_grid), dtype=np.float64)
    offsets = []
    for lap in range(max_laps):
        target = lap * track.length + s_grid
        if target[-1] > unique_p[-1]:
            break
        lap_xy = np.column_stack((
            np.interp(target, unique_p, xy[:, 0]),
            np.interp(target, unique_p, xy[:, 1]),
        ))
        offsets.append(np.einsum("mi,mi->m", lap_xy - center, normal))
    if not offsets:
        raise ValueError("Could not interpolate a complete lap")
    bank = np.stack(offsets, axis=0)
    lateral_mean = np.mean(bank, axis=0)
    if len(bank) >= 2:
        normal_var = np.var(bank - lateral_mean[None, :], axis=0, ddof=1)
    else:
        normal_var = np.zeros(samples, dtype=np.float64)
    normal_var = np.maximum(normal_var, normal_floor_std**2)
    return EmpiricalPrior(
        track_length=track.length,
        lateral_mean=lateral_mean,
        normal_variance=normal_var,
        source_laps=len(bank),
        tangent_std=tangent_std,
        normal_floor_std=normal_floor_std,
    )
