from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .base import SpatialPrior


@dataclass
class GeometricPrior(SpatialPrior):
    tangent_std: float | None = None
    normal_std: float | None = None

    def sample(self, track, s):
        arr = np.asarray(s, dtype=np.float64)
        mean = np.asarray(track.sample(arr), dtype=np.float64)
        tangent = np.asarray(track.tangent(arr), dtype=np.float64)
        if tangent.ndim == 1:
            tangent = tangent[None, :]
            mean2 = mean[None, :] if mean.ndim == 1 else mean
            scalar = True
        else:
            mean2 = mean
            scalar = False
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        ts = 0.5 * track.road_width if self.tangent_std is None else float(self.tangent_std)
        ns = 0.5 * track.road_width if self.normal_std is None else float(self.normal_std)
        cov = np.empty((len(tangent), 2, 2), dtype=np.float64)
        for i, (t, n) in enumerate(zip(tangent, normal)):
            cov[i] = ts * ts * np.outer(t, t) + ns * ns * np.outer(n, n)
        if scalar:
            return mean2[0], cov[0]
        return mean2, cov
