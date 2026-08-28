from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .base import SpatialPrior


@dataclass
class GeometricPrior(SpatialPrior):
    tangent_std: float | None = None
    normal_std: float | None = None

    def sample(self, track, s):
        ts = 0.5 * track.road_width if self.tangent_std is None else float(self.tangent_std)
        ns = 0.5 * track.road_width if self.normal_std is None else float(self.normal_std)
        ts2, ns2 = ts * ts, ns * ns

        if np.ndim(s) == 0:
            mean = np.asarray(track.sample(float(s)), dtype=np.float64)
            t = np.asarray(track.tangent(float(s)), dtype=np.float64)
            tx, ty = float(t[0]), float(t[1])
            # n=(-ty, tx); expand the two rank-one terms without temporary
            # outer products. This path is used once per PPO nominal step.
            cov = np.asarray([
                [ts2 * tx * tx + ns2 * ty * ty, (ts2 - ns2) * tx * ty],
                [(ts2 - ns2) * tx * ty, ts2 * ty * ty + ns2 * tx * tx],
            ], dtype=np.float64)
            return mean, cov

        arr = np.asarray(s, dtype=np.float64)
        mean = np.asarray(track.sample(arr), dtype=np.float64)
        tangent = np.asarray(track.tangent(arr), dtype=np.float64)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        cov = (
            ts2 * tangent[:, :, None] * tangent[:, None, :]
            + ns2 * normal[:, :, None] * normal[:, None, :]
        )
        return mean, cov
