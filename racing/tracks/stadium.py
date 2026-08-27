from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class StadiumTrack:
    """NASCAR-style stadium track expressed in a robot-local world frame.

    The canonical track is transformed so that ``start_s`` is exactly at
    ``origin_xy`` and its tangent is aligned with ``origin_yaw``. This lets
    every classic robot keep its own native reset pose while racing on the
    same geometric track.
    """

    width: float = 20.0
    height: float = 8.0
    road_width: float = 3.0
    origin_xy: tuple[float, float] = (0.0, 0.0)
    origin_yaw: float = 0.0

    @property
    def outer_radius(self) -> float:
        return 0.5 * self.height

    @property
    def centerline_radius(self) -> float:
        return self.outer_radius - 0.5 * self.road_width

    @property
    def left_arc_x(self) -> float:
        return self.outer_radius

    @property
    def right_arc_x(self) -> float:
        return self.width - self.outer_radius

    @property
    def center_y(self) -> float:
        return 0.5 * self.height

    @property
    def straight_length(self) -> float:
        return self.right_arc_x - self.left_arc_x

    @property
    def length(self) -> float:
        return 2.0 * self.straight_length + 2.0 * math.pi * self.centerline_radius

    @property
    def start_s(self) -> float:
        return 0.5 * self.straight_length

    def _canonical_sample(self, s: np.ndarray) -> np.ndarray:
        s = np.mod(np.asarray(s, dtype=np.float64), self.length)
        r = self.centerline_radius
        straight = self.straight_length
        yc = self.center_y
        arc = math.pi * r
        out = np.empty(s.shape + (2,), dtype=np.float64)

        m0 = s < straight
        out[m0, 0] = self.left_arc_x + s[m0]
        out[m0, 1] = yc - r

        m1 = (s >= straight) & (s < straight + arc)
        sr = s[m1] - straight
        theta = -0.5 * math.pi + sr / r
        out[m1, 0] = self.right_arc_x + r * np.cos(theta)
        out[m1, 1] = yc + r * np.sin(theta)

        m2 = (s >= straight + arc) & (s < 2.0 * straight + arc)
        st = s[m2] - straight - arc
        out[m2, 0] = self.right_arc_x - st
        out[m2, 1] = yc + r

        m3 = ~(m0 | m1 | m2)
        sl = s[m3] - 2.0 * straight - arc
        theta = 0.5 * math.pi + sl / r
        out[m3, 0] = self.left_arc_x + r * np.cos(theta)
        out[m3, 1] = yc + r * np.sin(theta)
        return out

    def _canonical_tangent(self, s: np.ndarray) -> np.ndarray:
        s = np.mod(np.asarray(s, dtype=np.float64), self.length)
        r = self.centerline_radius
        straight = self.straight_length
        arc = math.pi * r
        out = np.empty(s.shape + (2,), dtype=np.float64)

        m0 = s < straight
        out[m0] = (1.0, 0.0)
        m1 = (s >= straight) & (s < straight + arc)
        theta = -0.5 * math.pi + (s[m1] - straight) / r
        out[m1, 0] = -np.sin(theta)
        out[m1, 1] = np.cos(theta)
        m2 = (s >= straight + arc) & (s < 2.0 * straight + arc)
        out[m2] = (-1.0, 0.0)
        m3 = ~(m0 | m1 | m2)
        theta = 0.5 * math.pi + (s[m3] - 2.0 * straight - arc) / r
        out[m3, 0] = -np.sin(theta)
        out[m3, 1] = np.cos(theta)
        return out

    def _transform(self, xy: np.ndarray, vectors: bool = False) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        start = self._canonical_sample(np.asarray([self.start_s]))[0]
        c = math.cos(self.origin_yaw)
        sn = math.sin(self.origin_yaw)
        rot = np.asarray([[c, -sn], [sn, c]], dtype=np.float64)
        if vectors:
            return xy @ rot.T
        return (xy - start) @ rot.T + np.asarray(self.origin_xy, dtype=np.float64)

    def _inverse_transform(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        start = self._canonical_sample(np.asarray([self.start_s]))[0]
        c = math.cos(self.origin_yaw)
        sn = math.sin(self.origin_yaw)
        rot = np.asarray([[c, -sn], [sn, c]], dtype=np.float64)
        return (xy - np.asarray(self.origin_xy, dtype=np.float64)) @ rot + start

    def sample(self, s) -> np.ndarray:
        scalar = np.ndim(s) == 0
        arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        out = self._transform(self._canonical_sample(arr))
        return out[0] if scalar else out

    def tangent(self, s) -> np.ndarray:
        scalar = np.ndim(s) == 0
        arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        out = self._transform(self._canonical_tangent(arr), vectors=True)
        return out[0] if scalar else out

    def normal(self, s) -> np.ndarray:
        t = np.asarray(self.tangent(s), dtype=np.float64)
        if t.ndim == 1:
            return np.asarray([-t[1], t[0]], dtype=np.float64)
        return np.column_stack((-t[:, 1], t[:, 0]))

    def curvature(self, s) -> np.ndarray | float:
        """Signed centerline curvature in 1/m.

        The stadium is piecewise straight/circular.  Positive curvature follows
        the track's forward direction on both semicircles.
        """
        scalar = np.ndim(s) == 0
        arr = np.mod(np.atleast_1d(np.asarray(s, dtype=np.float64)), self.length)
        straight = self.straight_length
        arc = math.pi * self.centerline_radius
        on_arc = ((arr >= straight) & (arr < straight + arc)) | (arr >= 2.0 * straight + arc)
        out = np.where(on_arc, 1.0 / max(self.centerline_radius, 1e-12), 0.0)
        return float(out[0]) if scalar else out

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(xy, dtype=np.float64)
        scalar = p.ndim == 1
        if scalar:
            p = p[None, :]
        p = self._inverse_transform(p)
        px, py = p[:, 0], p[:, 1]
        r = self.centerline_radius
        xl, xr, yc = self.left_arc_x, self.right_arc_x, self.center_y
        straight = self.straight_length

        bx = np.clip(px, xl, xr)
        bd2 = (px - bx) ** 2 + (py - (yc - r)) ** 2
        bs = bx - xl

        tr = np.clip(np.arctan2(py - yc, px - xr), -0.5 * math.pi, 0.5 * math.pi)
        rx = xr + r * np.cos(tr)
        ry = yc + r * np.sin(tr)
        rd2 = (px - rx) ** 2 + (py - ry) ** 2
        rs = straight + r * (tr + 0.5 * math.pi)

        tx = np.clip(px, xl, xr)
        td2 = (px - tx) ** 2 + (py - (yc + r)) ** 2
        ts = straight + math.pi * r + (xr - tx)

        tl = np.arctan2(py - yc, px - xl)
        tl = np.where(tl < 0.5 * math.pi, tl + 2.0 * math.pi, tl)
        tl = np.clip(tl, 0.5 * math.pi, 1.5 * math.pi)
        lx = xl + r * np.cos(tl)
        ly = yc + r * np.sin(tl)
        ld2 = (px - lx) ** 2 + (py - ly) ** 2
        ls = 2.0 * straight + math.pi * r + r * (tl - 0.5 * math.pi)

        d2s = np.column_stack((bd2, rd2, td2, ld2))
        ss = np.column_stack((bs, rs, ts, ls))
        idx = np.argmin(d2s, axis=1)
        rows = np.arange(len(p))
        s_best = np.mod(ss[rows, idx], self.length)
        d2_best = d2s[rows, idx]
        if scalar:
            return np.asarray(s_best[0]), np.asarray(d2_best[0])
        return s_best, d2_best

    def signed_progress_delta(self, new_s: float, old_s: float) -> float:
        ds = float(new_s) - float(old_s)
        half = 0.5 * self.length
        if ds > half:
            ds -= self.length
        elif ds < -half:
            ds += self.length
        return ds

    def polyline(self, count: int = 500) -> np.ndarray:
        return self.sample(np.linspace(0.0, self.length, max(8, int(count)), endpoint=False))
