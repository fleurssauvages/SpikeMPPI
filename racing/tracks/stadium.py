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

    def __post_init__(self) -> None:
        # Cache the rigid transform used by every online geometry query.
        # Track geometry is immutable during a race, so rebuilding these tiny
        # arrays thousands of times per second is pure overhead.
        c = math.cos(float(self.origin_yaw))
        sn = math.sin(float(self.origin_yaw))
        self._rot = np.asarray([[c, -sn], [sn, c]], dtype=np.float64)
        self._origin = np.asarray(self.origin_xy, dtype=np.float64)
        # start_s lies halfway along the lower straight in canonical space.
        self._canonical_start = np.asarray(
            [self.left_arc_x + self.start_s, self.center_y - self.centerline_radius],
            dtype=np.float64,
        )

    def _canonical_sample_scalar(self, s: float) -> tuple[float, float]:
        s = float(s) % self.length
        r = self.centerline_radius
        straight = self.straight_length
        yc = self.center_y
        arc = math.pi * r
        if s < straight:
            return self.left_arc_x + s, yc - r
        if s < straight + arc:
            theta = -0.5 * math.pi + (s - straight) / r
            return self.right_arc_x + r * math.cos(theta), yc + r * math.sin(theta)
        if s < 2.0 * straight + arc:
            st = s - straight - arc
            return self.right_arc_x - st, yc + r
        theta = 0.5 * math.pi + (s - 2.0 * straight - arc) / r
        return self.left_arc_x + r * math.cos(theta), yc + r * math.sin(theta)

    def _canonical_tangent_scalar(self, s: float) -> tuple[float, float]:
        s = float(s) % self.length
        r = self.centerline_radius
        straight = self.straight_length
        arc = math.pi * r
        if s < straight:
            return 1.0, 0.0
        if s < straight + arc:
            theta = -0.5 * math.pi + (s - straight) / r
            return -math.sin(theta), math.cos(theta)
        if s < 2.0 * straight + arc:
            return -1.0, 0.0
        theta = 0.5 * math.pi + (s - 2.0 * straight - arc) / r
        return -math.sin(theta), math.cos(theta)

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
        if vectors:
            return xy @ self._rot.T
        return (xy - self._canonical_start) @ self._rot.T + self._origin

    def _inverse_transform(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return (xy - self._origin) @ self._rot + self._canonical_start

    def sample(self, s) -> np.ndarray:
        if np.ndim(s) == 0:
            x, y = self._canonical_sample_scalar(float(s))
            dx, dy = x - self._canonical_start[0], y - self._canonical_start[1]
            return np.asarray([
                dx * self._rot[0, 0] + dy * self._rot[0, 1] + self._origin[0],
                dx * self._rot[1, 0] + dy * self._rot[1, 1] + self._origin[1],
            ], dtype=np.float64)
        arr = np.asarray(s, dtype=np.float64)
        return self._transform(self._canonical_sample(arr))

    def tangent(self, s) -> np.ndarray:
        if np.ndim(s) == 0:
            x, y = self._canonical_tangent_scalar(float(s))
            return np.asarray([
                x * self._rot[0, 0] + y * self._rot[0, 1],
                x * self._rot[1, 0] + y * self._rot[1, 1],
            ], dtype=np.float64)
        arr = np.asarray(s, dtype=np.float64)
        return self._transform(self._canonical_tangent(arr), vectors=True)

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
        straight = self.straight_length
        arc = math.pi * self.centerline_radius
        if np.ndim(s) == 0:
            x = float(s) % self.length
            on_arc = (straight <= x < straight + arc) or x >= 2.0 * straight + arc
            return 1.0 / max(self.centerline_radius, 1e-12) if on_arc else 0.0
        arr = np.mod(np.asarray(s, dtype=np.float64), self.length)
        on_arc = ((arr >= straight) & (arr < straight + arc)) | (arr >= 2.0 * straight + arc)
        return np.where(on_arc, 1.0 / max(self.centerline_radius, 1e-12), 0.0)

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(xy, dtype=np.float64)
        scalar = p.ndim == 1
        if scalar:
            # Allocation-free scalar path. This is used heavily by the policy
            # nominal and by the online race loop.
            dx = float(p[0]) - float(self._origin[0])
            dy = float(p[1]) - float(self._origin[1])
            px = dx * self._rot[0, 0] + dy * self._rot[1, 0] + self._canonical_start[0]
            py = dx * self._rot[0, 1] + dy * self._rot[1, 1] + self._canonical_start[1]
            r = self.centerline_radius
            xl, xr, yc = self.left_arc_x, self.right_arc_x, self.center_y
            straight = self.straight_length

            bx = min(max(px, xl), xr)
            candidates = [((px - bx) ** 2 + (py - (yc - r)) ** 2, bx - xl)]

            tr = min(max(math.atan2(py - yc, px - xr), -0.5 * math.pi), 0.5 * math.pi)
            rx, ry = xr + r * math.cos(tr), yc + r * math.sin(tr)
            candidates.append(((px - rx) ** 2 + (py - ry) ** 2, straight + r * (tr + 0.5 * math.pi)))

            tx = min(max(px, xl), xr)
            candidates.append(((px - tx) ** 2 + (py - (yc + r)) ** 2, straight + math.pi * r + (xr - tx)))

            tl = math.atan2(py - yc, px - xl)
            if tl < 0.5 * math.pi:
                tl += 2.0 * math.pi
            tl = min(max(tl, 0.5 * math.pi), 1.5 * math.pi)
            lx, ly = xl + r * math.cos(tl), yc + r * math.sin(tl)
            candidates.append(((px - lx) ** 2 + (py - ly) ** 2, 2.0 * straight + math.pi * r + r * (tl - 0.5 * math.pi)))

            best_d2, best_s = min(candidates, key=lambda item: item[0])
            return np.asarray(best_s % self.length), np.asarray(best_d2)
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

        # Streaming minimum avoids allocating two (N, 4) temporary matrices.
        # This path is called on N*H rollout positions every MPPI update.
        s_best = bs.copy()
        d2_best = bd2.copy()
        mask = rd2 < d2_best
        d2_best[mask] = rd2[mask]
        s_best[mask] = rs[mask]
        mask = td2 < d2_best
        d2_best[mask] = td2[mask]
        s_best[mask] = ts[mask]
        mask = ld2 < d2_best
        d2_best[mask] = ld2[mask]
        s_best[mask] = ls[mask]
        s_best = np.mod(s_best, self.length)
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
