from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import xml.etree.ElementTree as ET

import numpy as np


BOX_BODY_NAME = "race_box"
TERRAIN_PREFIX = "race_terrain_"


@dataclass(frozen=True)
class RaceEnvironmentConfig:
    """Physical task/environment additions layered onto the classic robot XML.

    ``terrain`` is added to both the physical plant and the MPPI planner.
    The terrain is therefore a *known test-time task/environment change*: the
    pretrained locomotion policy was learned on flat ground, while MPPI receives
    the true ramp/stair/rock geometry and can adapt the flat-running nominal
    controls online.

    ``push_box`` follows the same transfer principle: the box is present in both
    plant and planner so candidate rollouts can predict robot-box contact.  The
    pretrained locomotion policy still receives only the original robot
    observation; the box state is hidden from the policy and used only as the
    MPPI progress target.
    """

    task: str = "run"  # run | push_box
    terrain: str = "flat"  # flat | ramps | stairs | rocky | mixed
    terrain_seed: int = 1
    terrain_scale: float = 1.0
    box_distance: float = 1.8
    # `box_size` is the square footprint edge.  The pushing crate is deliberately
    # lower than it is wide so Ant can make sustained body contact instead of
    # striking the lower edge like a kick.
    box_size: float = 0.90
    box_height: float = 0.45
    box_mass: float = 6.0
    box_friction: float = 0.60

    def validated(self) -> "RaceEnvironmentConfig":
        task = str(self.task).strip().lower()
        terrain = str(self.terrain).strip().lower()
        if task not in {"run", "push_box"}:
            raise ValueError("task must be 'run' or 'push_box'")
        if terrain not in {"flat", "ramps", "stairs", "rocky", "mixed"}:
            raise ValueError("terrain must be flat, ramps, stairs, rocky, or mixed")
        if task == "push_box" and terrain != "flat":
            raise ValueError("push_box is intentionally a flat-ground task; use --terrain flat")
        if not np.isfinite(self.terrain_scale) or self.terrain_scale <= 0.0:
            raise ValueError("terrain_scale must be positive")
        if not np.isfinite(self.box_distance) or self.box_distance <= 0.0:
            raise ValueError("box_distance must be positive")
        if not np.isfinite(self.box_size) or self.box_size <= 0.05:
            raise ValueError("box_size must be > 0.05 m")
        if not np.isfinite(self.box_height) or self.box_height <= 0.05:
            raise ValueError("box_height must be > 0.05 m")
        if not np.isfinite(self.box_mass) or self.box_mass <= 0.0:
            raise ValueError("box_mass must be positive")
        if not np.isfinite(self.box_friction) or self.box_friction <= 0.0:
            raise ValueError("box_friction must be positive")
        return RaceEnvironmentConfig(
            task=task,
            terrain=terrain,
            terrain_seed=int(self.terrain_seed),
            terrain_scale=float(self.terrain_scale),
            box_distance=float(self.box_distance),
            box_size=float(self.box_size),
            box_height=float(self.box_height),
            box_mass=float(self.box_mass),
            box_friction=float(self.box_friction),
        )

    @property
    def task_body_name(self) -> str | None:
        return BOX_BODY_NAME if self.task == "push_box" else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str | bytes | None) -> "RaceEnvironmentConfig":
        if text is None:
            return cls()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        text = str(text).strip()
        if not text:
            return cls()
        payload = json.loads(text)
        # Stage-8/11 replay JSON predates the independent box-height field and
        # used a cube whose height equaled box_size. Preserve exact old replay
        # geometry instead of silently applying the new low-crate default.
        if payload.get("task") == "push_box" and "box_height" not in payload:
            payload["box_height"] = float(payload.get("box_size", 0.90))
        return cls(**payload).validated()

    def plant_worldbody_xml(self, track) -> str:
        parts: list[str] = []
        if self.terrain != "flat":
            parts.append(build_terrain_worldbody_xml(
                track,
                kind=self.terrain,
                seed=self.terrain_seed,
                scale=self.terrain_scale,
            ))
        if self.task == "push_box":
            parts.append(build_box_worldbody_xml(
                track,
                distance=self.box_distance,
                size=self.box_size,
                height=self.box_height,
                mass=self.box_mass,
                friction=self.box_friction,
            ))
        return "\n".join(x for x in parts if x)

    def planner_worldbody_xml(self, track) -> str:
        # Test-time terrain/task geometry is known to MPPI.  The pretrained PPO
        # policy remains unchanged and was trained only on flat-ground running.
        # This makes terrain and box experiments task transfer, not hidden model
        # mismatch experiments.
        parts: list[str] = []
        if self.terrain != "flat":
            parts.append(build_terrain_worldbody_xml(
                track,
                kind=self.terrain,
                seed=self.terrain_seed,
                scale=self.terrain_scale,
            ))
        if self.task == "push_box":
            parts.append(build_box_worldbody_xml(
                track,
                distance=self.box_distance,
                size=self.box_size,
                height=self.box_height,
                mass=self.box_mass,
                friction=self.box_friction,
            ))
        return "\n".join(x for x in parts if x)


def _fmt(values) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return " ".join(f"{float(x):.10g}" for x in arr)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_ypr(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    qz = np.asarray([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
    qy = np.asarray([math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0])
    qx = np.asarray([math.cos(0.5 * roll), math.sin(0.5 * roll), 0.0, 0.0])
    q = _quat_mul(_quat_mul(qz, qy), qx)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def _pose_on_track(track, s: float, lateral: float = 0.0) -> tuple[np.ndarray, float]:
    p = np.asarray(track.sample(float(s)), dtype=np.float64)
    t = np.asarray(track.tangent(float(s)), dtype=np.float64)
    t /= max(float(np.linalg.norm(t)), 1e-12)
    n = np.asarray([-t[1], t[0]], dtype=np.float64)
    p = p + float(lateral) * n
    yaw = math.atan2(float(t[1]), float(t[0]))
    return p, yaw


def _static_box(
    name: str,
    track,
    s: float,
    *,
    length: float,
    width: float,
    height: float,
    lateral: float = 0.0,
    z: float | None = None,
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw_offset: float = 0.0,
    rgba=(0.34, 0.30, 0.25, 1.0),
) -> str:
    p, yaw = _pose_on_track(track, s, lateral)
    if z is None:
        z = 0.5 * float(height)
    quat = _quat_ypr(yaw + float(yaw_offset), float(pitch), float(roll))
    elem = ET.Element("geom", {
        "name": name,
        "type": "box",
        "pos": _fmt([p[0], p[1], z]),
        "quat": _fmt(quat),
        "size": _fmt([0.5 * length, 0.5 * width, 0.5 * height]),
        "friction": "1 0.005 0.0001",
        # Classic Gymnasium locomotion XMLs often set conaffinity=0 in the
        # default geom class.  Explicit masks are required for injected static
        # terrain to collide with the robot instead of inheriting that default.
        "contype": "1",
        "conaffinity": "1",
        "condim": "3",
        "rgba": _fmt(rgba),
    })
    return ET.tostring(elem, encoding="unicode")


def _ramp_span(
    track,
    s0: float,
    s1: float,
    *,
    prefix: str,
    scale: float,
    peak_height: float,
    segment_length: float,
) -> list[str]:
    """Tile a continuous up/down ramp over an entire track interval.

    The interval is sampled in track arc length, so this also follows curved
    sections.  Adjacent thin boxes overlap slightly to avoid collision seams.
    """
    span = max(float(s1) - float(s0), 1e-9)
    n = max(2, int(math.ceil(span / max(float(segment_length), 0.15))))
    ds = span / n
    peak = float(peak_height) * float(scale)
    thickness = 0.14
    width = 0.94 * float(track.road_width)
    out: list[str] = []

    def height_at(frac: float) -> float:
        f = min(max(float(frac), 0.0), 1.0)
        return peak * (1.0 - abs(2.0 * f - 1.0))

    for i in range(n):
        sa = float(s0) + i * ds
        sb = float(s0) + (i + 1) * ds
        za = height_at(i / n)
        zb = height_at((i + 1) / n)
        sm = 0.5 * (sa + sb)
        dz = zb - za
        pitch = math.atan2(dz, ds)
        length = 1.04 * math.hypot(ds, dz)
        z = 0.5 * (za + zb) + 0.5 * thickness * abs(math.cos(pitch))
        out.append(_static_box(
            f"{TERRAIN_PREFIX}{prefix}_{i:03d}", track, sm,
            length=length, width=width, height=thickness,
            z=z, pitch=-pitch, rgba=(0.45, 0.32, 0.18, 1.0),
        ))
    return out


def _stairs_span(
    track,
    s0: float,
    s1: float,
    *,
    prefix: str,
    scale: float,
    peak_height: float = 1.68,
    step_length: float = 0.48,
) -> list[str]:
    """Fill an entire track interval with ascending then descending stairs."""
    span = max(float(s1) - float(s0), 1e-9)
    n = max(5, int(math.ceil(span / max(float(step_length), 0.15))))
    if n % 2 == 0:
        n += 1  # ensure one exact crest step at frac=0.5
    ds = span / n
    peak = float(peak_height) * float(scale)
    width = 0.94 * float(track.road_width)
    out: list[str] = []
    for i in range(n):
        frac = (i + 0.5) / n
        h = peak * (1.0 - abs(2.0 * frac - 1.0))
        h = max(h, 0.04 * float(scale))
        out.append(_static_box(
            f"{TERRAIN_PREFIX}{prefix}_{i:03d}", track, float(s0) + (i + 0.5) * ds,
            length=1.04 * ds, width=width, height=h,
            rgba=(0.38, 0.38, 0.40, 1.0),
        ))
    return out


def _rock_patch(
    track,
    s0: float,
    s1: float,
    *,
    prefix: str,
    scale: float,
    rng: np.random.Generator,
    count: int,
) -> list[str]:
    out = []
    half_road = 0.5 * float(track.road_width)
    for i in range(int(count)):
        s = float(rng.uniform(s0, s1))
        # Stage 12: four times the previous linear rock dimensions.  Keep the
        # center sampling width-aware so the enlarged rocks stay predominantly
        # inside the 3 m road instead of hanging outside its edges.
        length = 4.0 * float(rng.uniform(0.35, 0.80))
        width = 4.0 * float(rng.uniform(0.25, 0.55))
        height = 4.0 * float(rng.uniform(0.08, 0.24)) * float(scale)
        lateral_limit = max(0.0, half_road - 0.5 * width - 0.04)
        lateral = float(rng.uniform(-lateral_limit, lateral_limit)) if lateral_limit > 0.0 else 0.0
        out.append(_static_box(
            f"{TERRAIN_PREFIX}{prefix}_{i:03d}", track, s,
            length=length, width=width, height=height, lateral=lateral,
            roll=float(rng.uniform(-0.22, 0.22)),
            pitch=float(rng.uniform(-0.22, 0.22)),
            yaw_offset=float(rng.uniform(-0.8, 0.8)),
            rgba=(0.30, 0.28, 0.26, 1.0),
        ))
    return out


def build_terrain_worldbody_xml(track, *, kind: str, seed: int = 1, scale: float = 1.0) -> str:
    """Create collidable terrain only on the top straight and second (left) turn."""
    kind = str(kind).strip().lower()
    if kind == "flat":
        return ""
    if kind not in {"ramps", "stairs", "rocky", "mixed"}:
        raise ValueError(f"unsupported terrain kind {kind!r}")

    r = float(track.centerline_radius)
    straight = float(track.straight_length)
    arc = math.pi * r
    top0 = straight + arc
    top1 = 2.0 * straight + arc
    turn0 = top1
    turn1 = float(track.length)
    rng = np.random.default_rng(int(seed))
    geoms: list[str] = []

    if kind == "ramps":
        # Stage 11: cover the *entire* upper straight and second/left turn.
        # Peaks are exactly 4x the stage-10 defaults before --terrain-scale:
        # 0.48 -> 1.92 m on the straight, 0.34 -> 1.36 m on the turn.
        geoms += _ramp_span(
            track, top0, top1, prefix="ramp_top", scale=scale,
            peak_height=1.92, segment_length=0.50,
        )
        geoms += _ramp_span(
            track, turn0, turn1, prefix="ramp_turn", scale=scale,
            peak_height=1.36, segment_length=0.36,
        )
    elif kind == "stairs":
        # The previous 0.42 m stair crest becomes 1.68 m (4x) and the
        # staircase now tiles each complete requested track section.
        geoms += _stairs_span(
            track, top0, top1, prefix="stairs_top", scale=scale,
            peak_height=1.68, step_length=0.50,
        )
        geoms += _stairs_span(
            track, turn0, turn1, prefix="stairs_turn", scale=scale,
            peak_height=1.68, step_length=0.36,
        )
    elif kind == "rocky":
        margin_top = min(0.7, 0.12 * (top1 - top0))
        margin_turn = min(0.5, 0.10 * (turn1 - turn0))
        geoms += _rock_patch(track, top0 + margin_top, top1 - margin_top, prefix="rock_top", scale=scale, rng=rng, count=34)
        geoms += _rock_patch(track, turn0 + margin_turn, turn1 - margin_turn, prefix="rock_turn", scale=scale, rng=rng, count=28)
    else:  # mixed
        # Preserve a heterogeneous transfer task: the full upper straight is
        # split between a 4x ramp and 4x stairs; the full second turn is rocky.
        top_mid = 0.5 * (top0 + top1)
        geoms += _ramp_span(
            track, top0, top_mid, prefix="mixed_ramp_top", scale=scale,
            peak_height=1.92, segment_length=0.50,
        )
        geoms += _stairs_span(
            track, top_mid, top1, prefix="mixed_stairs_top", scale=scale,
            peak_height=1.68, step_length=0.50,
        )
        geoms += _rock_patch(track, turn0, turn1, prefix="mixed_rock_turn", scale=scale, rng=rng, count=40)

    return "\n".join(geoms)


def build_box_worldbody_xml(
    track,
    *,
    distance: float = 1.8,
    size: float = 0.90,
    height: float = 0.45,
    mass: float = 6.0,
    friction: float = 0.60,
) -> str:
    s = float(track.start_s) + float(distance)
    p, yaw = _pose_on_track(track, s)
    half_xy = 0.5 * float(size)
    half_z = 0.5 * float(height)
    quat = _quat_ypr(yaw)
    body = ET.Element("body", {
        "name": BOX_BODY_NAME,
        "pos": _fmt([p[0], p[1], half_z + 0.004]),
        "quat": _fmt(quat),
    })
    ET.SubElement(body, "freejoint", {"name": f"{BOX_BODY_NAME}_joint"})

    # IMPORTANT: classic Ant/Humanoid models compile body inertia from geoms.
    # Some model/default combinations apply an inherited geom density even when
    # an injected geom specifies ``mass=...``.  That made --box-mass cosmetic
    # on Ant (0.90*0.90*0.45*5 = 1.8225 kg regardless of the requested mass).
    #
    # Use one physical collision geom and override its DENSITY explicitly.  The
    # compiler necessarily computes mass = density * volume, so this remains
    # exact under inertiafromgeom=true and also avoids any dependence on hidden
    # ballast geoms being retained by compiler optimizations.
    volume = float(size) * float(size) * float(height)
    density = float(mass) / max(volume, 1e-12)

    ET.SubElement(body, "geom", {
        "name": f"{BOX_BODY_NAME}_geom",
        "type": "box",
        "size": _fmt([half_xy, half_xy, half_z]),
        "density": f"{density:.12g}",
        "friction": _fmt([float(friction), 0.01, 0.001]),
        "rgba": _fmt([0.92, 0.48, 0.08, 1.0]),
        # Do not inherit the classic robot's geom collision mask.  The box must
        # collide with both the robot and the floor/terrain.
        "contype": "1",
        "conaffinity": "1",
        "condim": "4",
        "group": "0",
    })
    return ET.tostring(body, encoding="unicode")


__all__ = [
    "BOX_BODY_NAME",
    "TERRAIN_PREFIX",
    "RaceEnvironmentConfig",
    "build_box_worldbody_xml",
    "build_terrain_worldbody_xml",
]
