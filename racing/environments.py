from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import xml.etree.ElementTree as ET

import numpy as np


BOX_BODY_NAME = "race_box"
SLED_BODY_NAME = "race_sled"
TOW_ROBOT_SITE_NAME = "race_tow_robot_hitch"
TOW_SLED_SITE_NAME = "race_tow_sled_hitch"
TOW_TENDON_NAME = "race_tow_rope"
TERRAIN_PREFIX = "race_terrain_"
TASK_COLLISION_TYPE = 2


@dataclass(frozen=True)
class RaceEnvironmentConfig:
    """Physical task/environment additions layered onto the classic robot XML.

    ``terrain`` is added to both the physical plant and the MPPI planner.
    The terrain is therefore a *known test-time task/environment change*: the
    pretrained locomotion policy was learned on flat ground, while MPPI receives
    the true ramp/stair/rock geometry and can adapt the flat-running nominal
    controls online.

    ``push_box`` follows the same transfer principle: the object is present in
    both plant and planner so candidate rollouts can predict robot-object
    contact. ``tow_sled`` similarly adds a free sled plus a limited spatial
    tendon (cable) from the robot root to the sled. The pretrained locomotion
    policy remains unchanged; MPPI sees the transferred task dynamics and uses
    the pushed/towed body as its progress target.
    """

    task: str = "run"  # run | push_box | tow_sled
    push_object: str = "box"  # box | ball
    terrain: str = "flat"  # flat | ramps | stairs | rocky | mixed
    terrain_seed: int = 1
    terrain_scale: float = 1.0

    # Known Ant morphology transfer. The physical plant and MPPI planner use
    # the same modified Ant geometry, while the pretrained PPO weights remain
    # those learned on the nominal classic Ant. Exactly two legs are lengthened
    # and the opposite pair shortened, without changing joints or actuators.
    leg_mismatch: str = "none"  # none | same_side | diagonal
    short_leg_scale: float = 0.75
    long_leg_scale: float = 1.25

    box_distance: float = 1.8
    # `box_size` is the square footprint edge.  The pushing crate is deliberately
    # lower than it is wide so Ant can make sustained body contact instead of
    # striking the lower edge like a kick.
    box_size: float = 0.90
    box_height: float = 0.45
    box_mass: float = 6.0
    box_friction: float = 0.60
    ball_rolling_friction: float = 0.03

    # Towing transfer task. The sled is spawned behind the robot and attached by
    # a unilateral spatial-tendon length limit, which behaves like a cable: it
    # can pull when taut but cannot push the sled.
    sled_distance: float = 1.8
    sled_length: float = 1.0
    sled_width: float = 0.80
    sled_height: float = 0.16
    sled_mass: float = 8.0
    sled_friction: float = 0.60
    sled_rope_length: float = 1.25

    def validated(self) -> "RaceEnvironmentConfig":
        task = str(self.task).strip().lower()
        push_object = str(self.push_object).strip().lower()
        terrain = str(self.terrain).strip().lower()
        leg_mismatch = str(self.leg_mismatch).strip().lower()
        if task not in {"run", "push_box", "tow_sled"}:
            raise ValueError("task must be 'run', 'push_box', or 'tow_sled'")
        if push_object not in {"box", "ball"}:
            raise ValueError("push_object must be 'box' or 'ball'")
        if terrain not in {"flat", "ramps", "stairs", "rocky", "mixed"}:
            raise ValueError("terrain must be flat, ramps, stairs, rocky, or mixed")
        if leg_mismatch not in {"none", "same_side", "diagonal"}:
            raise ValueError("leg_mismatch must be none, same_side, or diagonal")
        if not np.isfinite(self.short_leg_scale) or not (0.1 <= float(self.short_leg_scale) < 1.0):
            raise ValueError("short_leg_scale must be in [0.1, 1.0)")
        if not np.isfinite(self.long_leg_scale) or float(self.long_leg_scale) <= 1.0:
            raise ValueError("long_leg_scale must be > 1.0")
        if leg_mismatch != "none" and (task != "run" or terrain != "flat"):
            raise ValueError("leg mismatch is an isolated simple-racing experiment; use --task run --terrain flat")
        if task in {"push_box", "tow_sled"} and terrain != "flat":
            raise ValueError(f"{task} is intentionally a flat-ground task; use --terrain flat")
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
        if not np.isfinite(self.ball_rolling_friction) or self.ball_rolling_friction < 0.0:
            raise ValueError("ball_rolling_friction must be non-negative")
        if not np.isfinite(self.sled_distance) or self.sled_distance <= 0.0:
            raise ValueError("sled_distance must be positive")
        if not np.isfinite(self.sled_length) or self.sled_length <= 0.1:
            raise ValueError("sled_length must be > 0.1 m")
        if not np.isfinite(self.sled_width) or self.sled_width <= 0.1:
            raise ValueError("sled_width must be > 0.1 m")
        if not np.isfinite(self.sled_height) or self.sled_height <= 0.03:
            raise ValueError("sled_height must be > 0.03 m")
        if not np.isfinite(self.sled_mass) or self.sled_mass <= 0.0:
            raise ValueError("sled_mass must be positive")
        if not np.isfinite(self.sled_friction) or self.sled_friction <= 0.0:
            raise ValueError("sled_friction must be positive")
        if not np.isfinite(self.sled_rope_length) or self.sled_rope_length <= 0.1:
            raise ValueError("sled_rope_length must be > 0.1 m")
        return RaceEnvironmentConfig(
            task=task,
            push_object=push_object,
            terrain=terrain,
            terrain_seed=int(self.terrain_seed),
            terrain_scale=float(self.terrain_scale),
            leg_mismatch=leg_mismatch,
            short_leg_scale=float(self.short_leg_scale),
            long_leg_scale=float(self.long_leg_scale),
            box_distance=float(self.box_distance),
            box_size=float(self.box_size),
            box_height=float(self.box_height),
            box_mass=float(self.box_mass),
            box_friction=float(self.box_friction),
            ball_rolling_friction=float(self.ball_rolling_friction),
            sled_distance=float(self.sled_distance),
            sled_length=float(self.sled_length),
            sled_width=float(self.sled_width),
            sled_height=float(self.sled_height),
            sled_mass=float(self.sled_mass),
            sled_friction=float(self.sled_friction),
            sled_rope_length=float(self.sled_rope_length),
        )

    def leg_length_scales(self, robot_name: str) -> dict[str, float]:
        """Return the known Ant leg scales used by both plant and MPPI planner."""
        if self.leg_mismatch == "none":
            return {}
        if str(robot_name).strip().lower() != "ant":
            raise ValueError("--leg-mismatch is currently supported only with --robot ant")
        short = float(self.short_leg_scale)
        long = float(self.long_leg_scale)
        # Classic Ant names: `back_leg` is rear-left and `right_back_leg` is
        # rear-right. Keep one fixed pair so seeds do not change morphology.
        if self.leg_mismatch == "same_side":
            long_legs = {"front_left_leg", "back_leg"}
        elif self.leg_mismatch == "diagonal":
            long_legs = {"front_left_leg", "right_back_leg"}
        else:
            raise ValueError(f"unsupported leg mismatch {self.leg_mismatch!r}")
        all_legs = ("front_left_leg", "front_right_leg", "back_leg", "right_back_leg")
        return {name: (long if name in long_legs else short) for name in all_legs}

    # Backward-compatible alias for Stage-18 callers. The returned scales are no
    # longer plant-only; race.py applies them to both plant and planner.
    def plant_leg_length_scales(self, robot_name: str) -> dict[str, float]:
        return self.leg_length_scales(robot_name)

    @property
    def task_body_name(self) -> str | None:
        if self.task == "push_box":
            return BOX_BODY_NAME
        if self.task == "tow_sled":
            return SLED_BODY_NAME
        return None

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
            parts.append(build_push_object_worldbody_xml(
                track,
                object_type=self.push_object,
                distance=self.box_distance,
                size=self.box_size,
                height=self.box_height,
                mass=self.box_mass,
                friction=self.box_friction,
                ball_rolling_friction=self.ball_rolling_friction,
            ))
        elif self.task == "tow_sled":
            parts.append(build_sled_model_xml(
                track,
                distance=self.sled_distance,
                length=self.sled_length,
                width=self.sled_width,
                height=self.sled_height,
                mass=self.sled_mass,
                friction=self.sled_friction,
                rope_length=self.sled_rope_length,
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
            parts.append(build_push_object_worldbody_xml(
                track,
                object_type=self.push_object,
                distance=self.box_distance,
                size=self.box_size,
                height=self.box_height,
                mass=self.box_mass,
                friction=self.box_friction,
                ball_rolling_friction=self.ball_rolling_friction,
            ))
        elif self.task == "tow_sled":
            parts.append(build_sled_model_xml(
                track,
                distance=self.sled_distance,
                length=self.sled_length,
                width=self.sled_width,
                height=self.sled_height,
                mass=self.sled_mass,
                friction=self.sled_friction,
                rope_length=self.sled_rope_length,
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


def build_terrain_worldbody_xml(
    track,
    *,
    kind: str,
    seed: int = 1,
    scale: float = 1.0,
) -> str:
    """Create the original static primitive terrain used by plant and planner."""
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


def build_push_object_worldbody_xml(
    track,
    *,
    object_type: str = "box",
    distance: float = 1.8,
    size: float = 0.90,
    height: float = 0.45,
    mass: float = 6.0,
    friction: float = 0.60,
    ball_rolling_friction: float = 0.03,
) -> str:
    """Build the free object used by the pushing-transfer task.

    ``size`` is the box footprint edge for ``box`` and the ball diameter for
    ``ball``.  The requested mass is imposed through an explicit geom density
    so it remains exact with the classic locomotion XMLs'
    ``inertiafromgeom=true`` compiler setting.
    """
    object_type = str(object_type).strip().lower()
    if object_type not in {"box", "ball"}:
        raise ValueError("object_type must be 'box' or 'ball'")

    s = float(track.start_s) + float(distance)
    p, yaw = _pose_on_track(track, s)
    body = ET.Element("body", {"name": BOX_BODY_NAME})
    ET.SubElement(body, "freejoint", {"name": f"{BOX_BODY_NAME}_joint"})

    if object_type == "ball":
        radius = 0.5 * float(size)
        volume = (4.0 / 3.0) * math.pi * radius ** 3
        density = float(mass) / max(volume, 1e-12)
        body.set("pos", _fmt([p[0], p[1], radius + 0.004]))
        # Sphere orientation has no geometric meaning, but keeping the spawn yaw
        # makes the free-joint initialization convention identical to the box.
        body.set("quat", _fmt(_quat_ypr(yaw)))
        ET.SubElement(body, "geom", {
            "name": f"{BOX_BODY_NAME}_geom",
            "type": "sphere",
            "size": _fmt([radius]),
            "density": f"{density:.12g}",
            "friction": _fmt([float(friction), 0.01, float(ball_rolling_friction)]),
            "rgba": _fmt([0.20, 0.55, 0.95, 1.0]),
            # A distinct type prevents movable task objects from being paired
            # with one another while preserving robot/object and ground/object
            # contacts through conaffinity=1.
            "contype": str(TASK_COLLISION_TYPE),
            "conaffinity": "1",
            "condim": "6",
            "group": "0",
        })
        return ET.tostring(body, encoding="unicode")

    half_xy = 0.5 * float(size)
    half_z = 0.5 * float(height)
    body.set("pos", _fmt([p[0], p[1], half_z + 0.004]))
    body.set("quat", _fmt(_quat_ypr(yaw)))

    # Classic Ant/Humanoid models compile body inertia from geoms.  Override
    # density explicitly so --box-mass remains exact under inertiafromgeom=true.
    volume = float(size) * float(size) * float(height)
    density = float(mass) / max(volume, 1e-12)
    ET.SubElement(body, "geom", {
        "name": f"{BOX_BODY_NAME}_geom",
        "type": "box",
        "size": _fmt([half_xy, half_xy, half_z]),
        "density": f"{density:.12g}",
        "friction": _fmt([float(friction), 0.01, 0.001]),
        "rgba": _fmt([0.92, 0.48, 0.08, 1.0]),
        "contype": str(TASK_COLLISION_TYPE),
        "conaffinity": "1",
        "condim": "4",
        "group": "0",
    })
    return ET.tostring(body, encoding="unicode")



def build_sled_model_xml(
    track,
    *,
    distance: float = 1.8,
    length: float = 1.0,
    width: float = 0.80,
    height: float = 0.16,
    mass: float = 8.0,
    friction: float = 0.60,
    rope_length: float = 1.25,
) -> str:
    """Build a free sled plus a cable-like spatial tendon to the robot root.

    The returned fragment intentionally contains both a worldbody ``<body>`` and
    a model-level ``<tendon>`` section. ``ClassicRobot._load_augmented_model``
    routes those elements to the correct MuJoCo XML sections and turns the
    ``<race_root_site>`` marker into a site attached to the robot root body.

    The spatial tendon has a one-sided length limit [0, rope_length], so it is
    slack below the limit and only transmits tension when stretched: a tow rope,
    not a rigid drawbar. The sled itself remains a normal free MuJoCo body.
    """
    length = float(length)
    width = float(width)
    height = float(height)
    mass = float(mass)
    rope_length = float(rope_length)

    # Spawn behind the robot along the track direction. The front hitch site is
    # on the sled's leading face, so the initial cable is short/slightly slack.
    s = float(track.start_s) - float(distance)
    p, yaw = _pose_on_track(track, s)
    half = np.asarray([0.5 * length, 0.5 * width, 0.5 * height], dtype=np.float64)
    volume = max(length * width * height, 1e-12)
    density = mass / volume

    body = ET.Element("body", {
        "name": SLED_BODY_NAME,
        "pos": _fmt([p[0], p[1], half[2] + 0.004]),
        "quat": _fmt(_quat_ypr(yaw)),
    })
    ET.SubElement(body, "freejoint", {"name": f"{SLED_BODY_NAME}_joint"})
    ET.SubElement(body, "geom", {
        "name": f"{SLED_BODY_NAME}_geom",
        "type": "box",
        "size": _fmt(half),
        "density": f"{density:.12g}",
        "friction": _fmt([float(friction), 0.01, 0.001]),
        "rgba": _fmt([0.18, 0.22, 0.26, 1.0]),
        "contype": str(TASK_COLLISION_TYPE),
        "conaffinity": "1",
        "condim": "4",
        "group": "0",
    })
    ET.SubElement(body, "site", {
        "name": TOW_SLED_SITE_NAME,
        "type": "sphere",
        "pos": _fmt([half[0], 0.0, 0.0]),
        "size": "0.035",
        "rgba": _fmt([0.95, 0.75, 0.10, 1.0]),
        "group": "0",
    })

    # This marker is consumed before MuJoCo compilation and replaced by a site
    # on the original robot root body. Keeping it in the environment fragment
    # avoids modifying the pretrained robot XML on disk.
    root_site = ET.Element("race_root_site", {
        "name": TOW_ROBOT_SITE_NAME,
        "type": "sphere",
        "pos": _fmt([-0.20, 0.0, -0.25]),
        "size": "0.035",
        "rgba": _fmt([0.95, 0.75, 0.10, 1.0]),
        "group": "0",
    })

    tendon = ET.Element("tendon")
    spatial = ET.SubElement(tendon, "spatial", {
        "name": TOW_TENDON_NAME,
        "limited": "true",
        "range": _fmt([0.0, rope_length]),
        "width": "0.012",
        "rgba": _fmt([0.95, 0.72, 0.10, 1.0]),
        "margin": "0.005",
        "solreflimit": "0.01 1",
    })
    ET.SubElement(spatial, "site", {"site": TOW_ROBOT_SITE_NAME})
    ET.SubElement(spatial, "site", {"site": TOW_SLED_SITE_NAME})

    return "\n".join([
        ET.tostring(body, encoding="unicode"),
        ET.tostring(root_site, encoding="unicode"),
        ET.tostring(tendon, encoding="unicode"),
    ])

def build_box_worldbody_xml(
    track,
    *,
    distance: float = 1.8,
    size: float = 0.90,
    height: float = 0.45,
    mass: float = 6.0,
    friction: float = 0.60,
) -> str:
    """Backward-compatible wrapper for older callers/replays."""
    return build_push_object_worldbody_xml(
        track,
        object_type="box",
        distance=distance,
        size=size,
        height=height,
        mass=mass,
        friction=friction,
    )


__all__ = [
    "BOX_BODY_NAME",
    "SLED_BODY_NAME",
    "TOW_ROBOT_SITE_NAME",
    "TOW_SLED_SITE_NAME",
    "TOW_TENDON_NAME",
    "TERRAIN_PREFIX",
    "RaceEnvironmentConfig",
    "build_box_worldbody_xml",
    "build_push_object_worldbody_xml",
    "build_sled_model_xml",
    "build_terrain_worldbody_xml",
]
