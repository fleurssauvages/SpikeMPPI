from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .model_params import ModelParameterScales
from .classic import ClassicModel, classic_xml_path


ANT_LEG_ROOTS = (
    "front_left_leg",
    "front_right_leg",
    "back_leg",
    "right_back_leg",
)


def _scale_xyz_text(text: str, scale: float) -> str:
    values = [float(x) for x in str(text).split()]
    if len(values) != 3:
        raise ValueError(f"expected xyz triplet, got {text!r}")
    return " ".join(f"{scale * x:.10g}" for x in values)


def _scale_fromto_text(text: str, scale: float) -> str:
    values = [float(x) for x in str(text).split()]
    if len(values) != 6:
        raise ValueError(f"expected fromto sextuple, got {text!r}")
    return " ".join(f"{scale * x:.10g}" for x in values)


def apply_ant_leg_length_scales_xml(root: ET.Element, leg_scales: dict[str, float]) -> None:
    """Scale Ant limb segment lengths without changing its state/action topology.

    Gymnasium's classic Ant encodes each leg as a named root body followed by
    nested bodies whose local ``pos`` values are the segment attachment offsets,
    while capsule endpoints are encoded with ``fromto``. Scaling both keeps the
    joint graph, qpos/qvel layout, actuator mapping, radii, joint axes, and joint
    limits unchanged; only the physical limb lengths differ.
    """
    unknown = sorted(set(leg_scales).difference(ANT_LEG_ROOTS))
    if unknown:
        raise ValueError(f"unknown Ant leg roots: {', '.join(unknown)}")
    for leg_name, raw_scale in leg_scales.items():
        scale = float(raw_scale)
        if not np.isfinite(scale) or scale <= 0.05:
            raise ValueError(f"Ant leg scale for {leg_name} must be finite and > 0.05")
        leg = root.find(f".//body[@name='{leg_name}']")
        if leg is None:
            raise ValueError(f"classic Ant XML does not contain leg body {leg_name!r}")
        # The named leg-root body itself is attached at torso origin and should
        # remain there. Descendant body positions are physical segment offsets.
        for body in leg.iter("body"):
            if body is leg:
                continue
            if "pos" in body.attrib:
                body.set("pos", _scale_xyz_text(body.attrib["pos"], scale))
        for geom in leg.iter("geom"):
            if "fromto" in geom.attrib:
                geom.set("fromto", _scale_fromto_text(geom.attrib["fromto"], scale))


def replace_ant_motors_with_muscles_xml(root: ET.Element) -> list[tuple[str, str]]:
    """Replace classic Ant motors with antagonistic muscle-like actuators.

    Directly attaching MuJoCo's full Hill-type ``<muscle>`` shortcut to Ant's
    hinge coordinates makes the force-length-velocity curve depend on arbitrary
    joint-coordinate scaling.  In particular, the classic Ant reaches angular
    velocities that can push the shortcut far into its force-velocity roll-off.

    Ant-Bio uses an affine one-sided pulling-force law with *instantaneous*
    excitation-to-force mapping. Temporal activation/twitch dynamics live in the
    controller proposal: standard MPPI has none, fixed Spike uses its common
    twitch kernel, and Spike-Bio uses heterogeneous motor-unit twitch kernels.
    This avoids filtering those controller-side kernels a second time in MuJoCo.

    For one original joint with peak motor authority F0, the two actuators obey

        tau = F0 (a_pos - a_neg)
              - K (a_pos + a_neg) q
              - B (a_pos + a_neg) qdot,

    up to force clamping.  Thus unilateral activation is close to the original
    torque motor while simultaneous activation increases mechanical impedance.
    Each actuator remains one-sided: its scalar force is clamped to pulling
    force only, and its control range is [0, 1].

    Returns ``[(pos_name, neg_name), ...]`` in original actuator order.
    """
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("classic Ant XML has no <actuator> section")
    motors = list(actuator)
    if not motors:
        raise ValueError("classic Ant XML has no actuators to convert")

    pairs: list[tuple[str, str]] = []
    for index, motor in enumerate(motors):
        if motor.tag != "motor":
            raise ValueError(
                "ant-bio muscle conversion expects only <motor> actuators; "
                f"found <{motor.tag}>"
            )
        joint = motor.attrib.get("joint")
        if not joint:
            raise ValueError("ant-bio muscle conversion requires joint-transmission motors")
        gear_values = [float(x) for x in motor.attrib.get("gear", "1").split()]
        if len(gear_values) != 1:
            raise ValueError("ant-bio currently requires scalar motor gear values")
        motor_gear = float(gear_values[0])
        if not np.isfinite(motor_gear) or abs(motor_gear) <= 1e-12:
            raise ValueError(f"invalid motor gear for {joint!r}: {motor.attrib.get('gear')!r}")

        # Classic <motor> has fixed gain 1, so |gear| is its peak generalized
        # force at |ctrl|=1. Keep that authority explicitly in the muscle force
        # law rather than in transmission geometry.
        peak_force = abs(motor_gear)
        motor_sign = 1.0 if motor_gear > 0.0 else -1.0

        # Modest activation-dependent impedance. At the classic Ant F0=150 this
        # gives K=18 Nm/rad and B=1.5 Nms/rad per fully active muscle. With only
        # one muscle active this perturbs the ideal motor weakly; with both active
        # the stiffness/damping contributions add, giving physical co-contraction.
        stiffness = 0.12 * peak_force
        damping = 0.01 * peak_force
        pull_limit = 1.50 * peak_force

        base = motor.attrib.get("name") or joint or f"act{index}"
        pos_name = f"{base}_muscle_pos"
        neg_name = f"{base}_muscle_neg"

        def make_muscle(name: str, transmission_sign: float) -> ET.Element:
            # Both actuators generate only negative scalar force (pulling). The
            # opposite transmission signs map that pulling force to opposite joint
            # torques. With length=l=s*q and velocity=s*qdot, the shared affine
            # gain -F0-K*l-B*v yields the closed-form joint torque documented above.
            gear = float(transmission_sign)
            return ET.Element("general", {
                "name": name,
                "joint": joint,
                "gear": f"{gear:.12g}",
                "dyntype": "none",
                "ctrllimited": "true",
                "ctrlrange": "0 1",
                "gaintype": "affine",
                "gainprm": f"{-peak_force:.12g} {-stiffness:.12g} {-damping:.12g}",
                "biastype": "none",
                "forcelimited": "true",
                "forcerange": f"{-pull_limit:.12g} 0",
            })

        # Scalar pulling force is negative. Negative transmission therefore gives
        # positive original-motor torque; positive transmission gives the antagonist.
        pos = make_muscle(pos_name, -motor_sign)
        neg = make_muscle(neg_name, motor_sign)
        actuator.remove(motor)
        actuator.append(pos)
        actuator.append(neg)
        pairs.append((pos_name, neg_name))

    return pairs


@dataclass
class RobotSnapshot:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray


class ClassicRobot:
    """Raw-MuJoCo adapter for the classic RL locomotion models.

    MPPI acts directly on ``MjData.ctrl``. The class also supports fixed
    parameter scales so the physical plant can be configured independently from
    the planning model for controlled mismatch experiments.
    """

    def __init__(
        self,
        info: ClassicModel,
        *,
        extra_worldbody_xml: str = "",
        leg_length_scales: dict[str, float] | None = None,
        actuator_model: str = "motor",
        variant_name: str | None = None,
        variant_display_name: str | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Install mujoco before constructing ClassicRobot") from exc
        self.mujoco = mujoco
        self.info = info
        self.xml_path = classic_xml_path(info)
        self.actuator_model = str(actuator_model).strip().lower()
        if self.actuator_model not in {"motor", "muscle"}:
            raise ValueError("actuator_model must be 'motor' or 'muscle'")
        if self.actuator_model == "muscle" and self.info.name != "ant":
            raise ValueError("antagonistic muscle conversion is currently implemented only for Ant")
        self._variant_name = str(variant_name or info.name)
        self._variant_display_name = str(variant_display_name or info.display_name)
        self._muscle_pairs: tuple[tuple[int, int], ...] = ()

        # Keep the original robot dimensions even when the race environment adds
        # dynamic task objects (e.g. a free box). These dimensions are used by
        # rollout/state helpers independently of added task bodies.
        base_model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.robot_nq = int(base_model.nq)
        self.robot_nv = int(base_model.nv)
        self.robot_nbody = int(base_model.nbody)
        # Reference generalized-force authority of the original classic motors.
        # For a MuJoCo motor shortcut, force = gain * ctrl and the scalar joint
        # transmission contributes ``gear`` to qfrc_actuator.  Ant-Bio uses this
        # as a compile-time calibration target instead of assuming that
        # muscle ``force=1`` happens to reproduce the same joint torque.
        if int(base_model.nu):
            self._source_motor_peak_force = np.abs(
                np.asarray(base_model.actuator_gear[:, 0], dtype=np.float64)
                * np.asarray(base_model.actuator_gainprm[:, 0], dtype=np.float64)
            )
        else:
            self._source_motor_peak_force = np.zeros(0, dtype=np.float64)
        leg_length_scales = dict(leg_length_scales or {})
        if leg_length_scales and self.info.name != "ant":
            raise ValueError("leg-length morphology transfer is currently implemented only for Ant")
        if str(extra_worldbody_xml).strip() or leg_length_scales or self.actuator_model == "muscle":
            self.model = self._load_augmented_model(
                str(extra_worldbody_xml), leg_length_scales=leg_length_scales
            )
        else:
            self.model = base_model
        self.data = mujoco.MjData(self.model)
        self._muscle_force_calibration = np.ones(int(self.model.nu), dtype=np.float64)
        if self.actuator_model == "muscle":
            pairs: list[tuple[int, int]] = []
            # The converter emits adjacent positive/negative muscles in original
            # joint-actuator order. Keep explicit indices for Spike-Bio decoding.
            if int(self.model.nu) % 2 != 0:
                raise ValueError("ant-bio muscle actuator count must be even")
            for k in range(0, int(self.model.nu), 2):
                pairs.append((k, k + 1))
            self._muscle_pairs = tuple(pairs)
            # Peak force and impedance are encoded directly in the affine pulling-force law.

        self._baseline_geom_friction = np.asarray(self.model.geom_friction, dtype=np.float64).copy()
        self._baseline_body_mass = np.asarray(self.model.body_mass, dtype=np.float64).copy()
        self._baseline_body_inertia = np.asarray(self.model.body_inertia, dtype=np.float64).copy()
        self._baseline_gainprm = np.asarray(self.model.actuator_gainprm, dtype=np.float64).copy()
        self._baseline_biasprm = np.asarray(self.model.actuator_biasprm, dtype=np.float64).copy()
        self._baseline_forcerange = np.asarray(self.model.actuator_forcerange, dtype=np.float64).copy()
        self._baseline_gravity = np.asarray(self.model.opt.gravity, dtype=np.float64).copy()
        self._parameter_scales = ModelParameterScales()
        self._ground_geom_ids = self._find_ground_geoms()

        self.reset()
        self.root_body_id = self._find_root_body()
        self.root_body_name = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_id)
            or f"body_{self.root_body_id}"
        )
        self._root_geom_mask = np.asarray(
            self.model.geom_bodyid == int(self.root_body_id), dtype=bool
        )
        self._ground_geom_mask = np.zeros(int(self.model.ngeom), dtype=bool)
        if len(self._ground_geom_ids):
            self._ground_geom_mask[self._ground_geom_ids] = True
        self.initial_root_height = float(self.data.xpos[self.root_body_id, 2])
        self.default_ctrl = self._default_ctrl()
        self._task_body_id = int(self.root_body_id)
        self._task_qpos_adr = self._find_free_qpos_adr(self._task_body_id)
        self._task_rest_height = (
            float(self.model.qpos0[self._task_qpos_adr + 2])
            if self._task_qpos_adr is not None else float(self.initial_root_height)
        )


    def _calibrate_muscle_force_to_source_motors(self) -> None:
        """Backward-compatible no-op.

        Ant-Bio authority is set analytically in the generated affine force law,
        with F0 equal to the source motor's peak generalized-force authority.
        """
        if self.is_muscle_model:
            self._muscle_force_calibration = np.ones(int(self.model.nu), dtype=np.float64)

    def _load_augmented_model(
        self,
        worldbody_fragment: str,
        *,
        leg_length_scales: dict[str, float] | None = None,
    ):
        """Compile the classic XML with race task/model additions.

        The historical argument name is retained for compatibility. In addition
        to worldbody children, the fragment may contain ``race_root_site``
        markers and model-level grouping sections such as ``tendon``.

        Assets are supplied through MuJoCo's in-memory asset mechanism so this
        remains robust for classic XMLs that reference files relative to the
        Gymnasium asset directory. Joint/state topology is preserved; the
        ``ant-bio`` variant replaces each original motor with an antagonistic
        MuJoCo muscle pair before compilation.
        """
        root = ET.fromstring(Path(self.xml_path).read_text(encoding="utf-8"))
        if leg_length_scales:
            apply_ant_leg_length_scales_xml(root, leg_length_scales)
        if self.actuator_model == "muscle":
            replace_ant_motors_with_muscles_xml(root)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"MuJoCo XML has no <worldbody>: {self.xml_path}")
        wrapper = ET.fromstring(f"<race_extra>{worldbody_fragment}</race_extra>")

        # Most race additions are ordinary worldbody geoms/bodies. Towing also
        # needs a site on the original robot root and a model-level <tendon>
        # section. Handle those two cases here while keeping the public
        # extra_worldbody_xml API backward compatible.
        model_level_sections = {
            "tendon", "equality", "contact", "sensor", "actuator",
            "keyframe", "custom",
        }
        for child in list(wrapper):
            if child.tag == "race_root_site":
                root_body = root.find(f".//body[@name='{self.info.root_body_hint}']")
                if root_body is None:
                    raise ValueError(
                        f"could not attach towing hitch: root body {self.info.root_body_hint!r} not found"
                    )
                name = child.attrib.get("name", "")
                if name and root.find(f".//site[@name='{name}']") is not None:
                    raise ValueError(f"duplicate injected root site {name!r}")
                root_body.append(ET.Element("site", dict(child.attrib)))
                continue

            if child.tag in model_level_sections:
                existing = root.find(child.tag)
                if existing is None:
                    root.append(child)
                else:
                    # Merge grouping sections such as <tendon> when the source
                    # robot XML already contains one.
                    for grandchild in list(child):
                        existing.append(grandchild)
                continue

            worldbody.append(child)
        xml = ET.tostring(root, encoding="unicode")

        assets = {}
        asset_root = Path(self.xml_path).parent
        for path in asset_root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(asset_root).as_posix()
                assets[rel] = path.read_bytes()
            except OSError:
                pass
        try:
            return self.mujoco.MjModel.from_xml_string(xml, assets=assets)
        except TypeError:
            # Compatibility with MuJoCo versions where ``assets`` is positional.
            return self.mujoco.MjModel.from_xml_string(xml, assets)

    @property
    def name(self) -> str:
        return self._variant_name

    @property
    def display_name(self) -> str:
        return self._variant_display_name

    @property
    def base_name(self) -> str:
        return self.info.name

    @property
    def is_muscle_model(self) -> bool:
        return self.actuator_model == "muscle"

    @property
    def muscle_pairs(self) -> tuple[tuple[int, int], ...]:
        return self._muscle_pairs

    @property
    def muscle_force_calibration(self) -> np.ndarray:
        """Legacy calibration diagnostic; explicit Ant-Bio XML scaling returns ones."""
        return np.asarray(self._muscle_force_calibration, dtype=np.float64).copy()

    @property
    def muscle_peak_force(self) -> np.ndarray:
        """Peak active pulling-force magnitude F0 for Ant-Bio."""
        if not self.is_muscle_model:
            return np.zeros(0, dtype=np.float64)
        return np.abs(np.asarray(self.model.actuator_gainprm[:, 0], dtype=np.float64)).copy()

    @property
    def muscle_active_stiffness(self) -> np.ndarray:
        """Activation-dependent stiffness coefficient K for each muscle."""
        if not self.is_muscle_model:
            return np.zeros(0, dtype=np.float64)
        return np.abs(np.asarray(self.model.actuator_gainprm[:, 1], dtype=np.float64)).copy()

    @property
    def muscle_active_damping(self) -> np.ndarray:
        """Activation-dependent damping coefficient B for each muscle."""
        if not self.is_muscle_model:
            return np.zeros(0, dtype=np.float64)
        return np.abs(np.asarray(self.model.actuator_gainprm[:, 2], dtype=np.float64)).copy()

    @property
    def motor_pool_count(self) -> int:
        return len(self._muscle_pairs) if self.is_muscle_model else int(self.model.nu)

    @property
    def navigation(self) -> str:
        return self.info.navigation

    @property
    def supports_stadium(self) -> bool:
        return self.navigation == "xy"

    @property
    def nu(self) -> int:
        return int(self.model.nu)

    @property
    def physics_dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def parameter_scales(self) -> ModelParameterScales:
        return self._parameter_scales

    def _find_ground_geoms(self) -> np.ndarray:
        ids: list[int] = []
        floor_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        if floor_id >= 0:
            ids.append(int(floor_id))
        plane_type = int(self.mujoco.mjtGeom.mjGEOM_PLANE)
        for gid in range(self.model.ngeom):
            if int(self.model.geom_type[gid]) == plane_type and gid not in ids:
                ids.append(gid)
                continue
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, gid
            )
            if name and str(name).startswith("race_terrain_") and gid not in ids:
                ids.append(gid)
        return np.asarray(ids, dtype=np.int32)

    def apply_model_parameters(self, params: ModelParameterScales) -> None:
        """Apply scales relative to the untouched XML model, never cumulatively."""
        p = params.clipped()
        self.model.geom_friction[:] = self._baseline_geom_friction
        if len(self._ground_geom_ids):
            self.model.geom_friction[self._ground_geom_ids] = (
                self._baseline_geom_friction[self._ground_geom_ids] * float(p.friction)
            )

        self.model.body_mass[:] = self._baseline_body_mass
        self.model.body_inertia[:] = self._baseline_body_inertia
        robot_end = min(int(self.robot_nbody), int(self.model.nbody))
        if robot_end > 1:
            self.model.body_mass[1:robot_end] = self._baseline_body_mass[1:robot_end] * float(p.mass)
            self.model.body_inertia[1:robot_end] = self._baseline_body_inertia[1:robot_end] * float(p.mass)

        self.model.actuator_gainprm[:] = self._baseline_gainprm
        self.model.actuator_biasprm[:] = self._baseline_biasprm
        self.model.actuator_forcerange[:] = self._baseline_forcerange
        if self.model.nu:
            if self.is_muscle_model:
                # Ant-Bio's affine pulling-force law is
                #   gain = -F0 - K*length - B*velocity.
                # Strength mismatch scales the whole force law and its pulling
                # clamp together, preserving the impedance-to-force ratio.
                self.model.actuator_gainprm[:, :3] = (
                    self._baseline_gainprm[:, :3] * float(p.motor)
                )
                self.model.actuator_forcerange[:] = (
                    self._baseline_forcerange * float(p.motor)
                )
            else:
                self.model.actuator_gainprm[:, 0] = self._baseline_gainprm[:, 0] * float(p.motor)

        g = float(np.linalg.norm(self._baseline_gravity))
        if g <= 1e-12:
            self.model.opt.gravity[:] = self._baseline_gravity
        else:
            theta = math.radians(float(p.slope_deg))
            # Positive slope means uphill in +world-x, so gravity has a -x component.
            self.model.opt.gravity[:] = np.asarray(
                [-g * math.sin(theta), 0.0, -g * math.cos(theta)], dtype=np.float64
            )

        self._parameter_scales = p
        try:
            self.mujoco.mj_setConst(self.model, self.data)
        except Exception:
            # Derived quantities needed by stepping are refreshed by mj_forward on
            # supported MuJoCo versions; mj_setConst is an optimization here.
            pass
        self.mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)

    def _find_root_body(self) -> int:
        hint = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, self.info.root_body_hint
        )
        if hint >= 0:
            return int(hint)
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == int(self.mujoco.mjtJoint.mjJNT_FREE):
                return int(self.model.jnt_bodyid[j])
        return 1 if self.model.nbody > 1 else 0

    def _find_free_qpos_adr(self, body_id: int) -> int | None:
        free_type = int(self.mujoco.mjtJoint.mjJNT_FREE)
        for j in range(int(self.model.njnt)):
            if int(self.model.jnt_bodyid[j]) == int(body_id) and int(self.model.jnt_type[j]) == free_type:
                return int(self.model.jnt_qposadr[j])
        return None

    @property
    def task_body_id(self) -> int:
        return int(self._task_body_id)

    @property
    def task_qpos_adr(self) -> int | None:
        return None if self._task_qpos_adr is None else int(self._task_qpos_adr)

    @property
    def task_rest_height(self) -> float:
        return float(self._task_rest_height)

    def set_task_target_body(self, name: str | None) -> None:
        if not name:
            body_id = int(self.root_body_id)
        else:
            body_id = int(self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, str(name)
            ))
            if body_id < 0:
                raise ValueError(f"task target body {name!r} does not exist in the MuJoCo model")
        qadr = self._find_free_qpos_adr(body_id)
        if qadr is None:
            raise ValueError("task target body must have a free joint for fast rollout tracking")
        self._task_body_id = body_id
        self._task_qpos_adr = qadr
        self._task_rest_height = float(self.model.qpos0[qadr + 2])

    def _default_ctrl(self) -> np.ndarray:
        if self.model.nu == 0:
            return np.zeros(0, dtype=np.float64)
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        ranges = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        for i in range(self.model.nu):
            if limited[i] and not (ranges[i, 0] <= 0.0 <= ranges[i, 1]):
                ctrl[i] = 0.5 * (ranges[i, 0] + ranges[i, 1])
        return ctrl

    def snapshot(self, data=None) -> RobotSnapshot:
        d = self.data if data is None else data
        return RobotSnapshot(
            time=float(d.time),
            qpos=np.asarray(d.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(d.qvel, dtype=np.float64).copy(),
            act=np.asarray(d.act, dtype=np.float64).copy(),
            ctrl=np.asarray(d.ctrl, dtype=np.float64).copy(),
        )

    def restore(self, snapshot: RobotSnapshot, data=None) -> None:
        d = self.data if data is None else data
        d.time = float(snapshot.time)
        d.qpos[:] = snapshot.qpos
        d.qvel[:] = snapshot.qvel
        if self.model.na:
            d.act[:] = snapshot.act
        if self.model.nu:
            d.ctrl[:] = snapshot.ctrl
        self.mujoco.mj_forward(self.model, d)

    def new_data(self, snapshot: RobotSnapshot | None = None):
        d = self.mujoco.MjData(self.model)
        self.restore(self.snapshot() if snapshot is None else snapshot, d)
        return d

    def xy(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        return np.asarray(d.xpos[self.root_body_id, :2], dtype=np.float64).copy()

    def task_xy(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        return np.asarray(d.xpos[self._task_body_id, :2], dtype=np.float64).copy()

    def task_height(self, data=None) -> float:
        d = self.data if data is None else data
        return float(d.xpos[self._task_body_id, 2])

    def task_up(self, data=None) -> float:
        d = self.data if data is None else data
        mat = np.asarray(d.xmat[self._task_body_id], dtype=np.float64).reshape(3, 3)
        return float(mat[2, 2])

    def root_height(self, data=None) -> float:
        d = self.data if data is None else data
        return float(d.xpos[self.root_body_id, 2])

    def root_up(self, data=None) -> float:
        d = self.data if data is None else data
        mat = np.asarray(d.xmat[self.root_body_id], dtype=np.float64).reshape(3, 3)
        return float(mat[2, 2])

    def torso_touching_ground(self, data=None) -> bool:
        """Return True only for an actual torso-ground MuJoCo contact.

        The torso is defined as any geom attached directly to the free root body.
        Ground includes the model floor plane and injected race terrain geoms.
        Contacts with obstacles or other robot limbs do not count as ground contact.
        """
        d = self.data if data is None else data
        if int(d.ncon) <= 0 or not np.any(self._root_geom_mask) or not np.any(self._ground_geom_mask):
            return False
        for i in range(int(d.ncon)):
            contact = d.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if g1 < 0 or g2 < 0:
                continue
            if (self._root_geom_mask[g1] and self._ground_geom_mask[g2]) or (
                self._root_geom_mask[g2] and self._ground_geom_mask[g1]
            ):
                return True
        return False

    def has_fallen(self, data=None, *, flipped_threshold: float = 0.0) -> bool:
        """Ant is fallen only when flipped *and* its torso contacts ground."""
        d = self.data if data is None else data
        return self.root_up(d) < float(flipped_threshold) and self.torso_touching_ground(d)

    def root_yaw(self, data=None) -> float:
        d = self.data if data is None else data
        mat = np.asarray(d.xmat[self.root_body_id], dtype=np.float64).reshape(3, 3)
        return math.atan2(float(mat[1, 0]), float(mat[0, 0]))

    def control_bounds(self, unlimited_span: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if self.model.nu == 0:
            return np.zeros(0), np.zeros(0)
        low = np.full(self.model.nu, -abs(float(unlimited_span)), dtype=np.float64)
        high = np.full(self.model.nu, abs(float(unlimited_span)), dtype=np.float64)
        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        ranges = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        if ranges.shape == (self.model.nu, 2):
            low[limited] = ranges[limited, 0]
            high[limited] = ranges[limited, 1]
        return low, high

    def control_scale(
        self, fraction: float = 0.08, unlimited_span: float = 1.0, minimum: float = 1e-3
    ) -> np.ndarray:
        low, high = self.control_bounds(unlimited_span)
        return np.maximum(float(fraction) * (high - low), float(minimum))

    def clip_ctrl(self, controls: np.ndarray, unlimited_span: float = 1.0) -> np.ndarray:
        low, high = self.control_bounds(unlimited_span)
        return np.clip(np.asarray(controls, dtype=np.float64), low, high)

    def step_control(self, ctrl: np.ndarray, *, substeps: int = 1, data=None) -> None:
        d = self.data if data is None else data
        if self.model.nu:
            d.ctrl[:] = self.clip_ctrl(ctrl)
        # MuJoCo's Python binding supports nstep directly, so repeated physics
        # steps stay inside C++ without reacquiring the GIL between substeps.
        self.mujoco.mj_step(self.model, d, nstep=max(1, int(substeps)))
