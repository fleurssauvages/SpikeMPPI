from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from racing.adaptation.model_params import ModelParameterScales
from .classic import ClassicModel, classic_xml_path


@dataclass
class RobotSnapshot:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray


class ClassicRobot:
    """Raw-MuJoCo adapter for the classic RL locomotion models.

    MPPI acts directly on ``MjData.ctrl``.  The class also supports applying a
    small set of parameter scales, which lets the physical plant and MPPI model
    intentionally differ for online adaptation experiments.
    """

    def __init__(self, info: ClassicModel, *, extra_worldbody_xml: str = "") -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Install mujoco before constructing ClassicRobot") from exc
        self.mujoco = mujoco
        self.info = info
        self.xml_path = classic_xml_path(info)

        # Keep the original robot dimensions even when the race environment adds
        # dynamic task objects (e.g. a free box).  The pretrained locomotion
        # policy must continue to see exactly the observation it was trained on.
        base_model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.robot_nq = int(base_model.nq)
        self.robot_nv = int(base_model.nv)
        self.robot_nbody = int(base_model.nbody)
        if str(extra_worldbody_xml).strip():
            self.model = self._load_augmented_model(str(extra_worldbody_xml))
        else:
            self.model = base_model
        self.data = mujoco.MjData(self.model)

        self._baseline_geom_friction = np.asarray(self.model.geom_friction, dtype=np.float64).copy()
        self._baseline_body_mass = np.asarray(self.model.body_mass, dtype=np.float64).copy()
        self._baseline_body_inertia = np.asarray(self.model.body_inertia, dtype=np.float64).copy()
        self._baseline_gainprm = np.asarray(self.model.actuator_gainprm, dtype=np.float64).copy()
        self._baseline_gravity = np.asarray(self.model.opt.gravity, dtype=np.float64).copy()
        self._parameter_scales = ModelParameterScales()
        self._ground_geom_ids = self._find_ground_geoms()

        self.reset()
        self.root_body_id = self._find_root_body()
        self.root_body_name = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_id)
            or f"body_{self.root_body_id}"
        )
        self.initial_root_height = float(self.data.xpos[self.root_body_id, 2])
        self.default_ctrl = self._default_ctrl()
        self._task_body_id = int(self.root_body_id)
        self._task_qpos_adr = self._find_free_qpos_adr(self._task_body_id)
        self._task_rest_height = (
            float(self.model.qpos0[self._task_qpos_adr + 2])
            if self._task_qpos_adr is not None else float(self.initial_root_height)
        )

    def _load_augmented_model(self, worldbody_fragment: str):
        """Compile the classic XML with extra worldbody elements.

        Assets are supplied through MuJoCo's in-memory asset mechanism so this
        remains robust for classic XMLs that reference files relative to the
        Gymnasium asset directory.  No actuator/joint in the original robot is
        modified.
        """
        root = ET.fromstring(Path(self.xml_path).read_text(encoding="utf-8"))
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"MuJoCo XML has no <worldbody>: {self.xml_path}")
        wrapper = ET.fromstring(f"<race_extra>{worldbody_fragment}</race_extra>")
        for child in list(wrapper):
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
        return self.info.name

    @property
    def display_name(self) -> str:
        return self.info.display_name

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
        if self.model.nu:
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
