from __future__ import annotations

import math
import time
import numpy as np


# Default race-view palette. The MuJoCo world is deliberately neutral so the
# track and robot remain easy to read from the overview camera.
FLOOR_RGBA = np.asarray([0.42, 0.42, 0.42, 1.0], dtype=np.float32)
ROAD_RGBA = np.asarray([0.08, 0.22, 0.34, 1.0], dtype=np.float32)
ROAD_ALT_RGBA = np.asarray([0.06, 0.28, 0.34, 1.0], dtype=np.float32)
CURB_RED_RGBA = np.asarray([0.90, 0.12, 0.10, 1.0], dtype=np.float32)
CURB_WHITE_RGBA = np.asarray([0.96, 0.96, 0.96, 1.0], dtype=np.float32)
CENTER_RGBA = np.asarray([1.00, 0.82, 0.12, 0.95], dtype=np.float32)
START_RGBA = np.asarray([0.98, 0.98, 0.98, 1.0], dtype=np.float32)


def configure_plain_floor(model, *, rgba=FLOOR_RGBA) -> None:
    """Make all MuJoCo plane geoms a plain, texture-free grey.

    Gymnasium's classic XMLs sometimes attach checker/grid materials to the
    ground plane. For racing we intentionally remove those materials from plane
    geoms and use a neutral grey so the visual-only track ribbon is prominent.
    This changes rendering only; contact/friction properties are untouched.
    """
    try:
        import mujoco

        plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
        color = np.asarray(rgba, dtype=np.float32).reshape(4)
        for gid in range(int(model.ngeom)):
            if int(model.geom_type[gid]) != plane_type:
                continue
            model.geom_rgba[gid] = color
            # A material can override geom_rgba and re-introduce a checker/grid
            # texture, so detach it for the floor only.
            try:
                model.geom_matid[gid] = -1
            except Exception:
                pass
    except Exception:
        # Viewer cosmetics must never prevent a simulation from running.
        return


def _add_line(mujoco, scn, p0, p1, rgba, width=3.0) -> None:
    if scn.ngeom >= len(scn.geoms):
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    a = np.asarray([p0[0], p0[1], p0[2] if len(p0) > 2 else 0.02], dtype=np.float64)
    b = np.asarray([p1[0], p1[1], p1[2] if len(p1) > 2 else 0.02], dtype=np.float64)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, float(width), a, b)
    geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
    scn.ngeom += 1


def _add_flat_segment(mujoco, scn, p0, p1, width, rgba, *, z=0.006, thickness=0.003) -> None:
    """Add one thin visual-only box forming a strip between two XY points."""
    if scn.ngeom >= len(scn.geoms):
        return
    a = np.asarray(p0, dtype=np.float64).reshape(-1)[:2]
    b = np.asarray(p1, dtype=np.float64).reshape(-1)[:2]
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return

    theta = math.atan2(float(delta[1]), float(delta[0]))
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pos = np.asarray([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), float(z)], dtype=np.float64)
    size = np.asarray([0.5 * length + 0.015, 0.5 * float(width), float(thickness)], dtype=np.float64)

    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        size,
        pos,
        rot.reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _configure_track_camera(
    cam,
    track,
    *,
    margin: float = 1.20,
    elevation: float = -76.0,
    distance_scale: float = 1.0,
    lookat_z: float = 0.35,
) -> None:
    """Configure a MuJoCo free camera to frame the complete stadium track."""
    import mujoco

    points = np.asarray(track.polyline(512), dtype=np.float64)
    lo = np.min(points, axis=0) - 0.65 * float(track.road_width)
    hi = np.max(points, axis=0) + 0.65 * float(track.road_width)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1e-6)

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray([center[0], center[1], float(lookat_z)], dtype=np.float64)
    cam.elevation = float(elevation)
    cam.azimuth = 90.0 + math.degrees(float(track.origin_yaw))
    base_distance = max(10.0, float(margin) * float(max(span[0], 1.25 * span[1])))
    cam.distance = max(1.0, float(distance_scale) * base_distance)


def make_track_camera(
    track,
    *,
    margin: float = 1.20,
    elevation: float = -76.0,
    distance_scale: float = 1.0,
    lookat_z: float = 0.35,
):
    """Create an off-screen-capable MuJoCo camera with the race framing."""
    import mujoco

    cam = mujoco.MjvCamera()
    _configure_track_camera(
        cam,
        track,
        margin=margin,
        elevation=elevation,
        distance_scale=distance_scale,
        lookat_z=lookat_z,
    )
    return cam


def frame_track_camera(
    viewer,
    track,
    *,
    margin: float = 1.20,
    elevation: float = -76.0,
    distance_scale: float = 1.0,
    lookat_z: float = 0.35,
) -> None:
    """Set a fixed overview camera that contains the complete track."""
    if viewer is None or track is None:
        return
    try:
        with viewer.lock():
            _configure_track_camera(
                viewer.cam,
                track,
                margin=margin,
                elevation=elevation,
                distance_scale=distance_scale,
                lookat_z=lookat_z,
            )
    except Exception:
        return


def draw_race_overlay_scene(
    scn,
    track,
    *,
    prior=None,
    controller_info=None,
    z: float = 0.012,
    clear: bool = False,
) -> None:
    """Append the visual race overlay to an arbitrary MuJoCo scene.

    ``clear=True`` is appropriate for ``viewer.user_scn``.  Off-screen
    ``Renderer.scene`` already contains model geoms, so callers should append
    with ``clear=False`` there.
    """
    if scn is None:
        return
    import mujoco

    if clear:
        scn.ngeom = 0

    capacity = len(scn.geoms)
    reserve = 12
    usable = max(0, capacity - int(scn.ngeom) - reserve)
    count = max(16, min(48, usable // 4 if usable else 16))
    s_coord = np.linspace(0.0, track.length, count, endpoint=False)
    center = np.asarray(track.sample(s_coord), dtype=np.float64)
    normal = np.asarray(track.normal(s_coord), dtype=np.float64)
    half_w = 0.5 * float(track.road_width)
    outer = center + half_w * normal
    inner = center - half_w * normal

    curb_width = min(0.18, 0.08 * float(track.road_width))
    for i in range(count):
        j = (i + 1) % count
        curb_color = CURB_RED_RGBA if i % 2 == 0 else CURB_WHITE_RGBA
        _add_flat_segment(
            mujoco, scn, outer[i], outer[j], curb_width,
            curb_color, z=z + 0.004, thickness=0.002,
        )
        _add_flat_segment(
            mujoco, scn, inner[i], inner[j], curb_width,
            curb_color, z=z + 0.004, thickness=0.002,
        )
        if i % 2 == 0:
            _add_flat_segment(
                mujoco, scn, center[i], center[j], 0.055,
                CENTER_RGBA, z=z + 0.008, thickness=0.0015,
            )

    # start = np.asarray(track.sample(0.0), dtype=np.float64)
    # n0 = np.asarray(track.normal(0.0), dtype=np.float64)
    # a = start - half_w * n0
    # b = start + half_w * n0
    # _add_flat_segment(
    #     mujoco, scn, a, b, 0.12, START_RGBA,
    #     z=z + 0.010, thickness=0.0015,
    # )

    if prior is not None and getattr(prior, "source_laps", 0) > 0:
        remaining = max(0, len(scn.geoms) - scn.ngeom)
        if remaining > 2:
            prior_count = min(remaining - 1, 96)
            prior_s = np.linspace(0.0, track.length, prior_count, endpoint=False)
            mean, _ = prior.sample(track, prior_s)
            for i in range(len(mean)):
                a = np.asarray([mean[i, 0], mean[i, 1], z + 0.020])
                b = np.asarray([mean[(i + 1) % len(mean), 0], mean[(i + 1) % len(mean), 1], z + 0.020])
                _add_line(mujoco, scn, a, b, (0.70, 0.15, 0.92, 1.0), 4.0)

    if controller_info is not None:
        nominal = np.asarray(controller_info.get("nominal_positions", []), dtype=np.float64)
        if nominal.ndim == 2 and nominal.shape[1] == 2:
            for i in range(len(nominal) - 1):
                a = np.asarray([nominal[i, 0], nominal[i, 1], z + 0.030])
                b = np.asarray([nominal[i + 1, 0], nominal[i + 1, 1], z + 0.030])
                _add_line(mujoco, scn, a, b, (0.10, 0.62, 1.00, 1.0), 4.0)


def draw_race_overlay(viewer, track, *, prior=None, controller_info=None, z: float = 0.012) -> None:
    """Draw the visual-only race overlay in the passive MuJoCo viewer."""
    if viewer is None:
        return
    try:
        with viewer.lock():
            draw_race_overlay_scene(
                viewer.user_scn,
                track,
                prior=prior,
                controller_info=controller_info,
                z=z,
                clear=True,
            )
    except Exception:
        return

def launch_minimal_viewer(
    model,
    data,
    *,
    track=None,
    show_ui: bool = False,
    camera_elevation: float = -50.0,
    camera_distance_scale: float = 1.0,
    camera_lookat_z: float = 0.35,
):
    """Launch the passive viewer with neutral floor and full-track overview."""
    import mujoco.viewer

    configure_plain_floor(model)
    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=bool(show_ui),
        show_right_ui=bool(show_ui),
    )
    if track is not None:
        frame_track_camera(
            viewer,
            track,
            elevation=camera_elevation,
            distance_scale=camera_distance_scale,
            lookat_z=camera_lookat_z,
        )
    return viewer


def safe_viewer_sync(viewer, *, state_only: bool = True) -> bool:
    """Best-effort sync. Returns False once the window is closing/closed."""
    if viewer is None:
        return False
    try:
        if not viewer.is_running():
            return False
        viewer.sync(state_only=bool(state_only))
        return True
    except Exception:
        return False


def close_viewer(viewer, *, settle_s: float = 0.20) -> None:
    """Request viewer shutdown and give the render thread time to exit."""
    if viewer is None:
        return
    try:
        viewer.close()
    except Exception:
        pass
    if settle_s > 0.0:
        time.sleep(float(settle_s))
