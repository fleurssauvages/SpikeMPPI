from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from racing.robots.model_params import ModelParameterScales
from racing.environments import RaceEnvironmentConfig
from racing.robots import make_robot
from racing.tracks import (
    StadiumTrack,
    close_viewer,
    configure_plain_floor,
    draw_race_overlay,
    draw_race_overlay_scene,
    launch_minimal_viewer,
    make_track_camera,
    safe_viewer_sync,
)


DEFAULT_REPLAY_ELEVATION = -20.0
DEFAULT_REPLAY_DISTANCE_SCALE = 0.60


def _scalar_text(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return str(value)


def load_recording(path: str | Path) -> dict[str, np.ndarray | str | float]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"robot_name", "qpos", "qvel", "state_time", "track", "plant_parameters"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise ValueError(
                "This recording predates exact-state replay and is missing: "
                + ", ".join(missing)
                + ". Run the race again with the updated project."
            )
        out: dict[str, np.ndarray | str | float] = {k: np.asarray(z[k]).copy() for k in z.files}
        out["robot_name"] = _scalar_text(z["robot_name"])
        out["controller_variant"] = _scalar_text(z["controller_variant"]) if "controller_variant" in z.files else "unknown"
        out["control_dt"] = float(np.asarray(z["control_dt"]).reshape(())) if "control_dt" in z.files else 0.02
    return out


def _prepare_replay(path: str | Path):
    record = load_recording(path)
    robot_name = str(record["robot_name"])

    track_values = np.asarray(record["track"], dtype=np.float64).reshape(-1)
    if len(track_values) < 6:
        raise ValueError("track recording is incomplete")
    track = StadiumTrack(
        width=float(track_values[0]),
        height=float(track_values[1]),
        road_width=float(track_values[2]),
        origin_xy=(float(track_values[3]), float(track_values[4])),
        origin_yaw=float(track_values[5]),
    )
    env_text = _scalar_text(record["environment_json"]) if "environment_json" in record else ""
    environment = RaceEnvironmentConfig.from_json(env_text)
    robot = make_robot(
        robot_name,
        extra_worldbody_xml=environment.plant_worldbody_xml(track),
        leg_length_scales=environment.leg_length_scales(robot_name),
    )
    robot.set_task_target_body(environment.task_body_name)

    plant_params = np.asarray(record["plant_parameters"], dtype=np.float64).reshape(-1)
    if len(plant_params) < 4:
        raise ValueError("plant_parameters must contain friction, mass, motor, slope_deg")
    robot.apply_model_parameters(
        ModelParameterScales(
            friction=float(plant_params[0]),
            mass=float(plant_params[1]),
            motor=float(plant_params[2]),
            slope_deg=float(plant_params[3]),
        )
    )

    qpos = np.asarray(record["qpos"], dtype=np.float64)
    qvel = np.asarray(record["qvel"], dtype=np.float64)
    state_time = np.asarray(record["state_time"], dtype=np.float64).reshape(-1)
    act = np.asarray(record.get("act", np.zeros((len(qpos), robot.model.na))), dtype=np.float64)
    controls = np.asarray(record.get("controls", np.zeros((max(0, len(qpos) - 1), robot.nu))), dtype=np.float64)

    if qpos.ndim != 2 or qpos.shape[1] != robot.model.nq:
        raise ValueError(f"recorded qpos has shape {qpos.shape}, expected (*, {robot.model.nq})")
    if qvel.ndim != 2 or qvel.shape[1] != robot.model.nv:
        raise ValueError(f"recorded qvel has shape {qvel.shape}, expected (*, {robot.model.nv})")
    if len(qpos) != len(qvel) or len(qpos) != len(state_time):
        raise ValueError("qpos, qvel, and state_time must have the same number of frames")
    if robot.model.na and (act.ndim != 2 or act.shape != (len(qpos), robot.model.na)):
        raise ValueError(f"recorded act has shape {act.shape}, expected ({len(qpos)}, {robot.model.na})")

    return record, robot, track, qpos, qvel, state_time, act, controls


def _frame_range(state_time: np.ndarray, start_s: float, end_s: float | None) -> tuple[np.ndarray, int, int]:
    relative_time = state_time - state_time[0]
    first = int(np.searchsorted(relative_time, max(0.0, float(start_s)), side="left"))
    if end_s is None:
        last = len(relative_time) - 1
    else:
        last = int(np.searchsorted(relative_time, max(float(start_s), float(end_s)), side="right") - 1)
        last = min(last, len(relative_time) - 1)
    if first >= len(relative_time) or last < first:
        raise ValueError("requested replay interval contains no recorded frames")
    return relative_time, first, last


def _set_recorded_state(robot, qpos, qvel, state_time, act, controls, i: int) -> None:
    robot.data.time = float(state_time[i])
    robot.data.qpos[:] = qpos[i]
    robot.data.qvel[:] = qvel[i]
    if robot.model.na:
        robot.data.act[:] = act[i]
    if robot.nu:
        if i == 0 or len(controls) == 0:
            robot.data.ctrl[:] = robot.default_ctrl
        else:
            robot.data.ctrl[:] = robot.clip_ctrl(controls[min(i - 1, len(controls) - 1)])
    robot.mujoco.mj_forward(robot.model, robot.data)


def export_gif(
    path: str | Path,
    gif_path: str | Path,
    *,
    speed: float = 1.0,
    fps: float | None = None,
    width: int = 960,
    height: int = 540,
    start_s: float = 0.0,
    end_s: float | None = None,
    camera_elevation: float = DEFAULT_REPLAY_ELEVATION,
    camera_distance_scale: float = DEFAULT_REPLAY_DISTANCE_SCALE,
) -> Path:
    """Render an exact-state replay to GIF without rerunning the controller."""
    try:
        import imageio.v2 as imageio
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "GIF export requires imageio and MuJoCo. Install project requirements first."
        ) from exc

    speed = float(speed)
    width = int(width)
    height = int(height)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("--speed must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("GIF dimensions must be positive")

    record, robot, track, qpos, qvel, state_time, act, controls = _prepare_replay(path)
    relative_time, first, last = _frame_range(state_time, start_s, end_s)

    # Default the GIF output rate to the controller frequency recorded by the
    # race.  The replay timeline is resampled to this output rate so changing
    # --gif-fps changes temporal resolution, while --speed changes playback
    # speed.  At the default rate (1/control_dt), each saved control state maps
    # to one GIF frame.
    control_dt = float(record.get("control_dt", np.nan))
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        diffs = np.diff(state_time[first : last + 1])
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if len(diffs) == 0:
            raise ValueError("cannot infer replay control frequency from recording")
        control_dt = float(np.median(diffs))
    source_fps = 1.0 / control_dt

    gif_fps = source_fps if fps is None else float(fps)
    if not np.isfinite(gif_fps) or gif_fps <= 0.0:
        raise ValueError("--gif-fps must be positive")

    sim_dt_per_frame = speed / gif_fps
    t0 = float(relative_time[first])
    t1 = float(relative_time[last])
    sample_times = np.arange(t0, t1 + 0.5 * sim_dt_per_frame, sim_dt_per_frame, dtype=np.float64)
    sample_times = np.minimum(sample_times, t1)
    indices = np.searchsorted(relative_time, sample_times, side="left")
    indices = np.clip(indices, first, last)
    prev = np.maximum(first, indices - 1)
    choose_prev = np.abs(relative_time[prev] - sample_times) < np.abs(relative_time[indices] - sample_times)
    indices = np.where(choose_prev, prev, indices)

    # GIF timing is stored in centiseconds.  The explicit GIF-PIL backend uses
    # ImageIO's seconds-based duration API.  The default Pillow v3 backend uses
    # milliseconds instead; passing 1/fps to it can round to a zero-delay frame
    # and make viewers fall back to an arbitrary (often slow) playback rate.
    frame_duration = 1.0 / gif_fps

    gif_path = Path(gif_path).expanduser().resolve()
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    configure_plain_floor(robot.model)
    # mujoco.Renderer rejects dimensions larger than the model's configured
    # off-screen framebuffer.  Grow that render-only buffer before creating the
    # renderer so common 16:9 GIF sizes work with classic Gymnasium XMLs.
    try:
        robot.model.vis.global_.offwidth = max(int(robot.model.vis.global_.offwidth), width)
        robot.model.vis.global_.offheight = max(int(robot.model.vis.global_.offheight), height)
    except Exception:
        pass

    camera = make_track_camera(
        track,
        elevation=float(camera_elevation),
        distance_scale=float(camera_distance_scale),
    )
    renderer = mujoco.Renderer(
        robot.model,
        height=height,
        width=width,
        max_geom=max(10000, int(robot.model.ngeom) + 512),
    )
    try:
        with imageio.get_writer(
            gif_path,
            format="GIF-PIL",
            mode="I",
            duration=frame_duration,
            loop=0,
        ) as writer:
            for i in indices:
                _set_recorded_state(robot, qpos, qvel, state_time, act, controls, int(i))
                renderer.update_scene(robot.data, camera=camera)
                draw_race_overlay_scene(renderer.scene, track, clear=False)
                writer.append_data(np.asarray(renderer.render(), dtype=np.uint8))
    finally:
        try:
            renderer.close()
        except Exception:
            pass

    print(
        f"wrote GIF {gif_path}: frames={len(indices)}, source_fps={source_fps:g}, "
        f"gif_fps={gif_fps:g}, size={width}x{height}, speed={speed:g}x, "
        f"robot={record['robot_name']}"
    )
    return gif_path


def replay(
    path: str | Path,
    *,
    speed: float = 1.0,
    loop: bool = False,
    show_ui: bool = False,
    start_s: float = 0.0,
    end_s: float | None = None,
    pause_at_end: float = 0.25,
    camera_elevation: float = DEFAULT_REPLAY_ELEVATION,
    camera_distance_scale: float = DEFAULT_REPLAY_DISTANCE_SCALE,
) -> None:
    """Replay the saved MuJoCo states without rerunning MPPI or the policy."""
    record, robot, track, qpos, qvel, state_time, act, controls = _prepare_replay(path)

    speed = float(speed)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("--speed must be positive")

    relative_time, first, last = _frame_range(state_time, start_s, end_s)

    viewer = launch_minimal_viewer(
        robot.model,
        robot.data,
        track=track,
        show_ui=show_ui,
        camera_elevation=float(camera_elevation),
        camera_distance_scale=float(camera_distance_scale),
    )
    try:
        draw_race_overlay(viewer, track)
        safe_viewer_sync(viewer, state_only=False)
        print(
            f"replay {Path(path).name}: robot={record['robot_name']}, controller={record['controller_variant']}, "
            f"frames={last-first+1}, speed={speed:g}x, camera_elevation={float(camera_elevation):g} deg"
        )

        keep_running = True
        while keep_running and viewer.is_running():
            wall0 = time.perf_counter()
            sim0 = float(relative_time[first])
            for i in range(first, last + 1):
                if not viewer.is_running():
                    keep_running = False
                    break

                target_wall = wall0 + (float(relative_time[i]) - sim0) / speed
                remaining = target_wall - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)

                _set_recorded_state(robot, qpos, qvel, state_time, act, controls, i)
                if not safe_viewer_sync(viewer, state_only=True):
                    keep_running = False
                    break

            if not loop:
                break
            if keep_running and pause_at_end > 0.0:
                time.sleep(float(pause_at_end))
    finally:
        close_viewer(viewer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved MuJoCo racing run")
    parser.add_argument("--file", default="racing/results/last_run.npz", help="recording produced by racing.experiments.race")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed; e.g. 0.5, 1, 2, 4")
    parser.add_argument("--loop", action="store_true", help="repeat until the viewer is closed")
    parser.add_argument("--viewer-ui", action="store_true", help="show MuJoCo side panels")
    parser.add_argument("--start", type=float, default=0.0, help="start time in recorded simulation seconds")
    parser.add_argument("--end", type=float, default=None, help="optional end time in recorded simulation seconds")
    parser.add_argument("--pause-at-end", type=float, default=0.25)
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=DEFAULT_REPLAY_ELEVATION,
        help="free-camera elevation in degrees; -90 is top-down, values closer to 0 are lower",
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=DEFAULT_REPLAY_DISTANCE_SCALE,
        help="multiplier on automatic full-track camera distance",
    )
    parser.add_argument("--gif", default=None, help="export replay to this GIF instead of opening the viewer")
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=None,
        help=(
            "GIF output frame rate; default is the recording's control frequency "
            "(1/control_dt). Use --speed to change playback speed"
        ),
    )
    parser.add_argument("--gif-width", type=int, default=960)
    parser.add_argument("--gif-height", type=int, default=540)
    args = parser.parse_args()

    if args.gif:
        export_gif(
            args.file,
            args.gif,
            speed=args.speed,
            fps=args.gif_fps,
            width=args.gif_width,
            height=args.gif_height,
            start_s=args.start,
            end_s=args.end,
            camera_elevation=args.camera_elevation,
            camera_distance_scale=args.camera_distance_scale,
        )
        return

    replay(
        args.file,
        speed=args.speed,
        loop=args.loop,
        show_ui=args.viewer_ui,
        start_s=args.start,
        end_s=args.end,
        pause_at_end=args.pause_at_end,
        camera_elevation=args.camera_elevation,
        camera_distance_scale=args.camera_distance_scale,
    )


if __name__ == "__main__":
    main()
