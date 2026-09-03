from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Optional
import numpy as np

from racing.robots.model_params import ModelParameterScales
from racing.environments import RaceEnvironmentConfig
from racing.mppi import ControllerConfig, ControllerVariant, JointMPPIController
from racing.policies import make_policy
from racing.priors import EmpiricalPrior, GeometricPrior, SpatialPrior
from racing.robots import make_robot
from racing.tracks import (
    StadiumTrack,
    close_viewer,
    draw_race_overlay,
    launch_minimal_viewer,
    safe_viewer_sync,
)


@dataclass
class RaceResult:
    robot_name: str
    controller_variant: str
    control_dt: float
    plant_integrator: str
    planner_mode: str
    xy: np.ndarray
    target_xy: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    state_time: np.ndarray
    controls: np.ndarray
    cumulative_progress: np.ndarray
    temperatures: np.ndarray
    esses: np.ndarray
    lap_times: list[float]
    completed_laps: int
    requested_laps: int
    off_track: bool
    fell: bool
    runtime_s: float
    simulated_time_s: float
    track: StadiumTrack
    environment: RaceEnvironmentConfig
    plant_parameters: ModelParameterScales


def run_race(
    *,
    robot_name: str,
    laps: int = 1,
    prior: Optional[SpatialPrior] = None,
    policy_spec: str | None = None,
    policy_speed: float | None = None,
    variant: ControllerVariant | str = ControllerVariant.MPPI,
    num_rollouts: int = 32,
    horizon: int = 50,
    control_dt: float | None = None,
    lbps_delta: float = 0.9,
    nominal_refine_iterations: int = 0,
    joint_noise_fraction: float = 0.08,
    seed: int = 1,
    max_steps: int | None = None,
    viewer: bool = True,
    viewer_ui: bool = False,
    controller_overlay: bool = False,
    rollout_workers: int = 16,
    rollout_chunk_size: int = 0,
    warm_start: bool = True,
    plant_integrator: str = "model",
    planner_mode: str = "rk4",
    profile_controller: bool = False,
    disable_gc: bool = False,
    verbose: bool = True,
    friction_scale: float = 1.0,
    mass_scale: float = 1.0,
    motor_scale: float = 1.0,
    slope_deg: float = 0.0,
    task: str = "run",
    terrain: str = "flat",
    terrain_seed: int = 1,
    terrain_scale: float = 1.0,
    leg_mismatch: str = "none",
    short_leg_scale: float = 0.75,
    long_leg_scale: float = 1.25,
    box_distance: float = 1.8,
    box_size: float = 0.90,
    box_height: float = 0.45,
    box_mass: float = 6.0,
    box_friction: float = 0.60,
    sled_distance: float = 1.8,
    sled_length: float = 1.0,
    sled_width: float = 0.80,
    sled_height: float = 0.16,
    sled_mass: float = 8.0,
    sled_friction: float = 0.60,
    sled_rope_length: float = 1.25,
    push_box_progress_weight: float = 1.0,
    push_robot_progress_weight: float = 0.35,
    push_approach_weight: float = 1.00,
    push_box_max_lift: float = 0.12,
    push_box_min_up: float = 0.75,
    sled_progress_weight: float = 1.0,
    sled_robot_progress_weight: float = 0.25,
    sled_max_lift: float = 0.12,
    sled_min_up: float = 0.70,
) -> RaceResult:
    """Race a classic MuJoCo robot using a policy-seeded joint-space controller.

    The locomotion policy provides either the closed-loop nominal controller or
    the warm-start nominal sequence used by standard joint-space MPPI.

    ``plant`` is the rendered/physical environment and may be configured with
    fixed test-time perturbations. Known task/terrain/morphology changes are also
    compiled into ``planner``. Plant and planner integrators are selectable
    independently; both remain fixed for the duration of a run. Planner mode
    also selects the matching contact-solver profile.
    """
    # Build the track from the untouched robot reset pose, then compile task/terrain
    # additions into both plant and planner models.  PPO remains a flat-ground
    # pretrained policy, while MPPI is given the true test-time task geometry.
    probe = make_robot(robot_name)
    if probe.nu <= 0:
        raise ValueError(f"{probe.name} has no MuJoCo actuators (model.nu=0)")
    if not probe.supports_stadium:
        raise ValueError(
            f"{probe.display_name} is constrained to 1-D forward locomotion and cannot turn on the stadium track. "
            "Only Ant is supported for the 2-D race."
        )
    origin_xy = tuple(map(float, probe.xy()))
    origin_yaw = float(probe.root_yaw())
    track = StadiumTrack(origin_xy=origin_xy, origin_yaw=origin_yaw)
    environment = RaceEnvironmentConfig(
        task=task, terrain=terrain, terrain_seed=terrain_seed, terrain_scale=terrain_scale,
        leg_mismatch=leg_mismatch, short_leg_scale=short_leg_scale, long_leg_scale=long_leg_scale,
        box_distance=box_distance, box_size=box_size, box_height=box_height,
        box_mass=box_mass, box_friction=box_friction,
        sled_distance=sled_distance, sled_length=sled_length, sled_width=sled_width,
        sled_height=sled_height, sled_mass=sled_mass, sled_friction=sled_friction,
        sled_rope_length=sled_rope_length,
    ).validated()

    leg_scales = environment.leg_length_scales(robot_name)
    plant = make_robot(
        robot_name,
        extra_worldbody_xml=environment.plant_worldbody_xml(track),
        leg_length_scales=leg_scales,
    )
    planner = make_robot(
        robot_name,
        extra_worldbody_xml=environment.planner_worldbody_xml(track),
        leg_length_scales=leg_scales,
    )
    plant.set_task_target_body(environment.task_body_name)
    planner.set_task_target_body(environment.task_body_name)

    # Fail loudly if the source MJCF compiler changes/overrides the requested
    # free task-body mass. Ant uses inertiafromgeom=true, so the
    # environment builders impose mass through explicit geom density.
    compiled_task_mass_plant = None
    compiled_task_mass_planner = None
    if environment.task in {"push_box", "tow_sled"}:
        compiled_task_mass_plant = float(plant.model.body_mass[plant.task_body_id])
        compiled_task_mass_planner = float(planner.model.body_mass[planner.task_body_id])
        if environment.task == "push_box":
            requested_task_mass = float(environment.box_mass)
            mass_flag = "--box-mass"
            object_label = "box"
        else:
            requested_task_mass = float(environment.sled_mass)
            mass_flag = "--sled-mass"
            object_label = "sled"
        if not np.isclose(compiled_task_mass_plant, requested_task_mass, rtol=2e-6, atol=1e-9):
            raise RuntimeError(
                f"compiled plant {object_label} mass {compiled_task_mass_plant:.9g} kg does not match "
                f"{mass_flag} {requested_task_mass:.9g} kg"
            )
        if not np.isclose(compiled_task_mass_planner, requested_task_mass, rtol=2e-6, atol=1e-9):
            raise RuntimeError(
                f"compiled planner {object_label} mass {compiled_task_mass_planner:.9g} kg does not match "
                f"{mass_flag} {requested_task_mass:.9g} kg"
            )

    if int(plant.model.nq) != int(planner.model.nq) or int(plant.model.nv) != int(planner.model.nv):
        raise RuntimeError("plant/planner dynamic state dimensions differ after environment construction")

    plant_params = ModelParameterScales(
        friction=float(friction_scale),
        mass=float(mass_scale),
        motor=float(motor_scale),
        slope_deg=float(slope_deg),
    ).clipped()
    plant.apply_model_parameters(plant_params)
    planner.apply_model_parameters(ModelParameterScales())

    # Resolve the controller period before configuring planner physics.  The
    # fast-rk4 planner deliberately takes one RK4 step per control interval,
    # whereas rk4/implicitfast retain the source model timestep.
    prior = prior or GeometricPrior()
    policy = make_policy(policy_spec, race_speed=policy_speed, robot_name=robot_name)
    if control_dt is None:
        control_dt = float(getattr(policy, "control_dt", 0.02))
    control_dt = float(control_dt)
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")

    # Plant integrator remains independent. Planner physics is intentionally
    # fused into one mode so integrator, timestep, and contact profile cannot drift:
    #   rk4          -> RK4 + source timestep + source solver/contact settings
    #   fast-rk4     -> RK4 + one step/control + fast solver/contact profile
    #   implicitfast -> implicitfast + source timestep + the same fast profile
    integrator_names = {"model", "euler", "implicitfast"}
    plant_integrator = str(plant_integrator).strip().lower()
    planner_mode = str(planner_mode).strip().lower()
    if plant_integrator not in integrator_names:
        raise ValueError("plant_integrator must be model, euler, or implicitfast")
    if planner_mode not in {"rk4", "fast-rk4", "implicitfast"}:
        raise ValueError("planner_mode must be rk4, fast-rk4, or implicitfast")
    integrator_map = {
        "euler": plant.mujoco.mjtIntegrator.mjINT_EULER,
        "implicitfast": plant.mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    if plant_integrator != "model":
        plant.model.opt.integrator = integrator_map[plant_integrator]
        plant.mujoco.mj_forward(plant.model, plant.data)

    # The plant must always advance exactly one control interval, independently
    # of the planner timestep/substep count.
    plant_ratio = control_dt / max(float(plant.physics_dt), 1e-12)
    plant_control_substeps = max(1, int(round(plant_ratio)))
    plant_actual_dt = plant_control_substeps * float(plant.physics_dt)
    if abs(plant_actual_dt - control_dt) > 0.25 * float(plant.physics_dt):
        raise ValueError(
            f"control_dt={control_dt:g} is not close to an integer multiple of plant "
            f"MuJoCo timestep={plant.physics_dt:g}; nearest is {plant_actual_dt:g}."
        )

    # The planner mode is explicit: unlike the old `model` option, `rk4`
    # always selects RK4 regardless of the XML's original integrator.
    if planner_mode in {"rk4", "fast-rk4"}:
        planner.model.opt.integrator = planner.mujoco.mjtIntegrator.mjINT_RK4
    else:
        planner.model.opt.integrator = planner.mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    if planner_mode == "fast-rk4":
        # One RK4 step covers the whole control interval.  This halves the
        # planner mj_step count for the default 20 ms control / 10 ms model.
        planner.model.opt.timestep = control_dt

    if planner_mode in {"fast-rk4", "implicitfast"}:
        planner.model.opt.iterations = min(int(planner.model.opt.iterations), 20)
        planner.model.opt.ls_iterations = min(int(planner.model.opt.ls_iterations), 10)
        planner.model.opt.tolerance = max(float(planner.model.opt.tolerance), 1e-6)
        planner.model.opt.noslip_iterations = 0

    planner.mujoco.mj_forward(planner.model, planner.data)
    policy.reset(planner, planner.data)

    # Reuse the same fast rollout/fused cost ABI for pushing and towing. The task
    # body is the box or sled respectively. Towing does not need an
    # approach-to-object term because the cable already couples robot and sled.
    if environment.task == "tow_sled":
        task_progress_weight = float(sled_progress_weight)
        task_robot_progress_weight = float(sled_robot_progress_weight)
        task_approach_weight = 0.0
        task_max_lift = float(sled_max_lift)
        task_min_up = float(sled_min_up)
    else:
        task_progress_weight = float(push_box_progress_weight)
        task_robot_progress_weight = float(push_robot_progress_weight)
        task_approach_weight = float(push_approach_weight)
        task_max_lift = float(push_box_max_lift)
        task_min_up = float(push_box_min_up)

    cfg = ControllerConfig(
        control_dt=float(control_dt),
        horizon=int(horizon),
        num_rollouts=int(num_rollouts),
        lbps_delta=float(lbps_delta),
        nominal_refine_iterations=int(nominal_refine_iterations),
        joint_noise_fraction=float(joint_noise_fraction),
        rollout_workers=int(rollout_workers),
        rollout_chunk_size=int(rollout_chunk_size),
        warm_start=bool(warm_start),
        box_progress_weight=task_progress_weight,
        robot_progress_weight=task_robot_progress_weight,
        robot_box_approach_weight=task_approach_weight,
        box_max_lift=task_max_lift,
        box_min_up=task_min_up,
    )
    controller = JointMPPIController(planner, track, prior, policy, cfg, variant=variant, seed=seed)

    current_s, _ = track.project(plant.task_xy())
    current_s = float(current_s)
    cumulative = 0.0
    target = int(laps) * track.length
    max_steps = int(max_steps) if max_steps is not None else max(1000, 6000 * int(laps))

    initial_state = plant.snapshot()
    xy_hist = [plant.xy()]
    target_xy_hist = [plant.task_xy()]
    qpos_hist = [initial_state.qpos.copy()]
    qvel_hist = [initial_state.qvel.copy()]
    act_hist = [initial_state.act.copy()]
    state_time_hist = [float(initial_state.time)]
    controls: list[np.ndarray] = []
    progress_hist = [0.0]
    temperatures: list[float] = []
    esses: list[float] = []
    lap_times: list[float] = []
    completed = 0
    off_track = False
    fell = False
    profile_rows: list[dict[str, float]] = []

    handle = None
    if viewer:
        handle = launch_minimal_viewer(plant.model, plant.data, track=track, show_ui=viewer_ui)
        draw_race_overlay(
            handle,
            track,
            prior=prior if getattr(prior, "source_laps", 0) > 0 else None,
        )
        safe_viewer_sync(handle, state_only=False)

    if verbose:
        print(
            f"controller={controller.variant.value}  robot={plant.name}  nu={plant.nu}  "
            f"rollouts={cfg.num_rollouts}  H={cfg.horizon}  dt={cfg.control_dt:g}s  "
            f"planner={controller.rollout_backend_name}  "
            f"plant={plant_integrator}:{plant_control_substeps}x{plant.physics_dt:g}s  "
            f"planner_mode={planner_mode}:{controller.control_substeps}x{planner.physics_dt:g}s  "
            f"warm_start={cfg.warm_start}  task={environment.task} terrain={environment.terrain} "
            f"leg_mismatch={environment.leg_mismatch}"
        )
        if environment.leg_mismatch != "none":
            scale_text = ", ".join(f"{name}={scale:g}x" for name, scale in leg_scales.items())
            print(
                f"known Ant leg morphology: pattern={environment.leg_mismatch}  {scale_text}; "
                "plant/planner geometry=modified, PPO checkpoint=nominal pretrained policy"
            )
        if environment.task == "push_box":
            shape_desc = f"footprint={environment.box_size:g}m height={environment.box_height:g}m"
            print(
                "push reward: "
                f"box_progress={cfg.box_progress_weight:g}, "
                f"robot_progress={cfg.robot_progress_weight:g}, "
                f"approach={cfg.robot_box_approach_weight:g}; "
                f"object=box box_distance={environment.box_distance:g}m "
                f"{shape_desc} mass={environment.box_mass:g}kg "
                f"(compiled plant={compiled_task_mass_plant:g}kg, planner={compiled_task_mass_planner:g}kg)"
            )
        elif environment.task == "tow_sled":
            print(
                "tow reward: "
                f"sled_progress={cfg.box_progress_weight:g}, "
                f"robot_progress={cfg.robot_progress_weight:g}; "
                f"sled_distance={environment.sled_distance:g}m "
                f"size={environment.sled_length:g}x{environment.sled_width:g}x{environment.sled_height:g}m "
                f"mass={environment.sled_mass:g}kg rope={environment.sled_rope_length:g}m "
                f"(compiled plant={compiled_task_mass_plant:g}kg, planner={compiled_task_mass_planner:g}kg)"
            )

    gc_was_enabled = gc.isenabled()
    if disable_gc and gc_was_enabled:
        gc.disable()

    t0 = time.perf_counter()
    try:
        for step in range(max_steps):
            if handle is not None and not handle.is_running():
                break

            # The controller reads the physical qpos/qvel through the current data,
            # but all candidate rollouts use the separate planning MuJoCo model.
            ctrl, info = controller.step(plant.data, current_s)
            if profile_controller:
                tm_all = info.get("timing_ms", {})
                if step >= 5:
                    profile_rows.append({k: float(v) for k, v in tm_all.items()})
            if profile_controller and (step < 5 or (step + 1) % 50 == 0):
                tm = info.get("timing_ms", {})
                total_ms = float(tm.get("total", math.nan))
                deadline_ms = 1000.0 * cfg.control_dt
                status = "OK" if total_ms <= deadline_ms else "MISS"
                nominal_detail = []
                if tm.get("policy", 0.0) > 0.005:
                    nominal_detail.append(f"policy {tm['policy']:.2f}")
                if tm.get("warm_start", 0.0) > 0.005:
                    nominal_detail.append(f"warm {tm['warm_start']:.2f}")
                if tm.get("prior", 0.0) > 0.005:
                    nominal_detail.append(f"prior {tm['prior']:.2f}")
                nominal_suffix = f" ({', '.join(nominal_detail)})" if nominal_detail else ""

                if tm.get("rollout_fused", 0.0) > 0.005:
                    rollout_detail = [f"fused {tm['rollout_fused']:.2f}"]
                else:
                    rollout_detail = [f"physics {tm.get('rollout_physics', 0.0):.2f}"]
                    if tm.get("rollout_cost", 0.0) > 0.005:
                        rollout_detail.append(f"cost {tm['rollout_cost']:.2f}")

                print(
                    f"MPPI [{step + 1:5d}]  "
                    f"nominal {tm.get('nominal', math.nan):7.2f} ms{nominal_suffix}  |  "
                    f"sample {tm.get('sampling', 0.0):6.2f} ms  |  "
                    f"rollout {tm.get('rollouts', 0.0):7.2f} ms ({', '.join(rollout_detail)})  |  "
                    f"update {tm.get('update', 0.0):6.2f} ms  |  "
                    f"total {total_ms:7.2f} / {deadline_ms:.2f} ms  [{status}]"
                )
            plant.step_control(ctrl, substeps=plant_control_substeps, data=plant.data)
            after = plant.snapshot()

            p = plant.xy()
            task_p = plant.task_xy()
            new_s, d2 = track.project(task_p)
            _, robot_d2 = track.project(p)
            ds = track.signed_progress_delta(float(new_s), current_s)
            cumulative += ds
            current_s = float(new_s)

            xy_hist.append(p)
            target_xy_hist.append(task_p)
            qpos_hist.append(after.qpos.copy())
            qvel_hist.append(after.qvel.copy())
            act_hist.append(after.act.copy())
            state_time_hist.append(float(after.time))
            controls.append(np.asarray(ctrl, dtype=np.float64).copy())
            progress_hist.append(cumulative)
            temperatures.append(float(info.get("temperature", math.nan)))
            esses.append(float(info.get("ess", math.nan)))

            new_completed = int(math.floor(max(cumulative, 0.0) / track.length + 1e-12))
            while completed < min(new_completed, int(laps)):
                completed += 1
                lap_times.append((step + 1) * cfg.control_dt)
                if verbose:
                    print(
                        f"lap {completed}/{laps}  sim_t={lap_times[-1]:.2f}s  "
                        f"lambda={temperatures[-1]:.4g}  ESS={esses[-1]:.1f}"
                    )

            allowed = max(0.0, 0.5 * track.road_width - cfg.hard_collision_clearance)
            off_track = float(d2) > allowed * allowed or float(robot_d2) > allowed * allowed
            # A low or tilted Ant is not necessarily fallen: terminate only
            # when the torso is actually inverted and in contact with ground.
            fell = plant.has_fallen(flipped_threshold=cfg.min_root_up)
            finished = cumulative >= target

            if off_track or fell or finished:
                if finished:
                    completed = int(laps)
                break

            if handle is not None:
                if controller_overlay:
                    draw_race_overlay(handle, track, prior=prior, controller_info=info)
                    if not safe_viewer_sync(handle, state_only=False):
                        handle = None
                elif not safe_viewer_sync(handle, state_only=True):
                    handle = None
    finally:
        if handle is not None:
            close_viewer(handle)
            handle = None
        controller.close()
        if disable_gc and gc_was_enabled:
            gc.enable()

    runtime = time.perf_counter() - t0
    if profile_controller and profile_rows:
        deadline_ms = 1000.0 * cfg.control_dt
        totals = np.asarray([r.get("total", math.inf) for r in profile_rows], dtype=np.float64)
        miss = 100.0 * float(np.mean(totals > deadline_ms))
        has_fused = max(r.get("rollout_fused", 0.0) for r in profile_rows) > 0.005
        metrics = [
            ("nominal", "nominal"),
            ("warm", "warm_start"),
            ("prior", "prior"),
            ("sample", "sampling"),
            ("rollout", "rollouts"),
        ]
        if has_fused:
            metrics.append(("fused", "rollout_fused"))
        else:
            metrics.extend((("physics", "rollout_physics"), ("cost", "rollout_cost")))
        metrics.extend((("update", "update"), ("total", "total")))
        print(f"MPPI profile  warm-up excluded  n={len(profile_rows)}")
        print("                 p50       p95")
        for label, key in metrics:
            vals = np.asarray([r.get(key, 0.0) for r in profile_rows], dtype=np.float64)
            if key not in {"total", "rollouts", "rollout_physics"} and np.max(np.abs(vals)) < 0.005:
                continue
            print(f"  {label:<10} {np.median(vals):8.2f}  {np.percentile(vals, 95):8.2f} ms")
        print(f"  deadline   {deadline_ms:8.2f} ms    misses {miss:5.1f}%")

    return RaceResult(
        robot_name=plant.name,
        controller_variant=controller.variant.value,
        control_dt=float(cfg.control_dt),
        plant_integrator=plant_integrator,
        planner_mode=planner_mode,
        xy=np.asarray(xy_hist, dtype=np.float64),
        target_xy=np.asarray(target_xy_hist, dtype=np.float64),
        qpos=np.asarray(qpos_hist, dtype=np.float64),
        qvel=np.asarray(qvel_hist, dtype=np.float64),
        act=np.asarray(act_hist, dtype=np.float64),
        state_time=np.asarray(state_time_hist, dtype=np.float64),
        controls=np.asarray(controls, dtype=np.float64),
        cumulative_progress=np.asarray(progress_hist, dtype=np.float64),
        temperatures=np.asarray(temperatures, dtype=np.float64),
        esses=np.asarray(esses, dtype=np.float64),
        lap_times=lap_times,
        completed_laps=completed,
        requested_laps=int(laps),
        off_track=off_track,
        fell=fell,
        runtime_s=runtime,
        simulated_time_s=len(controls) * cfg.control_dt,
        track=track,
        environment=environment,
        plant_parameters=plant_params,
    )


def save_result(result: RaceResult, path: str | Path) -> Path:
    """Save metrics plus the full MuJoCo state trajectory for exact visual replay."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        replay_format_version=np.asarray(3, dtype=np.int64),
        robot_name=result.robot_name,
        controller_variant=result.controller_variant,
        control_dt=result.control_dt,
        plant_integrator=np.asarray(result.plant_integrator),
        planner_mode=np.asarray(result.planner_mode),
        xy=result.xy,
        target_xy=result.target_xy,
        environment_json=np.asarray(result.environment.to_json()),
        qpos=result.qpos,
        qvel=result.qvel,
        act=result.act,
        state_time=result.state_time,
        controls=result.controls,
        cumulative_progress=result.cumulative_progress,
        temperatures=result.temperatures,
        esses=result.esses,
        lap_times=np.asarray(result.lap_times),
        completed_laps=result.completed_laps,
        requested_laps=result.requested_laps,
        off_track=result.off_track,
        fell=result.fell,
        runtime_s=result.runtime_s,
        simulated_time_s=result.simulated_time_s,
        track=np.asarray([
            result.track.width,
            result.track.height,
            result.track.road_width,
            result.track.origin_xy[0],
            result.track.origin_xy[1],
            result.track.origin_yaw,
        ], dtype=np.float64),
        plant_parameters=np.asarray([
            result.plant_parameters.friction,
            result.plant_parameters.mass,
            result.plant_parameters.motor,
            result.plant_parameters.slope_deg,
        ]),
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-seeded direct-joint MuJoCo racing")
    parser.add_argument("--robot", default="ant", choices=["ant"], help="Ant stadium racing")
    parser.add_argument(
        "--policy",
        default="auto",
        help="auto uses racing/policies/checkpoints/<robot>_rapid; also accepts a checkpoint directory, neutral, or module:function",
    )
    parser.add_argument("--policy-speed", type=float, default=None, help="optional cap on the learned maximum racing speed; omitted uses the curriculum envelope")
    parser.add_argument("--prior", default=None, help="Empirical prior .npz; geometric when omitted")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="MPPI/policy control dt; defaults to trained policy dt when available",
    )
    parser.add_argument("--lbps-delta", type=float, default=0.95)
    parser.add_argument("--nominal-refine-iters", type=int, default=0)
    parser.add_argument("--joint-noise", type=float, default=0.5, help="actuator-range exploration-noise scale for MPPI")
    parser.add_argument(
        "--variant",
        choices=[v.value for v in ControllerVariant],
        default=ControllerVariant.MPPI.value,
        help="nominal executes the pretrained policy directly; mppi runs standard policy-seeded MPPI",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--workers", type=int, default=16,
        help="native rollout threads (default: 16; use 0 for automatic)",
    )
    parser.add_argument(
        "--rollout-chunk-size", type=int, default=0,
        help="native rollout thread-pool chunk size; 0=automatic. For 64 rollouts/16 workers, benchmark 2 and 4",
    )
    parser.add_argument(
        "--warm-start", action=argparse.BooleanOptionalAction, default=True,
        help="shift the optimized sequence between updates (default: enabled; use --no-warm-start for the original behavior)",
    )
    parser.add_argument(
        "--plant-integrator", choices=["model", "euler", "implicitfast"], default="model",
        help="integrator for the rendered/physical plant; model preserves the source XML setting",
    )
    parser.add_argument(
        "--planner-mode", choices=["rk4", "fast-rk4", "implicitfast"], default="rk4",
        help=(
            "planner physics profile: rk4 uses RK4 with the source timestep/solver; "
            "fast-rk4 uses one RK4 step per control interval plus the fast solver/contact profile; "
            "implicitfast uses implicitfast at the source timestep with that same fast profile"
        ),
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="print MPPI timing breakdown for the first steps and every 50 updates",
    )
    parser.add_argument(
        "--disable-gc", action="store_true",
        help="disable Python cyclic GC during the race loop to reduce real-time jitter",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="disable the MuJoCo viewer")
    parser.add_argument("--viewer-ui", action="store_true", help="show MuJoCo left/right UI panels")
    parser.add_argument("--controller-overlay", action="store_true")

    # Physical environment perturbations. The planning model does not receive
    # these values directly.
    parser.add_argument("--friction-scale", type=float, default=1.0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--motor-scale", type=float, default=1.0)
    parser.add_argument("--slope-deg", type=float, default=0.0)

    parser.add_argument(
        "--task", choices=["run", "push_box", "tow_sled"], default="run",
        help="run tracks robot progress; push_box tracks the pushed object; tow_sled tracks a cable-towed sled; all reuse the same pretrained running policy",
    )
    parser.add_argument(
        "--terrain", choices=["flat", "ramps", "stairs", "rocky", "mixed"], default="flat",
        help="known test-time terrain on the upper straight and second turn; PPO stays flat-ground pretrained; push_box/tow_sled require flat",
    )
    parser.add_argument("--terrain-seed", type=int, default=1, help="deterministic rocky/mixed terrain seed")
    parser.add_argument("--terrain-scale", type=float, default=1.0, help="scale obstacle heights/ramp rise")
    parser.add_argument(
        "--leg-mismatch", choices=["none", "same_side", "diagonal"], default="none",
        help="known Ant leg-length transfer for simple flat racing; plant/planner use modified geometry while PPO remains nominal-pretrained",
    )
    parser.add_argument(
        "--short-leg-scale", type=float, default=0.75,
        help="length scale for the two short Ant legs when --leg-mismatch is enabled (default: 0.75)",
    )
    parser.add_argument(
        "--long-leg-scale", type=float, default=1.25,
        help="length scale for the two long Ant legs when --leg-mismatch is enabled (default: 1.25)",
    )
    parser.add_argument("--box-distance", type=float, default=1.8, help="initial box center distance ahead of the robot along track [m] (default: 1.8)")
    parser.add_argument("--box-size", type=float, default=0.90, help="box footprint edge [m] (default: 0.90)")
    parser.add_argument("--box-height", type=float, default=0.45, help="box height [m] (default: 0.45; low crate reduces kicking/tipping)")
    parser.add_argument("--box-mass", type=float, default=0.25, help="box mass [kg] (default: 6.0)")
    parser.add_argument("--box-friction", type=float, default=0.60, help="box-ground sliding friction coefficient (default: 0.60)")
    parser.add_argument("--sled-distance", type=float, default=1.8, help="initial sled center distance behind the robot along track [m] (default: 1.8)")
    parser.add_argument("--sled-length", type=float, default=1.0, help="sled length along the track [m] (default: 1.0)")
    parser.add_argument("--sled-width", type=float, default=0.80, help="sled width [m] (default: 0.80)")
    parser.add_argument("--sled-height", type=float, default=0.16, help="sled body height [m] (default: 0.16)")
    parser.add_argument("--sled-mass", type=float, default=0.25, help="sled mass [kg] (default: 8.0)")
    parser.add_argument("--sled-friction", type=float, default=0.60, help="sled-ground sliding friction coefficient (default: 0.60)")
    parser.add_argument("--sled-rope-length", type=float, default=3.0, help="maximum tow-cable length [m] (default: 1.25)")
    parser.add_argument("--push-box-progress-weight", type=float, default=1.0, help="primary box track-progress reward weight")
    parser.add_argument("--push-robot-progress-weight", type=float, default=0.35, help="coupled robot-progress shaping weight; robot cannot earn it by running past a stationary box")
    parser.add_argument("--push-approach-weight", type=float, default=1.00, help="dense reward for reducing/maintaining robot-box distance")
    parser.add_argument("--push-box-max-lift", type=float, default=0.12, help="reject MPPI candidates lifting the box more than this above reset height [m]")
    parser.add_argument("--push-box-min-up", type=float, default=0.75, help="reject MPPI candidates tipping the box below this world-up cosine")
    parser.add_argument("--sled-progress-weight", type=float, default=1.0, help="primary towed-sled track-progress reward weight")
    parser.add_argument("--sled-robot-progress-weight", type=float, default=10.0, help="robot-progress shaping while towing; capped by sled progress")
    parser.add_argument("--sled-max-lift", type=float, default=0.12, help="reject MPPI candidates lifting the sled more than this above reset height [m]")
    parser.add_argument("--sled-min-up", type=float, default=0.70, help="reject MPPI candidates tipping the sled below this world-up cosine")

    parser.add_argument(
        "--save",
        default="racing/results/last_run.npz",
        help="record metrics and full MuJoCo states for replay (default: racing/results/last_run.npz)",
    )
    parser.add_argument("--no-save", action="store_true", help="do not save a replay file")
    args = parser.parse_args()

    prior = EmpiricalPrior.load(args.prior) if args.prior else GeometricPrior()
    result = run_race(
        robot_name=args.robot,
        laps=args.laps,
        prior=prior,
        policy_spec=args.policy,
        policy_speed=args.policy_speed,
        variant=args.variant,
        num_rollouts=args.rollouts,
        horizon=args.horizon,
        control_dt=args.dt,
        lbps_delta=args.lbps_delta,
        nominal_refine_iterations=args.nominal_refine_iters,
        joint_noise_fraction=args.joint_noise,
        seed=args.seed,
        max_steps=args.max_steps,
        viewer=not args.headless,
        viewer_ui=args.viewer_ui,
        controller_overlay=args.controller_overlay,
        rollout_workers=args.workers,
        rollout_chunk_size=args.rollout_chunk_size,
        warm_start=args.warm_start,
        plant_integrator=args.plant_integrator,
        planner_mode=args.planner_mode,
        profile_controller=args.profile,
        disable_gc=args.disable_gc,
        friction_scale=args.friction_scale,
        mass_scale=args.mass_scale,
        motor_scale=args.motor_scale,
        slope_deg=args.slope_deg,
        task=args.task,
        terrain=args.terrain,
        terrain_seed=args.terrain_seed,
        terrain_scale=args.terrain_scale,
        leg_mismatch=args.leg_mismatch,
        short_leg_scale=args.short_leg_scale,
        long_leg_scale=args.long_leg_scale,
        box_distance=args.box_distance,
        box_size=args.box_size,
        box_height=args.box_height,
        box_mass=args.box_mass,
        box_friction=args.box_friction,
        sled_distance=args.sled_distance,
        sled_length=args.sled_length,
        sled_width=args.sled_width,
        sled_height=args.sled_height,
        sled_mass=args.sled_mass,
        sled_friction=args.sled_friction,
        sled_rope_length=args.sled_rope_length,
        push_box_progress_weight=args.push_box_progress_weight,
        push_robot_progress_weight=args.push_robot_progress_weight,
        push_approach_weight=args.push_approach_weight,
        push_box_max_lift=args.push_box_max_lift,
        push_box_min_up=args.push_box_min_up,
        sled_progress_weight=args.sled_progress_weight,
        sled_robot_progress_weight=args.sled_robot_progress_weight,
        sled_max_lift=args.sled_max_lift,
        sled_min_up=args.sled_min_up,
    )
    print(
        f"finished {result.robot_name} task={result.environment.task}: "
        f"{result.completed_laps}/{result.requested_laps} laps, "
        f"off_track={result.off_track}, fell={result.fell}, "
        f"progress={result.cumulative_progress[-1]:.2f}m, "
        f"sim={result.simulated_time_s:.2f}s, compute={result.runtime_s:.2f}s"
    )
    if not args.no_save and args.save:
        saved = save_result(result, args.save)
        print(f"saved replay: {saved}")
        print(f"replay with: python -m racing.experiments.replay --file {saved}")


if __name__ == "__main__":
    main()
