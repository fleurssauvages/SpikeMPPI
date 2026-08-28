from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Optional
import numpy as np

from racing.adaptation import ModelParameterScales, OnlineSystemIdentifier, SystemIDConfig
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
    model_estimates: np.ndarray
    model_estimate_steps: np.ndarray


def run_race(
    *,
    robot_name: str,
    laps: int = 1,
    prior: Optional[SpatialPrior] = None,
    policy_spec: str | None = None,
    policy_speed: float | None = None,
    variant: ControllerVariant | str = ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI,
    num_rollouts: int = 128,
    horizon: int = 15,
    control_dt: float | None = None,
    lbps_delta: float = 0.9,
    nominal_refine_iterations: int = 0,
    spg_lookahead_steps: int = 3,
    spg_mix: float = 1.0,
    spg_null_std_scale: float = 0.15,
    spg_pseudoinverse_damping: float = 1e-6,
    sensitivity_epsilon_fraction: float = 1e-3,
    joint_noise_fraction: float = 0.08,
    seed: int = 1,
    max_steps: int | None = None,
    viewer: bool = True,
    viewer_ui: bool = False,
    controller_overlay: bool = False,
    rollout_workers: int = 0,
    rollout_backend: str = "native",
    rollout_chunk_size: int = 0,
    warm_start: bool = False,
    spg_jacobian_refresh_interval: int = 1,
    spg_jacobian_refresh_prefix: int = 0,
    planner_integrator: str = "model",
    profile_controller: bool = False,
    disable_gc: bool = False,
    verbose: bool = True,
    friction_scale: float = 1.0,
    mass_scale: float = 1.0,
    motor_scale: float = 1.0,
    slope_deg: float = 0.0,
    task: str = "run",
    push_object: str = "box",
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
    ball_rolling_friction: float = 0.03,
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
    online_adaptation: bool = False,
    sysid_history: int = 12,
    sysid_interval: int = 8,
    sysid_estimate_slope: bool = False,
) -> RaceResult:
    """Race a classic MuJoCo robot using a policy-seeded joint-space controller.

    SPG is the default proposal.  At each receding-horizon update the locomotion
    policy generates a feasible joint-control nominal. MuJoCo finite differences
    estimate J_t = d p_xy(t+L) / d u_t around that nominal, and the 2-D spatial
    prior covariance is projected into the full actuator space before MPPI/LBPS.

    ``plant`` is the rendered/physical environment and may be perturbed. Known
    test-time task/terrain/morphology changes are also compiled into ``planner``;
    optional friction/mass/motor/slope perturbations remain plant-only unless online
    adaptation estimates them from recent transitions.
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
            "Use ant, humanoid, or swimmer for the 2-D race."
        )
    origin_xy = tuple(map(float, probe.xy()))
    origin_yaw = float(probe.root_yaw())
    track = StadiumTrack(origin_xy=origin_xy, origin_yaw=origin_yaw)
    environment = RaceEnvironmentConfig(
        task=task, push_object=push_object, terrain=terrain, terrain_seed=terrain_seed, terrain_scale=terrain_scale,
        leg_mismatch=leg_mismatch, short_leg_scale=short_leg_scale, long_leg_scale=long_leg_scale,
        box_distance=box_distance, box_size=box_size, box_height=box_height,
        box_mass=box_mass, box_friction=box_friction, ball_rolling_friction=ball_rolling_friction,
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
    # free task-body mass. Classic Ant/Humanoid use inertiafromgeom=true, so the
    # environment builders impose mass through explicit geom density.
    compiled_task_mass_plant = None
    compiled_task_mass_planner = None
    if environment.task in {"push_box", "tow_sled"}:
        compiled_task_mass_plant = float(plant.model.body_mass[plant.task_body_id])
        compiled_task_mass_planner = float(planner.model.body_mass[planner.task_body_id])
        if environment.task == "push_box":
            requested_task_mass = float(environment.box_mass)
            mass_flag = "--box-mass"
            object_label = environment.push_object
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

    # Keep the physical plant on the XML integrator, but optionally use a
    # lower-latency integrator in the planning copy.
    planner_integrator = str(planner_integrator).strip().lower()
    if planner_integrator not in {"model", "euler", "implicitfast"}:
        raise ValueError("planner_integrator must be model, euler, or implicitfast")
    if planner_integrator != "model":
        integrator_map = {
            "euler": planner.mujoco.mjtIntegrator.mjINT_EULER,
            "implicitfast": planner.mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }
        planner.model.opt.integrator = integrator_map[planner_integrator]
        planner.mujoco.mj_forward(planner.model, planner.data)
    prior = prior or GeometricPrior()
    policy = make_policy(policy_spec, race_speed=policy_speed, robot_name=robot_name)
    policy.reset(planner, planner.data)

    if control_dt is None:
        control_dt = float(getattr(policy, "control_dt", 0.02))

    # Reuse the same fast rollout/fused cost ABI for pushing and towing. The task
    # body is the box/ball or sled respectively. Towing does not need an
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
        task_min_up = -1.0 if environment.push_object == "ball" else float(push_box_min_up)

    cfg = ControllerConfig(
        control_dt=float(control_dt),
        horizon=int(horizon),
        num_rollouts=int(num_rollouts),
        lbps_delta=float(lbps_delta),
        nominal_refine_iterations=int(nominal_refine_iterations),
        spg_lookahead_steps=int(spg_lookahead_steps),
        spg_mix=float(spg_mix),
        spg_null_std_scale=float(spg_null_std_scale),
        spg_pseudoinverse_damping=float(spg_pseudoinverse_damping),
        sensitivity_epsilon_fraction=float(sensitivity_epsilon_fraction),
        joint_noise_fraction=float(joint_noise_fraction),
        rollout_workers=int(rollout_workers),
        rollout_backend=str(rollout_backend),
        rollout_chunk_size=int(rollout_chunk_size),
        warm_start=bool(warm_start),
        spg_jacobian_refresh_interval=int(spg_jacobian_refresh_interval),
        spg_jacobian_refresh_prefix=int(spg_jacobian_refresh_prefix),
        box_progress_weight=task_progress_weight,
        robot_progress_weight=task_robot_progress_weight,
        robot_box_approach_weight=task_approach_weight,
        box_max_lift=task_max_lift,
        box_min_up=task_min_up,
    )
    controller = JointMPPIController(planner, track, prior, policy, cfg, variant=variant, seed=seed)

    identifier = None
    if online_adaptation:
        identifier = OnlineSystemIdentifier(
            planner,
            SystemIDConfig(
                history=int(sysid_history),
                update_interval=int(sysid_interval),
                estimate_slope=bool(sysid_estimate_slope),
            ),
        )

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
    estimate_rows: list[list[float]] = [[1.0, 1.0, 1.0, 0.0]]
    estimate_steps: list[int] = [0]
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
            f"backend={controller.rollout_backend_name}  planner_integrator={planner_integrator} "
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
            shape_desc = (
                f"diameter={environment.box_size:g}m"
                if environment.push_object == "ball"
                else f"footprint={environment.box_size:g}m height={environment.box_height:g}m"
            )
            print(
                "push reward: "
                f"box_progress={cfg.box_progress_weight:g}, "
                f"robot_progress={cfg.robot_progress_weight:g}, "
                f"approach={cfg.robot_box_approach_weight:g}; "
                f"object={environment.push_object} box_distance={environment.box_distance:g}m "
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
        if controller.variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            print(
                f"SPG: lookahead={cfg.spg_lookahead_steps}, mix={cfg.spg_mix:g}, "
                f"null_std={cfg.spg_null_std_scale:g}, damping={cfg.spg_pseudoinverse_damping:g}, "
                f"jac_refresh={cfg.spg_jacobian_refresh_interval}, prefix={cfg.spg_jacobian_refresh_prefix}"
            )

    gc_was_enabled = gc.isenabled()
    if disable_gc and gc_was_enabled:
        gc.disable()

    t0 = time.perf_counter()
    try:
        for step in range(max_steps):
            if handle is not None and not handle.is_running():
                break

            before = plant.snapshot()
            # The controller reads the physical qpos/qvel through the snapshot,
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
                rtf = deadline_ms / total_ms if total_ms > 0.0 else math.nan
                print(
                    "MPPI timing "
                    f"step={step + 1} nominal={tm.get('nominal', math.nan):.2f}ms "
                    f"(policy={tm.get('policy', 0.0):.2f} warm={tm.get('warm_start', 0.0):.2f} "
                    f"spg_jac={tm.get('sensitivity', 0.0):.2f} prior={tm.get('prior', 0.0):.2f}) "
                    f"sample={tm.get('sampling', 0.0):.2f}ms "
                    f"rollouts={tm.get('rollouts', 0.0):.2f}ms "
                    f"(physics={tm.get('rollout_physics', 0.0):.2f} cost={tm.get('rollout_cost', 0.0):.2f}"
                    + (f" fused={tm.get('rollout_fused', 0.0):.2f}" if tm.get('rollout_fused', 0.0) > 0.0 else "")
                    + ") "
                    + f"update={tm.get('update', 0.0):.2f}ms total={total_ms:.2f}ms "
                    + f"deadline={deadline_ms:.2f}ms xRT={rtf:.2f} "
                    + f"J={info.get('spg_refresh_mode', 'full')}"
                )
            plant.step_control(ctrl, substeps=controller.control_substeps, data=plant.data)
            after = plant.snapshot()

            if identifier is not None:
                identifier.observe(before, ctrl, after, substeps=controller.control_substeps)
                if identifier.should_update(step):
                    est = identifier.update()
                    controller.sync_planning_model()
                    estimate_rows.append([est.friction, est.mass, est.motor, est.slope_deg])
                    estimate_steps.append(step + 1)
                    if verbose:
                        print(
                            "sysid "
                            f"step={step+1} friction={est.friction:.3f} mass={est.mass:.3f} "
                            f"motor={est.motor:.3f} slope={est.slope_deg:.2f}deg "
                            f"loss={identifier.last_loss:.3g}"
                        )

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
            fell = (
                plant.root_height() < cfg.fall_height_fraction * max(plant.initial_root_height, 1e-6)
                or plant.root_up() < cfg.min_root_up
            )
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
        keys = ("policy", "warm_start", "sensitivity", "prior", "sampling", "rollouts", "rollout_physics", "rollout_cost", "rollout_fused", "update", "total")
        deadline_ms = 1000.0 * cfg.control_dt
        summary = []
        for key in keys:
            vals = np.asarray([r.get(key, 0.0) for r in profile_rows], dtype=np.float64)
            summary.append(f"{key}=p50 {np.median(vals):.2f}/p95 {np.percentile(vals, 95):.2f}ms")
        totals = np.asarray([r.get("total", math.inf) for r in profile_rows], dtype=np.float64)
        miss = 100.0 * float(np.mean(totals > deadline_ms))
        print("MPPI profile (warm-up excluded): " + ", ".join(summary))
        print(f"deadline={deadline_ms:.2f}ms  misses={miss:.1f}%  samples={len(profile_rows)}")

    return RaceResult(
        robot_name=plant.name,
        controller_variant=controller.variant.value,
        control_dt=float(cfg.control_dt),
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
        model_estimates=np.asarray(estimate_rows, dtype=np.float64),
        model_estimate_steps=np.asarray(estimate_steps, dtype=np.int64),
    )


def save_result(result: RaceResult, path: str | Path) -> Path:
    """Save metrics plus the full MuJoCo state trajectory for exact visual replay."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        replay_format_version=np.asarray(2, dtype=np.int64),
        robot_name=result.robot_name,
        controller_variant=result.controller_variant,
        control_dt=result.control_dt,
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
        model_estimates=result.model_estimates,
        model_estimate_steps=result.model_estimate_steps,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-seeded direct-joint MuJoCo SPG-MPPI racing")
    parser.add_argument("--robot", default="ant", help="ant or humanoid for stadium racing")
    parser.add_argument(
        "--policy",
        default="auto",
        help="auto uses racing/policies/checkpoints/<robot>_rapid; also accepts a checkpoint directory, neutral, or module:function",
    )
    parser.add_argument("--policy-speed", type=float, default=None, help="optional cap on the learned maximum racing speed; omitted uses the curriculum envelope")
    parser.add_argument("--prior", default=None, help="Empirical prior .npz; geometric when omitted")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="MPPI/policy control dt; defaults to trained policy dt when available",
    )
    parser.add_argument("--lbps-delta", type=float, default=0.9)
    parser.add_argument("--nominal-refine-iters", type=int, default=0)
    parser.add_argument("--spg-lookahead", type=int, default=3)
    parser.add_argument("--spg-mix", type=float, default=1.0, help="1.0 = pure SPG task/null-space proposal; lower values blend standard joint noise")
    parser.add_argument("--spg-null-std", type=float, default=0.15, help="uninformed exploration scale restricted to the Jacobian null space")
    parser.add_argument("--spg-damping", type=float, default=1e-6, help="damping used in J^dagger = J^T (J J^T + lambda I)^-1")
    parser.add_argument("--spg-epsilon", type=float, default=1e-3, help="finite-difference fraction of actuator range for SPG sensitivities")
    parser.add_argument("--joint-noise", type=float, default=0.08, help="actuator-range noise scale; for SPG this sets null-space/default exploration")
    parser.add_argument(
        "--variant",
        choices=[v.value for v in ControllerVariant],
        default=ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI.value,
        help="SPG is the default; standard_mppi is retained only as an ablation",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="native rollout threads; 0=auto (all logical CPU cores)",
    )
    parser.add_argument(
        "--rollout-chunk-size", type=int, default=0,
        help="native rollout thread-pool chunk size; 0=automatic. For 64 rollouts/16 workers, benchmark 2 and 4",
    )
    parser.add_argument(
        "--rollout-backend", choices=["fused", "native", "python"], default="native",
        help="fused uses the custom exact C++ rollout+cost evaluator; native uses mujoco.rollout; python keeps the legacy evaluator",
    )
    parser.add_argument(
        "--warm-start", action="store_true",
        help="shift the previous optimized MPPI sequence instead of rebuilding H policy actions every control tick",
    )
    parser.add_argument(
        "--spg-refresh", type=int, default=1,
        help="full SPG Jacobian refresh interval: 1=every tick (original), 0=initial only, N>1=every N ticks",
    )
    parser.add_argument(
        "--spg-refresh-prefix", type=int, default=0,
        help="when reusing a shifted SPG Jacobian, freshly finite-difference this many leading horizon steps",
    )
    parser.add_argument(
        "--planner-integrator", choices=["model", "euler", "implicitfast"], default="model",
        help="integrator for the planning copy only; the physical plant remains on the XML integrator",
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
        "--push-object", choices=["box", "ball"], default="box",
        help="object used by --task push_box; box uses --box-size as footprint edge, ball uses it as diameter",
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
    parser.add_argument("--box-size", type=float, default=0.90, help="box footprint edge or ball diameter [m] (default: 0.90)")
    parser.add_argument("--box-height", type=float, default=0.45, help="box height [m] (default: 0.45; low crate reduces kicking/tipping)")
    parser.add_argument("--box-mass", type=float, default=6.0, help="box mass [kg] (default: 6.0)")
    parser.add_argument("--box-friction", type=float, default=0.60, help="pushed-object sliding friction coefficient (default: 0.60)")
    parser.add_argument("--ball-rolling-friction", type=float, default=0.03, help="MuJoCo rolling-friction coefficient for --push-object ball (default: 0.03 m; requires condim=6)")
    parser.add_argument("--sled-distance", type=float, default=1.8, help="initial sled center distance behind the robot along track [m] (default: 1.8)")
    parser.add_argument("--sled-length", type=float, default=1.0, help="sled length along the track [m] (default: 1.0)")
    parser.add_argument("--sled-width", type=float, default=0.80, help="sled width [m] (default: 0.80)")
    parser.add_argument("--sled-height", type=float, default=0.16, help="sled body height [m] (default: 0.16)")
    parser.add_argument("--sled-mass", type=float, default=8.0, help="sled mass [kg] (default: 8.0)")
    parser.add_argument("--sled-friction", type=float, default=0.60, help="sled-ground sliding friction coefficient (default: 0.60)")
    parser.add_argument("--sled-rope-length", type=float, default=1.25, help="maximum tow-cable length [m] (default: 1.25)")
    parser.add_argument("--push-box-progress-weight", type=float, default=1.0, help="primary box track-progress reward weight")
    parser.add_argument("--push-robot-progress-weight", type=float, default=0.35, help="coupled robot-progress shaping weight; robot cannot earn it by running past a stationary box")
    parser.add_argument("--push-approach-weight", type=float, default=1.00, help="dense reward for reducing/maintaining robot-box distance")
    parser.add_argument("--push-box-max-lift", type=float, default=0.12, help="reject MPPI candidates lifting the box more than this above reset height [m]")
    parser.add_argument("--push-box-min-up", type=float, default=0.75, help="reject MPPI candidates tipping the box below this world-up cosine")
    parser.add_argument("--sled-progress-weight", type=float, default=1.0, help="primary towed-sled track-progress reward weight")
    parser.add_argument("--sled-robot-progress-weight", type=float, default=0.25, help="robot-progress shaping while towing; capped by sled progress")
    parser.add_argument("--sled-max-lift", type=float, default=0.12, help="reject MPPI candidates lifting the sled more than this above reset height [m]")
    parser.add_argument("--sled-min-up", type=float, default=0.70, help="reject MPPI candidates tipping the sled below this world-up cosine")

    parser.add_argument("--adapt-model", action="store_true", help="online system-identification of the SPG-MPPI planning model")
    parser.add_argument("--sysid-history", type=int, default=12)
    parser.add_argument("--sysid-interval", type=int, default=8)
    parser.add_argument("--sysid-estimate-slope", action="store_true")
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
        spg_lookahead_steps=args.spg_lookahead,
        spg_mix=args.spg_mix,
        spg_null_std_scale=args.spg_null_std,
        spg_pseudoinverse_damping=args.spg_damping,
        sensitivity_epsilon_fraction=args.spg_epsilon,
        joint_noise_fraction=args.joint_noise,
        seed=args.seed,
        max_steps=args.max_steps,
        viewer=not args.headless,
        viewer_ui=args.viewer_ui,
        controller_overlay=args.controller_overlay,
        rollout_workers=args.workers,
        rollout_backend=args.rollout_backend,
        rollout_chunk_size=args.rollout_chunk_size,
        warm_start=args.warm_start,
        spg_jacobian_refresh_interval=args.spg_refresh,
        spg_jacobian_refresh_prefix=args.spg_refresh_prefix,
        planner_integrator=args.planner_integrator,
        profile_controller=args.profile,
        disable_gc=args.disable_gc,
        friction_scale=args.friction_scale,
        mass_scale=args.mass_scale,
        motor_scale=args.motor_scale,
        slope_deg=args.slope_deg,
        task=args.task,
        push_object=args.push_object,
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
        ball_rolling_friction=args.ball_rolling_friction,
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
        online_adaptation=args.adapt_model,
        sysid_history=args.sysid_history,
        sysid_interval=args.sysid_interval,
        sysid_estimate_slope=args.sysid_estimate_slope,
    )
    realtime_factor = (
        result.simulated_time_s / result.runtime_s if result.runtime_s > 0.0 else math.inf
    )
    print(
        f"finished {result.robot_name} task={result.environment.task}: {result.completed_laps}/{result.requested_laps} laps, "
        f"off_track={result.off_track}, fell={result.fell}, "
        f"progress={result.cumulative_progress[-1]:.2f}m, "
        f"sim={result.simulated_time_s:.2f}s, compute={result.runtime_s:.2f}s, "
        f"xRT={realtime_factor:.2f}"
    )
    if len(result.model_estimates) > 1:
        f, m, a, slope = result.model_estimates[-1]
        print(f"final model estimate: friction={f:.3f}, mass={m:.3f}, motor={a:.3f}, slope={slope:.2f}deg")
    if not args.no_save and args.save:
        saved = save_result(result, args.save)
        print(f"saved replay: {saved}")
        print(f"replay with: python -m racing.experiments.replay --file {saved}")


if __name__ == "__main__":
    main()
