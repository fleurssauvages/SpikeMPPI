from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Optional
import numpy as np

from racing.adaptation import ModelParameterScales, OnlineSystemIdentifier, SystemIDConfig
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
    verbose: bool = True,
    friction_scale: float = 1.0,
    mass_scale: float = 1.0,
    motor_scale: float = 1.0,
    slope_deg: float = 0.0,
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

    ``plant`` is the rendered/physical environment and may be perturbed. ``planner``
    starts from nominal model parameters. When online adaptation is enabled, recent
    plant transitions update the planning model without revealing the true scales.
    """
    plant = make_robot(robot_name)
    planner = make_robot(robot_name)
    if plant.nu <= 0:
        raise ValueError(f"{plant.name} has no MuJoCo actuators (model.nu=0)")
    if not plant.supports_stadium:
        raise ValueError(
            f"{plant.display_name} is constrained to 1-D forward locomotion and cannot turn on the stadium track. "
            "Use ant, humanoid, or swimmer for the 2-D race."
        )

    plant_params = ModelParameterScales(
        friction=float(friction_scale),
        mass=float(mass_scale),
        motor=float(motor_scale),
        slope_deg=float(slope_deg),
    ).clipped()
    plant.apply_model_parameters(plant_params)
    planner.apply_model_parameters(ModelParameterScales())

    origin_xy = tuple(map(float, plant.xy()))
    origin_yaw = float(plant.root_yaw())
    track = StadiumTrack(origin_xy=origin_xy, origin_yaw=origin_yaw)
    prior = prior or GeometricPrior()
    policy = make_policy(policy_spec, race_speed=policy_speed, robot_name=robot_name)
    policy.reset(planner, planner.data)

    if control_dt is None:
        control_dt = float(getattr(policy, "control_dt", 0.02))

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

    current_s, _ = track.project(plant.xy())
    current_s = float(current_s)
    cumulative = 0.0
    target = int(laps) * track.length
    max_steps = int(max_steps) if max_steps is not None else max(1000, 6000 * int(laps))

    initial_state = plant.snapshot()
    xy_hist = [plant.xy()]
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
            f"rollouts={cfg.num_rollouts}  H={cfg.horizon}  dt={cfg.control_dt:g}s"
        )
        if controller.variant == ControllerVariant.SENSITIVITY_PROJECTED_GAUSSIAN_MPPI:
            print(
                f"SPG: lookahead={cfg.spg_lookahead_steps}, mix={cfg.spg_mix:g}, "
                f"null_std={cfg.spg_null_std_scale:g}, damping={cfg.spg_pseudoinverse_damping:g}"
            )

    t0 = time.perf_counter()
    try:
        for step in range(max_steps):
            if handle is not None and not handle.is_running():
                break

            before = plant.snapshot()
            # The controller reads the physical qpos/qvel through the snapshot,
            # but all candidate rollouts use the separate planning MuJoCo model.
            ctrl, info = controller.step(plant.data, current_s)
            plant.step_control(ctrl, substeps=controller.control_substeps, data=plant.data)
            after = plant.snapshot()

            if identifier is not None:
                identifier.observe(before, ctrl, after, substeps=controller.control_substeps)
                if identifier.should_update(step):
                    est = identifier.update()
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
            new_s, d2 = track.project(p)
            ds = track.signed_progress_delta(float(new_s), current_s)
            cumulative += ds
            current_s = float(new_s)

            xy_hist.append(p)
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
            off_track = float(d2) > allowed * allowed
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

    runtime = time.perf_counter() - t0
    return RaceResult(
        robot_name=plant.name,
        controller_variant=controller.variant.value,
        control_dt=float(cfg.control_dt),
        xy=np.asarray(xy_hist, dtype=np.float64),
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
        replay_format_version=np.asarray(1, dtype=np.int64),
        robot_name=result.robot_name,
        controller_variant=result.controller_variant,
        control_dt=result.control_dt,
        xy=result.xy,
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
    parser.add_argument("--workers", type=int, default=0)
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
        friction_scale=args.friction_scale,
        mass_scale=args.mass_scale,
        motor_scale=args.motor_scale,
        slope_deg=args.slope_deg,
        online_adaptation=args.adapt_model,
        sysid_history=args.sysid_history,
        sysid_interval=args.sysid_interval,
        sysid_estimate_slope=args.sysid_estimate_slope,
    )
    print(
        f"finished {result.robot_name}: {result.completed_laps}/{result.requested_laps} laps, "
        f"off_track={result.off_track}, fell={result.fell}, "
        f"progress={result.cumulative_progress[-1]:.2f}m, "
        f"sim={result.simulated_time_s:.2f}s, compute={result.runtime_s:.2f}s"
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
