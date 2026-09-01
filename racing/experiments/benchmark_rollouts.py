"""Benchmark the MPPI physics/cost hot path without loading a policy.

Run from the directory containing ``racing/``::

    python -m racing.experiments.benchmark_rollouts --scene all --rollouts 32 --horizon 30

The first fused call performs the semantic cross-check and is excluded from the
measurements. Results therefore represent steady-state controller updates.
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from racing.environments import RaceEnvironmentConfig
from racing.mppi.rollout import NativeRolloutBatcher, RolloutCostConfig
from racing.robots import make_robot
from racing.tracks import StadiumTrack


SCENES = {
    "flat": {"task": "run", "terrain": "flat"},
    "rocky": {"task": "run", "terrain": "rocky"},
    "mixed": {"task": "run", "terrain": "mixed"},
    "push_box": {"task": "push_box", "terrain": "flat"},
    "tow_sled": {"task": "tow_sled", "terrain": "flat"},
}


def _make_planner(robot_name: str, scene: str, integrator: str, contact_mode: str):
    probe = make_robot(robot_name)
    track = StadiumTrack(
        origin_xy=tuple(map(float, probe.xy())),
        origin_yaw=float(probe.root_yaw()),
    )
    spec = SCENES[scene]
    env = RaceEnvironmentConfig(**spec).validated()
    planner = make_robot(
        robot_name,
        extra_worldbody_xml=env.planner_worldbody_xml(track),
    )
    planner.set_task_target_body(env.task_body_name)

    if integrator != "model":
        planner.model.opt.integrator = {
            "euler": planner.mujoco.mjtIntegrator.mjINT_EULER,
            "implicitfast": planner.mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }[integrator]
    if contact_mode == "fast":
        planner.model.opt.iterations = min(int(planner.model.opt.iterations), 20)
        planner.model.opt.ls_iterations = min(int(planner.model.opt.ls_iterations), 10)
        planner.model.opt.tolerance = max(float(planner.model.opt.tolerance), 1e-6)
        planner.model.opt.noslip_iterations = 0
    planner.mujoco.mj_forward(planner.model, planner.data)
    return planner, track


def _measure(
    robot_name: str,
    scene: str,
    backend: str,
    *,
    rollouts: int,
    horizon: int,
    control_dt: float,
    workers: int,
    chunk_size: int,
    repeats: int,
    integrator: str,
    contact_mode: str,
) -> tuple[float, float, float, str]:
    robot, track = _make_planner(robot_name, scene, integrator, contact_mode)
    batcher = NativeRolloutBatcher(
        robot,
        workers=workers,
        batch_hint=rollouts,
        chunk_size=chunk_size,
        fused=(backend == "fused"),
    )
    try:
        rng = np.random.default_rng(7)
        low, high = robot.control_bounds()
        nominal = np.broadcast_to(robot.default_ctrl, (horizon, robot.nu)).copy()
        span = np.maximum(high - low, 1e-6)
        controls = nominal[None, :, :] + 0.08 * span[None, None, :] * rng.standard_normal(
            (rollouts, horizon, robot.nu)
        )
        np.clip(controls, low, high, out=controls)
        controls[0] = nominal
        substeps = max(1, int(round(float(control_dt) / float(robot.physics_dt))))
        start = robot.snapshot()
        current_s = float(track.project(robot.task_xy())[0])
        cost = RolloutCostConfig()

        # Warm buffers/JIT and run the fused semantic verifier before timing.
        for _ in range(2):
            batcher.evaluate(
                start,
                controls,
                track,
                current_s,
                control_substeps=substeps,
                nominal_controls=nominal,
                cost_cfg=cost,
            )
        samples = []
        for _ in range(max(1, int(repeats))):
            t0 = time.perf_counter()
            batcher.evaluate(
                start,
                controls,
                track,
                current_s,
                control_substeps=substeps,
                nominal_controls=nominal,
                cost_cfg=cost,
            )
            samples.append(1e3 * (time.perf_counter() - t0))
        ordered = sorted(samples)
        p50 = float(statistics.median(ordered))
        p95 = float(ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)])
        return p50, p95, 1000.0 / max(p50, 1e-12), batcher.uses_fused and "fused" or "native"
    finally:
        batcher.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark obstacle-sensitive MPPI rollouts")
    parser.add_argument("--robot", default="ant")
    parser.add_argument("--scene", choices=["all", *SCENES], default="all")
    parser.add_argument("--backend", choices=["all", "native", "fused"], default="all")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--integrator", choices=["model", "euler", "implicitfast"], default="implicitfast")
    parser.add_argument("--contact-mode", choices=["model", "fast"], default="fast")
    args = parser.parse_args()

    scenes = list(SCENES) if args.scene == "all" else [args.scene]
    backends = ["native", "fused"] if args.backend == "all" else [args.backend]
    print("scene       backend  p50_ms  p95_ms  updates/s")
    for scene in scenes:
        for backend in backends:
            try:
                p50, p95, hz, actual = _measure(
                    args.robot,
                    scene,
                    backend,
                    rollouts=max(1, args.rollouts),
                    horizon=max(1, args.horizon),
                    control_dt=args.dt,
                    workers=args.workers,
                    chunk_size=args.chunk_size,
                    repeats=args.repeats,
                    integrator=args.integrator,
                    contact_mode=args.contact_mode,
                )
                print(f"{scene:<11} {actual:<7} {p50:7.2f} {p95:7.2f} {hz:10.1f}")
            except RuntimeError as exc:
                print(f"{scene:<11} {backend:<7} skipped: {exc}")


if __name__ == "__main__":
    main()
