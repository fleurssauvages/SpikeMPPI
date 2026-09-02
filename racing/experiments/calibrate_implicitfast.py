from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from racing.adaptation import ModelParameterScales, OnlineSystemIdentifier, SystemIDConfig
from racing.policies import make_policy
from racing.priors import GeometricPrior
from racing.robots import RobotSnapshot, make_robot
from racing.tracks import StadiumTrack


def _set_integrator(robot, name: str) -> None:
    name = str(name).strip().lower()
    if name == "model":
        return
    table = {
        "euler": robot.mujoco.mjtIntegrator.mjINT_EULER,
        "implicitfast": robot.mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    if name not in table:
        raise ValueError(f"unknown integrator {name!r}")
    robot.model.opt.integrator = table[name]
    robot.mujoco.mj_forward(robot.model, robot.data)


def _snapshot_from_arrays(time_value, qpos, qvel, act, ctrl) -> RobotSnapshot:
    return RobotSnapshot(
        time=float(time_value),
        qpos=np.asarray(qpos, dtype=np.float64).copy(),
        qvel=np.asarray(qvel, dtype=np.float64).copy(),
        act=np.asarray(act, dtype=np.float64).copy(),
        ctrl=np.asarray(ctrl, dtype=np.float64).copy(),
    )


def _load_replay_transitions(path: Path, planner, max_transitions: int):
    with np.load(path, allow_pickle=False) as rec:
        required = {"qpos", "qvel", "act", "state_time", "controls"}
        missing = sorted(required.difference(rec.files))
        if missing:
            raise ValueError(f"replay is missing fields: {', '.join(missing)}")
        if "environment_json" in rec.files:
            try:
                env = json.loads(str(np.asarray(rec["environment_json"]).item()))
            except Exception:
                env = {}
            if env and (
                env.get("task", "run") != "run"
                or env.get("terrain", "flat") != "flat"
                or env.get("leg_mismatch", "none") != "none"
            ):
                raise ValueError(
                    "offline integrator calibration expects a nominal flat 'run' replay with no leg mismatch"
                )
        if "plant_parameters" in rec.files:
            pp = np.asarray(rec["plant_parameters"], dtype=np.float64).reshape(-1)
            if pp.size >= 4 and not np.allclose(pp[:4], [1.0, 1.0, 1.0, 0.0], rtol=0.0, atol=1e-10):
                raise ValueError(
                    "calibration replay contains physical perturbations; collect an unperturbed model-integrator run"
                )
        qpos = np.asarray(rec["qpos"], dtype=np.float64)
        qvel = np.asarray(rec["qvel"], dtype=np.float64)
        act = np.asarray(rec["act"], dtype=np.float64)
        times = np.asarray(rec["state_time"], dtype=np.float64).reshape(-1)
        ctrls = np.asarray(rec["controls"], dtype=np.float64)
    if qpos.shape[1] != int(planner.model.nq) or qvel.shape[1] != int(planner.model.nv):
        raise ValueError(
            f"replay state dimensions ({qpos.shape[1]}, {qvel.shape[1]}) do not match nominal Ant "
            f"({planner.model.nq}, {planner.model.nv})"
        )
    n = min(len(ctrls), len(times) - 1, len(qpos) - 1, len(qvel) - 1, len(act) - 1)
    if n <= 0:
        raise ValueError("replay contains no transitions")
    start = max(0, n - int(max_transitions))
    transitions = []
    for i in range(start, n):
        dt = max(float(times[i + 1] - times[i]), planner.physics_dt)
        substeps = max(1, int(round(dt / planner.physics_dt)))
        before_ctrl = ctrls[i - 1] if i > 0 else np.zeros(planner.nu, dtype=np.float64)
        before = _snapshot_from_arrays(times[i], qpos[i], qvel[i], act[i], before_ctrl)
        after = _snapshot_from_arrays(times[i + 1], qpos[i + 1], qvel[i + 1], act[i + 1], ctrls[i])
        transitions.append((before, ctrls[i], after, substeps))
    return transitions


def _collect_nominal_transitions(policy_spec: str, max_transitions: int, warmup_steps: int):
    plant = make_robot("ant")
    track = StadiumTrack(origin_xy=tuple(map(float, plant.xy())), origin_yaw=float(plant.root_yaw()))
    prior = GeometricPrior()
    policy = make_policy(policy_spec, robot_name="ant")
    policy.reset(plant, plant.data)
    control_dt = float(getattr(policy, "control_dt", 0.02))
    substeps = max(1, int(round(control_dt / plant.physics_dt)))
    current_s = float(track.project(plant.xy())[0])
    transitions = []
    total = int(warmup_steps) + int(max_transitions)
    for step in range(total):
        before = plant.snapshot()
        ctrl = policy.action(plant, plant.data, track=track, prior=prior, current_s=current_s)
        plant.step_control(ctrl, substeps=substeps, data=plant.data)
        after = plant.snapshot()
        current_s = float(track.project(plant.xy())[0])
        if step >= int(warmup_steps):
            transitions.append((before, ctrl, after, substeps))
        if plant.has_fallen(flipped_threshold=0.0):
            break
    if len(transitions) < 10:
        raise RuntimeError(f"only collected {len(transitions)} usable transitions")
    return transitions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate an implicitfast Ant planning model against RK4/model-integrator transitions"
    )
    parser.add_argument(
        "--replay", default=None,
        help="flat run replay generated with the reference/model plant integrator; preferred over recollecting",
    )
    parser.add_argument("--policy", default="auto", help="policy used when --replay is omitted")
    parser.add_argument("--transitions", type=int, default=240)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--levels", type=int, default=5, help="coordinate-search refinement levels")
    parser.add_argument("--sweeps-per-level", type=int, default=4)
    parser.add_argument("--damping-step", type=float, default=0.40)
    parser.add_argument("--armature-step", type=float, default=0.40)
    parser.add_argument(
        "--output", default="racing/results/implicitfast_calibration.json",
        help="calibration JSON consumed by race.py --planner-calibration",
    )
    args = parser.parse_args()

    planner = make_robot("ant")
    _set_integrator(planner, "implicitfast")

    if args.replay:
        source = str(Path(args.replay).expanduser().resolve())
        transitions = _load_replay_transitions(Path(args.replay).expanduser(), planner, args.transitions)
    else:
        source = f"nominal_policy:{args.policy}"
        transitions = _collect_nominal_transitions(args.policy, args.transitions, args.warmup_steps)

    cfg = SystemIDConfig(
        history=max(16, len(transitions)),
        update_interval=1,
        smoothing=1.0,
        estimate_friction=False,
        estimate_mass=False,
        estimate_motor=False,
        estimate_slope=False,
        estimate_damping=True,
        estimate_armature=True,
        damping_step=float(args.damping_step),
        armature_step=float(args.armature_step),
    )
    identifier = OnlineSystemIdentifier(planner, cfg, initial_estimate=ModelParameterScales())
    for before, ctrl, after, substeps in transitions:
        identifier.observe(before, ctrl, after, substeps=substeps)

    baseline = ModelParameterScales()
    baseline_loss = identifier.loss(baseline)
    print(f"transitions={len(identifier.history)} baseline_loss={baseline_loss:.8g}")

    for level in range(max(1, int(args.levels))):
        cfg.damping_step = float(args.damping_step) * (0.5 ** level)
        cfg.armature_step = float(args.armature_step) * (0.5 ** level)
        for _ in range(max(1, int(args.sweeps_per_level))):
            identifier.update()
        p = identifier.estimate
        loss = identifier.loss(p)
        print(
            f"level={level + 1} damping={p.damping:.5f} armature={p.armature:.5f} "
            f"loss={loss:.8g}"
        )

    final = identifier.estimate.clipped()
    final_loss = identifier.loss(final)
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "robot": "ant",
        "reference_integrator": "model",
        "planner_integrator": "implicitfast",
        "source": source,
        "num_transitions": len(identifier.history),
        "baseline_loss": float(baseline_loss),
        "calibrated_loss": float(final_loss),
        "improvement_fraction": float((baseline_loss - final_loss) / max(baseline_loss, 1e-12)),
        "parameters": final.to_dict(),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {out}")
    print(
        "use with: --planner-integrator implicitfast "
        f"--planner-calibration {out}"
    )


if __name__ == "__main__":
    main()
