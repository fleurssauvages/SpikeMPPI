from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from racing.experiments.race import run_race
from racing.mppi import ControllerVariant


def _row(label: str, horizon: int, result):
    return {
        "configuration": label,
        "horizon": int(horizon),
        "completed_laps": int(result.completed_laps),
        "progress_m": float(result.cumulative_progress[-1]),
        "simulated_time_s": float(result.simulated_time_s),
        "compute_time_s": float(result.runtime_s),
        "mean_ess": float(np.nanmean(result.esses)) if len(result.esses) else float("nan"),
        "fell": bool(result.fell),
        "off_track": bool(result.off_track),
        "lap_time_s": float(result.lap_times[-1]) if result.lap_times else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare matched RK4/model, mismatched implicitfast planner, and matched implicitfast across horizons"
    )
    parser.add_argument("--policy", default="auto")
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 24, 32, 50])
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--joint-noise", type=float, default=0.5)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-backend", choices=["auto", "fused", "native", "python"], default="fused")
    parser.add_argument("--planner-calibration", default=None)
    parser.add_argument("--csv", default="racing/results/integrator_comparison.csv")
    args = parser.parse_args()

    rows = []
    for horizon in args.horizons:
        configs = [
            ("matched_model", "model", "model", None),
            ("rk4_plant_implicitfast_planner", "model", "implicitfast", None),
            ("matched_implicitfast", "implicitfast", "implicitfast", None),
        ]
        if args.planner_calibration:
            configs.append(
                ("rk4_plant_implicitfast_calibrated", "model", "implicitfast", args.planner_calibration)
            )
        for label, plant_integrator, planner_integrator, calibration in configs:
            print(f"H={horizon} {label}")
            result = run_race(
                robot_name="ant",
                policy_spec=args.policy,
                variant=ControllerVariant.MPPI,
                num_rollouts=args.rollouts,
                horizon=int(horizon),
                joint_noise_fraction=float(args.joint_noise),
                laps=int(args.laps),
                max_steps=args.max_steps,
                seed=int(args.seed),
                viewer=False,
                verbose=False,
                rollout_workers=int(args.workers),
                rollout_backend=args.rollout_backend,
                plant_integrator=plant_integrator,
                planner_integrator=planner_integrator,
                planner_calibration=calibration,
            )
            row = _row(label, horizon, result)
            rows.append(row)
            print(
                f"  progress={row['progress_m']:.2f}m lap={row['lap_time_s']:.3g}s "
                f"compute={row['compute_time_s']:.2f}s ESS={row['mean_ess']:.2f} "
                f"fell={row['fell']} off_track={row['off_track']}"
            )

    out = Path(args.csv).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
