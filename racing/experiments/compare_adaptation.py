from __future__ import annotations

import argparse
import csv
from pathlib import Path

from racing.experiments.race import run_race
from racing.mppi import ControllerVariant


def _row(label, result):
    return {
        "mode": label,
        "completed_laps": result.completed_laps,
        "off_track": result.off_track,
        "fell": result.fell,
        "progress_m": float(result.cumulative_progress[-1]),
        "simulated_time_s": result.simulated_time_s,
        "compute_time_s": result.runtime_s,
        "mean_ess": float(result.esses.mean()) if len(result.esses) else float("nan"),
        "final_friction_est": float(result.model_estimates[-1, 0]),
        "final_mass_est": float(result.model_estimates[-1, 1]),
        "final_motor_est": float(result.model_estimates[-1, 2]),
        "final_slope_est_deg": float(result.model_estimates[-1, 3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare policy-only, fixed-model SPG-MPPI, and adaptive-model SPG-MPPI on the same perturbed MuJoCo plant"
    )
    parser.add_argument("--robot", choices=["ant", "humanoid"], default="ant")
    parser.add_argument("--policy", default="auto")
    parser.add_argument("--policy-speed", type=float, default=None)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--friction-scale", type=float, default=0.55)
    parser.add_argument("--mass-scale", type=float, default=1.10)
    parser.add_argument("--motor-scale", type=float, default=0.90)
    parser.add_argument("--slope-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--viewer", action="store_true", help="show each run sequentially; headless by default")
    parser.add_argument("--csv", default="racing/results/adaptation_comparison.csv")
    args = parser.parse_args()

    common = dict(
        robot_name=args.robot,
        policy_spec=args.policy,
        policy_speed=args.policy_speed,
        laps=args.laps,
        max_steps=args.max_steps,
        friction_scale=args.friction_scale,
        mass_scale=args.mass_scale,
        motor_scale=args.motor_scale,
        slope_deg=args.slope_deg,
        seed=args.seed,
        viewer=args.viewer,
        verbose=False,
    )

    print("1/3 policy-only")
    policy_only = run_race(
        **common,
        variant=ControllerVariant.POLICY_NOMINAL,
        num_rollouts=1,
        horizon=args.horizon,
    )
    print("2/3 SPG-MPPI, fixed nominal model")
    fixed = run_race(
        **common,
        variant=ControllerVariant.SPG_MPPI,
        num_rollouts=args.rollouts,
        horizon=args.horizon,
        online_adaptation=False,
    )
    print("3/3 SPG-MPPI, online model adaptation")
    adaptive = run_race(
        **common,
        variant=ControllerVariant.SPG_MPPI,
        num_rollouts=args.rollouts,
        horizon=args.horizon,
        online_adaptation=True,
        sysid_estimate_slope=abs(args.slope_deg) > 1e-9,
    )

    rows = [
        _row("policy_only", policy_only),
        _row("spg_fixed_model", fixed),
        _row("spg_adaptive_model", adaptive),
    ]
    fields = list(rows[0])
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"{'mode':<22} {'laps':>5} {'progress':>10} {'fell':>7} {'offtrk':>7} {'compute':>10}")
    for row in rows:
        print(
            f"{row['mode']:<22} {row['completed_laps']:>5d} {row['progress_m']:>10.2f} "
            f"{str(row['fell']):>7} {str(row['off_track']):>7} {row['compute_time_s']:>10.2f}"
        )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
