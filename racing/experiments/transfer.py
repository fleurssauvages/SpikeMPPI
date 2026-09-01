from __future__ import annotations

import argparse
from pathlib import Path

from racing.experiments.race import run_race
from racing.priors import GeometricPrior, distill_empirical_prior


def main() -> None:
    parser = argparse.ArgumentParser(description="Classic-MuJoCo SPG prior transfer")
    parser.add_argument("--source", default="ant")
    parser.add_argument("--target", default="humanoid")
    parser.add_argument("--source-policy", default="default")
    parser.add_argument("--target-policy", default="default")
    parser.add_argument("--source-laps", type=int, default=5)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--normal-floor-std", type=float, default=0.15)
    parser.add_argument("--prior-samples", type=int, default=512)
    parser.add_argument("--prior-out", default="results/transferred_prior.npz")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="disable viewers; viewers are ON by default")
    args = parser.parse_args()

    source = run_race(
        robot_name=args.source,
        laps=args.source_laps,
        prior=GeometricPrior(),
        policy_spec=args.source_policy,
        num_rollouts=args.rollouts,
        horizon=args.horizon,
        control_dt=args.dt,
        seed=args.seed,
        viewer=not args.headless,
    )
    if source.completed_laps < args.source_laps:
        raise RuntimeError(
            f"Source completed {source.completed_laps}/{args.source_laps} laps; "
            "refusing to build a transfer prior from incomplete data."
        )

    prior = distill_empirical_prior(
        source.track,
        source.xy,
        source.cumulative_progress,
        source.completed_laps,
        samples=args.prior_samples,
        normal_floor_std=args.normal_floor_std,
    )
    out = prior.save(Path(args.prior_out))
    print(f"saved {prior.source_laps}-lap prior: {out}")

    target = run_race(
        robot_name=args.target,
        laps=args.target_laps,
        prior=prior,
        policy_spec=args.target_policy,
        num_rollouts=args.rollouts,
        horizon=args.horizon,
        control_dt=args.dt,
        seed=args.seed + 1,
        viewer=not args.headless,
    )
    print(
        f"target {target.robot_name}: {target.completed_laps}/{target.requested_laps} laps, "
        f"off_track={target.off_track}, fell={target.fell}"
    )


if __name__ == "__main__":
    main()
