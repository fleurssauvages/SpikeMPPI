from __future__ import annotations

import argparse
import numpy as np
from racing.robots import make_robot
from racing.tracks import StadiumTrack
from racing.priors import GeometricPrior
from racing.policies import make_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="ant", choices=["ant"])
    parser.add_argument("--policy", default="default")
    args = parser.parse_args()
    robot = make_robot(args.robot)
    track = StadiumTrack(origin_xy=tuple(robot.xy()), origin_yaw=robot.root_yaw())
    prior = GeometricPrior()
    policy = make_policy(args.policy)
    ctrl = policy.action(robot, robot.data, track=track, prior=prior, current_s=track.start_s)
    assert ctrl.shape == (robot.nu,)
    snap = robot.snapshot()
    robot.step_control(ctrl, substeps=max(1, int(round(0.02 / robot.physics_dt))))
    moved = np.linalg.norm(robot.xy() - np.asarray(track.origin_xy))
    robot.restore(snap)
    print(f"robot={robot.name} nu={robot.nu} nv={robot.model.nv} dt={robot.physics_dt:g}")
    print(f"root={robot.root_body_name} home_height={robot.initial_root_height:.3f} m")
    print(f"default policy one-step root displacement={moved:.6g} m")
    print("native MuJoCo smoke test: OK")


if __name__ == "__main__":
    main()
