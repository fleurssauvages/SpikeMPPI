from __future__ import annotations

import argparse
import numpy as np

from racing.robots import make_robot


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MuJoCo robot smoke test")
    parser.add_argument("--robot", default="ant", choices=["ant"])
    args = parser.parse_args()

    robot = make_robot(args.robot)
    before = np.asarray(robot.xy(), dtype=np.float64).copy()
    ctrl = np.zeros(robot.nu, dtype=np.float64)
    robot.step_control(ctrl, substeps=max(1, int(round(0.02 / robot.physics_dt))))
    after = np.asarray(robot.xy(), dtype=np.float64)
    moved = float(np.linalg.norm(after - before))
    print(f"zero-control one-step root displacement={moved:.6g} m")


if __name__ == "__main__":
    main()
