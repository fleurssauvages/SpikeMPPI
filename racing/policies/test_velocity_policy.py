from __future__ import annotations

import argparse
import time
import numpy as np

from racing.policies.brax_velocity import BraxVelocityPolicy
from racing.robots import make_robot
from racing.tracks import close_viewer, launch_minimal_viewer, safe_viewer_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual high-speed policy test without MPPI")
    parser.add_argument("--robot", default="ant", choices=["ant", "spinner", "snake", "crawler", "biped"])
    parser.add_argument("--policy", required=True, help="checkpoint directory")
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--max-speed", type=float, default=None, help="optional visualization speed cap")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    robot = make_robot(args.robot)
    policy = BraxVelocityPolicy(args.policy)
    policy.reset(robot, robot.data)
    ratio = policy.control_dt / robot.physics_dt
    substeps = max(1, int(round(ratio)))

    vmax = policy.learned_max_speed
    if args.max_speed is not None:
        vmax = min(vmax, max(0.0, float(args.max_speed)))
    vmax = max(0.5, vmax)
    yawmax = max(0.5, policy.command_envelope.max_yaw)
    speeds = sorted(set([0.0, min(1.0, vmax), min(2.0, vmax), min(3.0, vmax), vmax]))
    commands = [np.asarray([v, 0.0, 0.0], dtype=np.float32) for v in speeds]
    turn_speed = min(max(1.0, 0.65 * vmax), vmax)
    turn_rate = min(yawmax, 1.5)
    commands += [
        np.asarray([turn_speed, 0.0, turn_rate], dtype=np.float32),
        np.asarray([turn_speed, 0.0, -turn_rate], dtype=np.float32),
        np.asarray([0.5, min(0.5, policy.command_envelope.vy_max), 0.0], dtype=np.float32),
        np.asarray([0.5, max(-0.5, policy.command_envelope.vy_min), 0.0], dtype=np.float32),
    ]

    print(
        f"checkpoint envelope: straight={policy.learned_max_speed:.2f} m/s, "
        f"yaw={policy.command_envelope.max_yaw:.2f} rad/s"
    )

    handle = None
    if not args.headless:
        handle = launch_minimal_viewer(robot.model, robot.data, show_ui=False)
    try:
        steps_per_segment = max(1, int(round(args.segment_seconds / policy.control_dt)))
        for command in commands:
            print("command", command.tolist())
            for _ in range(steps_per_segment):
                if handle is not None and not handle.is_running():
                    return
                ctrl = policy.action_for_command(robot, robot.data, command)
                robot.step_control(ctrl, substeps=substeps)
                if handle is not None:
                    if not safe_viewer_sync(handle, state_only=True):
                        handle = None
                    time.sleep(max(0.0, 0.25 * policy.control_dt))
                from racing.policies.velocity_env import training_spec
                spec = training_spec(robot.name)
                if (
                    robot.root_height() < float(spec.healthy_height_fraction) * robot.initial_root_height
                    or robot.root_up() < float(spec.min_root_up)
                ):
                    print("robot fell; ending test")
                    return
    finally:
        if handle is not None:
            close_viewer(handle)


if __name__ == "__main__":
    main()
