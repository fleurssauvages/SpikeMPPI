from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .rapid_locomotion import (
    GridAdaptiveCurriculum,
    RapidCurriculumConfig,
    RapidDomainRandomizationConfig,
    RapidPPOConfig,
    RapidRewardConfig,
)
from .velocity_env import make_domain_randomizer, make_velocity_env, training_spec


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            arr = np.asarray(value)
            if arr.size == 1:
                out[str(key)] = float(arr.reshape(()))
        except Exception:
            pass
    return out


def _build_network_factory(ppo_networks, linen, ppo_cfg: RapidPPOConfig):
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(ppo_cfg.policy_layers),
        value_hidden_layer_sizes=tuple(ppo_cfg.value_layers),
        distribution_type="tanh_normal",
        activation=linen.elu,
        init_noise_std=float(ppo_cfg.init_noise_std),
    )


def _evaluate_frontier_native(
    *,
    robot_name: str,
    curriculum: GridAdaptiveCurriculum,
    params,
    network_factory,
    observation_size: int,
    action_size: int,
    control_dt: float,
    seconds: float,
    seed: int,
) -> tuple[list[tuple[int, int]], list[dict[str, float]]]:
    """Evaluate active boundary bins and return those eligible to expand.

    This host-side evaluation is the adaptation that lets us preserve a shared
    Grid Adaptive Curriculum while the MJX environment itself remains purely
    jitted/vmapped.  Each frontier command is tested in the same native MuJoCo
    model later used by the racer.
    """
    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from racing.policies.brax_velocity import body_motion_numpy, rapid_observation_numpy
    from racing.robots import make_robot

    networks = network_factory(
        observation_size,
        action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    inference = jax.jit(ppo_networks.make_inference_fn(networks)(params, deterministic=True))
    key = jax.random.PRNGKey(int(seed))

    cfg = curriculum.config
    spec = training_spec(robot_name)
    frontier = curriculum.frontier_indices()
    successful: list[tuple[int, int]] = []
    rows: list[dict[str, float]] = []
    target_steps = max(1, int(round(float(seconds) / float(control_dt))))

    for i, j in frontier:
        robot = make_robot(robot_name)
        robot.reset()
        substeps = max(1, int(round(float(control_dt) / robot.physics_dt)))
        vx = float(curriculum.vx_values[i])
        wz = float(curriculum.wz_values[j])
        command = np.asarray([vx, 0.0, wz], dtype=np.float32)
        previous_action = np.zeros(robot.nu, dtype=np.float64)
        lin_sum = 0.0
        yaw_sum = 0.0
        survived = True

        low, high = robot.control_bounds()
        for _ in range(target_steps):
            obs = rapid_observation_numpy(robot, robot.data, command, previous_action)
            key, act_key = jax.random.split(key)
            normalized, _ = inference(jnp.asarray(obs), act_key)
            previous_action = np.clip(np.asarray(normalized, dtype=np.float64).reshape(robot.nu), -1.0, 1.0)
            ctrl = 0.5 * (low + high) + 0.5 * (high - low) * previous_action
            robot.step_control(ctrl, substeps=substeps)

            body_linear, body_angular = body_motion_numpy(robot, robot.data)
            lin_error2 = float(np.sum((body_linear[:2] - command[:2]) ** 2))
            yaw_error2 = float((body_angular[2] - command[2]) ** 2)
            lin_sum += math.exp(-lin_error2 / 0.25)
            yaw_sum += math.exp(-yaw_error2 / 0.25)
            if (
                robot.root_height() < float(spec.healthy_height_fraction) * max(robot.initial_root_height, 1e-6)
                or robot.root_up() < float(spec.min_root_up)
                or not np.all(np.isfinite(robot.data.qpos))
                or not np.all(np.isfinite(robot.data.qvel))
            ):
                survived = False
                break

        # Divide by the requested duration, not survived steps: a fall therefore
        # automatically fails the curriculum threshold.
        lin_score = lin_sum / target_steps
        yaw_score = yaw_sum / target_steps
        passed = (
            survived
            and lin_score >= float(cfg.forward_success_threshold)
            and yaw_score >= float(cfg.yaw_success_threshold)
        )
        if passed:
            successful.append((i, j))
        rows.append({
            "vx": vx,
            "wz": wz,
            "tracking_lin": lin_score,
            "tracking_yaw": yaw_score,
            "survived": float(survived),
            "passed": float(passed),
        })

    return successful, rows


def _save_training_state(
    output: Path,
    *,
    brax_model,
    params,
    metadata: dict,
    curriculum: GridAdaptiveCurriculum,
) -> None:
    brax_model.save_params(str(output / "params.pkl"), params)
    metadata = dict(metadata)
    metadata["curriculum"] = curriculum.to_dict()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "curriculum.json").write_text(json.dumps(curriculum.to_dict(), indent=2), encoding="utf-8")


def train_policy(
    *,
    robot: str,
    output: str | Path,
    impl: str = "warp",
    num_envs: int = 4096,
    total_steps: int = 400_000_000,
    phase_steps: int = 20_000_000,
    episode_length: int = 1000,
    seed: int = 1,
    ctrl_dt: float | None = None,
    contacts_per_env: int = 8,
    njmax: int = 512,
    domain_randomization: bool = True,
    pushes: bool = True,
    resume: bool = False,
    frontier_eval_seconds: float = 4.0,
    until_failure: bool = False,
    stall_patience: int = 4,
    max_phases: int = 100,
    max_forward_speed: float | None = None,
):
    try:
        import jax
        from flax import linen
        from brax.io import model as brax_model
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo
        from mujoco_playground import wrapper
    except ImportError as exc:
        raise RuntimeError(
            "Training requires current Brax main, JAX, Flax, mujoco-mjx and mujoco-playground."
        ) from exc

    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    ppo_cfg = RapidPPOConfig(num_envs=int(num_envs), total_timesteps=int(total_steps))
    curriculum_cfg = RapidCurriculumConfig(
        phase_timesteps=max(1, int(phase_steps)),
        frontier_eval_seconds=float(frontier_eval_seconds),
    )
    reward_cfg = RapidRewardConfig()
    domain_cfg = RapidDomainRandomizationConfig()
    curriculum = GridAdaptiveCurriculum(curriculum_cfg)
    params = None
    completed_steps = 0
    phase_index = 0
    speed_stall_phases = 0

    metadata_path = output / "metadata.json"
    params_path = output / "params.pkl"
    curriculum_path = output / "curriculum.json"
    if resume and params_path.exists() and metadata_path.exists():
        params = brax_model.load_params(str(params_path))
        old_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        completed_steps = int(old_meta.get("completed_steps", 0))
        phase_index = int(old_meta.get("completed_phases", 0))
        speed_stall_phases = int(old_meta.get("speed_stall_phases", 0))
        if curriculum_path.exists():
            curriculum = GridAdaptiveCurriculum.from_dict(
                json.loads(curriculum_path.read_text(encoding="utf-8"))
            )
        print(f"Resuming at {completed_steps:,}/{total_steps:,} steps from {output}")

    stall_patience = max(1, int(stall_patience))
    max_phases = max(1, int(max_phases))
    if max_forward_speed is not None and float(max_forward_speed) <= 0.0:
        raise ValueError("max_forward_speed must be positive when provided")

    contacts_per_env = max(1, int(contacts_per_env))
    naconmax = max(512, contacts_per_env * int(num_envs))
    spec = training_spec(robot)
    use_ctrl_dt = float(ctrl_dt if ctrl_dt is not None else spec.ctrl_dt)

    network_factory = _build_network_factory(ppo_networks, linen, ppo_cfg)
    randomization_fn = None
    if domain_randomization:
        randomization_fn = make_domain_randomizer(
            friction_range=(domain_cfg.friction_min, domain_cfg.friction_max),
            motor_strength_range=(domain_cfg.motor_strength_min, domain_cfg.motor_strength_max),
        )

    print("JAX backend:", jax.default_backend())
    print("Devices:", jax.devices())
    print(f"Rapid-Locomotion-style training: robot={robot}, envs={num_envs}, impl={impl}")
    if until_failure:
        print(
            f"target=until straight-speed stall, curriculum phase={phase_steps:,} steps, "
            f"patience={stall_patience} phases, safety max_phases={max_phases}"
        )
    else:
        print(f"target={total_steps:,} steps, curriculum phase={phase_steps:,} steps")
    print(f"PPO: rollout={ppo_cfg.unroll_length}, epochs={ppo_cfg.num_updates_per_batch}, "
          f"minibatches={ppo_cfg.num_minibatches}, lr={ppo_cfg.learning_rate:g}")
    print(f"Warp buffers: naconmax={naconmax} total ({contacts_per_env}/env), njmax={int(njmax)}/env")

    global_started = time.time()
    final_metrics: dict[str, Any] = {}
    last_env = None

    while True:
        if until_failure:
            if phase_index >= max_phases:
                print(f"stopping: reached --max-phases={max_phases} safety limit")
                break
        elif completed_steps >= int(total_steps):
            break

        phase_index += 1
        requested_phase_steps = (
            int(phase_steps)
            if until_failure
            else min(int(phase_steps), int(total_steps) - completed_steps)
        )
        active_cells = curriculum.active_cells()
        print(
            f"\nphase {phase_index}: active_bins={len(active_cells)}, "
            f"learned_straight_speed={curriculum.learned_forward_speed():.2f} m/s, "
            f"learned_yaw={curriculum.learned_yaw_rate():.2f} rad/s"
        )

        env = make_velocity_env(
            robot,
            impl=impl,
            command_cells=active_cells,
            curriculum_config=curriculum.config,
            reward_config=reward_cfg,
            domain_config=domain_cfg,
            ctrl_dt=ctrl_dt,
            episode_length=episode_length,
            command_hold_s=curriculum.config.command_hold_s,
            enable_pushes=bool(pushes),
            naconmax=naconmax,
            njmax=int(njmax),
        )
        last_env = env
        phase_started = time.time()

        def progress(step: int, metrics: dict) -> None:
            reward = metrics.get("eval/episode_reward", float("nan"))
            lin = metrics.get("eval/episode_tracking_lin_vel_per_step", float("nan"))
            yaw = metrics.get("eval/episode_tracking_ang_vel_per_step", float("nan"))
            global_target = "until-stall" if until_failure else f"{total_steps}"
            print(
                f"phase={phase_index:02d} global~{completed_steps + int(step):>10d}/{global_target} "
                f"eval_reward={float(np.asarray(reward)):.3f} "
                f"lin={float(np.asarray(lin)):.3f} yaw={float(np.asarray(yaw)):.3f} "
                f"phase_time={(time.time()-phase_started)/60:.1f}m",
                flush=True,
            )

        phase_ckpt = (output / "brax_checkpoints" / f"phase_{phase_index:03d}").resolve()
        phase_ckpt.mkdir(parents=True, exist_ok=True)
        make_policy, params, final_metrics = ppo.train(
            environment=env,
            eval_env=env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            randomization_fn=randomization_fn,
            num_timesteps=int(requested_phase_steps),
            num_envs=int(num_envs),
            episode_length=int(episode_length),
            action_repeat=1,
            learning_rate=float(ppo_cfg.learning_rate),
            entropy_cost=float(ppo_cfg.entropy_cost),
            discounting=float(ppo_cfg.discounting),
            unroll_length=int(ppo_cfg.unroll_length),
            batch_size=int(ppo_cfg.batch_size),
            num_minibatches=int(ppo_cfg.num_minibatches),
            num_updates_per_batch=int(ppo_cfg.num_updates_per_batch),
            normalize_observations=bool(ppo_cfg.normalize_observations),
            reward_scaling=1.0,
            clipping_epsilon=float(ppo_cfg.clipping_epsilon),
            gae_lambda=float(ppo_cfg.gae_lambda),
            max_grad_norm=float(ppo_cfg.max_grad_norm),
            vf_loss_coefficient=float(ppo_cfg.vf_loss_coefficient),
            normalize_advantage=True,
            num_evals=1,
            num_eval_envs=min(128, max(8, int(num_envs) // 8)),
            deterministic_eval=True,
            network_factory=network_factory,
            progress_fn=progress,
            policy_params_fn=lambda *args: None,
            seed=int(seed + phase_index - 1),
            save_checkpoint_path=str(phase_ckpt),
            restore_params=params,
        )
        del make_policy

        # Brax rounds a training phase upward to its rollout/minibatch quantum.
        # Track the requested paper budget so the CLI remains intuitive.
        completed_steps += int(requested_phase_steps)

        speed_before_eval = curriculum.certified_forward_speed()
        successes, frontier_rows = _evaluate_frontier_native(
            robot_name=robot,
            curriculum=curriculum,
            params=params,
            network_factory=network_factory,
            observation_size=int(env.observation_size),
            action_size=int(env.action_size),
            control_dt=float(env.dt),
            seconds=float(curriculum.config.frontier_eval_seconds),
            seed=int(seed + 10_000 + phase_index),
        )
        added = curriculum.expand_from_successes(successes)
        extended = 0
        if until_failure:
            extended = curriculum.extend_forward_from_successes(
                successes,
                max_forward_speed=max_forward_speed,
            )

        speed_after_eval = curriculum.certified_forward_speed()
        speed_progress = speed_after_eval > speed_before_eval + 0.25 * float(curriculum.config.grid_step_vx)
        if until_failure:
            speed_stall_phases = 0 if speed_progress else speed_stall_phases + 1

        if frontier_rows:
            best_vx = max((r["vx"] for r in frontier_rows if r["passed"] > 0.5), default=float("nan"))
            print(
                f"curriculum: tested {len(frontier_rows)} frontier bins, "
                f"passed={len(successes)}, added={added}, forward_rows_extended={extended}, "
                f"best_passed_vx={best_vx:.2f}, certified_straight={speed_after_eval:.2f} m/s"
            )
            if until_failure:
                print(f"speed-stall counter: {speed_stall_phases}/{stall_patience}")
        else:
            print("curriculum: grid has no uncertified frontier left")

        metadata = {
            "format": "racing_rapid_locomotion_policy_v2",
            "paper": "Margolis et al., Rapid Locomotion via Reinforcement Learning",
            "robot": str(robot),
            "impl": str(impl),
            "observation_version": "rapid_v2",
            "observation_size": int(env.observation_size),
            "action_size": int(env.action_size),
            "control_dt": float(env.dt),
            "policy_hidden_layer_sizes": list(ppo_cfg.policy_layers),
            "value_hidden_layer_sizes": list(ppo_cfg.value_layers),
            "activation": "elu",
            "init_noise_std": float(ppo_cfg.init_noise_std),
            "normalize_observations": True,
            "training_mode": "until_failure" if until_failure else "fixed_steps",
            "total_steps_target": None if until_failure else int(total_steps),
            "completed_steps": int(completed_steps),
            "completed_phases": int(phase_index),
            "phase_steps": int(phase_steps),
            "speed_stall_phases": int(speed_stall_phases),
            "stall_patience": int(stall_patience),
            "max_phases": int(max_phases),
            "max_forward_speed_guard": None if max_forward_speed is None else float(max_forward_speed),
            "num_envs": int(num_envs),
            "seed": int(seed),
            "contacts_per_env": int(contacts_per_env),
            "naconmax": int(naconmax),
            "njmax": int(njmax),
            "ppo": {
                "learning_rate": ppo_cfg.learning_rate,
                "discounting": ppo_cfg.discounting,
                "gae_lambda": ppo_cfg.gae_lambda,
                "unroll_length": ppo_cfg.unroll_length,
                "num_updates_per_batch": ppo_cfg.num_updates_per_batch,
                "num_minibatches": ppo_cfg.num_minibatches,
                "batch_size": ppo_cfg.batch_size,
                "entropy_cost": ppo_cfg.entropy_cost,
                "clipping_epsilon": ppo_cfg.clipping_epsilon,
                "vf_loss_coefficient": ppo_cfg.vf_loss_coefficient,
            },
            "reward": reward_cfg.__dict__,
            "domain_randomization": {
                **domain_cfg.__dict__,
                "enabled": bool(domain_randomization),
                "pushes_enabled": bool(pushes),
                "mjx_direct_fields": ["geom_friction", "actuator_gainprm"],
                "paper_fields_not_directly_batched": ["payload_mass", "body_com", "restitution"],
            },
            "final_metrics": _scalar_metrics(final_metrics),
            "last_frontier_evaluation": frontier_rows,
            "race_heading_gain": 2.0,
            "race_curvature_lookahead_m": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
            "implementation_notes": [
                "Grid curriculum is shared and updated between PPO phases using native-MuJoCo frontier evaluation.",
                "Adam optimizer moments restart at phase boundaries because current Brax restore_params restores network/normalizer parameters only.",
                "Mini-Cheetah-specific feet-air-time/collision topology terms are omitted for classic MuJoCo Ant/Humanoid.",
                "Policy actions are native normalized MuJoCo actuator controls, not PD joint-position targets.",
            ],
        }
        _save_training_state(
            output,
            brax_model=brax_model,
            params=params,
            metadata=metadata,
            curriculum=curriculum,
        )
        (output / f"frontier_phase_{phase_index:03d}.json").write_text(
            json.dumps(frontier_rows, indent=2), encoding="utf-8"
        )

        if until_failure:
            learned_speed = curriculum.learned_forward_speed()
            if max_forward_speed is not None and learned_speed >= float(max_forward_speed) - 1e-9:
                print(
                    f"stopping: certified straight speed reached the guard "
                    f"{float(max_forward_speed):.2f} m/s"
                )
                break
            if speed_stall_phases >= stall_patience:
                print(
                    f"stopping: no increase in certified straight speed for "
                    f"{speed_stall_phases} consecutive PPO phases; "
                    f"last certified speed={learned_speed:.2f} m/s"
                )
                break

    elapsed_h = (time.time() - global_started) / 3600.0
    print(f"\nSaved final policy to {output}")
    print(f"Training wall time: {elapsed_h:.2f} h")
    print(f"Learned straight speed envelope: {curriculum.learned_forward_speed():.2f} m/s")
    print(f"Learned yaw envelope: {curriculum.learned_yaw_rate():.2f} rad/s")
    return output


def main() -> None:
    paper = RapidPPOConfig()
    curriculum = RapidCurriculumConfig()
    parser = argparse.ArgumentParser(
        description="Train a Rapid-Locomotion-style high-speed Ant/Humanoid policy with MJX/Brax PPO"
    )
    parser.add_argument("--robot", choices=["ant", "humanoid"], default="ant")
    parser.add_argument("--output", default=None)
    parser.add_argument("--impl", choices=["warp", "jax"], default="warp")
    parser.add_argument("--num-envs", type=int, default=paper.num_envs)
    parser.add_argument("--steps", type=int, default=paper.total_timesteps)
    parser.add_argument("--phase-steps", type=int, default=curriculum.phase_timesteps)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ctrl-dt", type=float, default=None)
    parser.add_argument(
        "--contacts-per-env",
        type=int,
        default=8,
        help="MuJoCo-Warp broadphase/contact capacity per parallel world",
    )
    parser.add_argument("--njmax", type=int, default=512, help="MuJoCo-Warp constraint capacity per world")
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--no-pushes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frontier-eval-seconds", type=float, default=curriculum.frontier_eval_seconds)
    parser.add_argument(
        "--until-failure",
        action="store_true",
        help=(
            "keep running full PPO curriculum phases, extend +vx by one grid step after the "
            "straight outer edge passes, and stop after repeated phases without speed progress"
        ),
    )
    parser.add_argument(
        "--stall-patience",
        type=int,
        default=4,
        help="consecutive PPO phases without a higher certified straight speed before stopping",
    )
    parser.add_argument(
        "--max-phases",
        type=int,
        default=100,
        help="safety limit for --until-failure",
    )
    parser.add_argument(
        "--max-forward-speed",
        type=float,
        default=None,
        help="optional safety ceiling in m/s for dynamically extended forward curriculum",
    )
    args = parser.parse_args()

    output = args.output or f"racing/policies/checkpoints/{args.robot}_rapid"
    train_policy(
        robot=args.robot,
        output=output,
        impl=args.impl,
        num_envs=args.num_envs,
        total_steps=args.steps,
        phase_steps=args.phase_steps,
        episode_length=args.episode_length,
        seed=args.seed,
        ctrl_dt=args.ctrl_dt,
        contacts_per_env=args.contacts_per_env,
        njmax=args.njmax,
        domain_randomization=not args.no_domain_randomization,
        pushes=not args.no_pushes,
        resume=args.resume,
        frontier_eval_seconds=args.frontier_eval_seconds,
        until_failure=args.until_failure,
        stall_patience=args.stall_patience,
        max_phases=args.max_phases,
        max_forward_speed=args.max_forward_speed,
    )


if __name__ == "__main__":
    main()
