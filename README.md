# Racing: Rapid-Locomotion policy + direct-joint SPG-MPPI + online MuJoCo adaptation

This project uses the simple classic MuJoCo locomotion robots, with **Ant** as
the primary 2-D stadium racer.  The default policy trainer is an adaptation of
Margolis et al., *Rapid Locomotion via Reinforcement Learning* (RSS 2022 / IJRR
2024) to the classic MuJoCo Ant/Humanoid models and the current MJX/Warp + Brax
training stack.

Paper: https://doi.org/10.1177/02783649231224053
Released reference code: https://github.com/Improbable-AI/rapid-locomotion-rl

The online racing controller is intentionally separate from the locomotion
policy:

```text
track curvature / local path
          |
          v
fastest certified [vx, vy, wz] command
          |
          v
Rapid-Locomotion-style policy pi
          |
          v
native joint nominal U_bar [H, nu]
          |
          v
SPG direct-joint MPPI [N, H, nu]
          |
          v
       MuJoCo plant
          |
   recent transitions
          |
          v
online MuJoCo model identification
          |
          +------> next MPPI planning model
```

MPPI never optimizes a high-level velocity command.  It directly perturbs and
optimizes `MjData.ctrl` for every robot actuator. SPG is now the default proposal: the spatial prior covariance is center-corrected about the policy nominal, projected through the MuJoCo joint-to-planar-position sensitivity, and sampled directly in joint space. Standard MPPI remains available only as an ablation.

## What is reproduced from Rapid Locomotion

The default `rapid` trainer uses the paper/released-code choices that transfer
cleanly across morphologies:

- 50 Hz policy/controller rate;
- joint Grid Adaptive Curriculum over `(v_x, omega_z)`;
- initial command area `v_x in [-1,1] m/s`, `omega_z in [-1,1] rad/s`;
- 0.5 m/s x 0.5 rad/s curriculum grid;
- lateral command sampled separately from a small interval;
- 10 s command resampling interval;
- tracking thresholds 0.8 linear and 0.5 yaw;
- PPO: gamma 0.99, GAE 0.95, rollout length 21, 5 epochs, 4 minibatches,
  entropy 0.01, value coefficient 1.0, clip 0.2, learning rate 1e-3;
- 4096 parallel environments and a 400M-step full training budget;
- ELU policy body with `[512, 256, 128]` hidden layers;
- velocity tracking reward `exp(-error^2 / 0.25)` and the released auxiliary
  scales for vertical velocity, roll/pitch angular velocity, torque, joint
  acceleration, and action rate;
- ground-friction and motor-strength randomization plus periodic pushes.

The curriculum is stored in the checkpoint.  Newly unlocked bins are used for
training, but only bins that have subsequently passed the tracking test are
marked **certified**.  Racing selects the fastest command from this certified
joint `(v_x, omega_z)` envelope, so an untested outer curriculum shell is not
mistaken for a demonstrated capability.

## Deliberate differences from the Mini Cheetah paper

This is not a bit-for-bit Mini Cheetah reproduction.  It is the paper's method
ported to the classic MuJoCo Ant/Humanoid:

1. The original robot outputs 12 joint-position targets to a low-gain PD loop.
   Ant/Humanoid here output normalized native MuJoCo actuator controls directly.
2. Mini-Cheetah-specific feet-air-time, knee collision, joint-limit and body
   height/orientation terms are not copied blindly onto a different morphology.
3. The paper's teacher/student latent dynamics module is not used as the race
   adaptation mechanism.  This project deliberately uses explicit online
   MuJoCo system identification plus MPPI so adaptation can be measured
   separately from the learned locomotion prior.
4. The released Grid Adaptive Curriculum updates inside the massively parallel
   simulator.  Here a shared grid is updated between PPO phases.  Each frontier
   bin is tested in native MuJoCo, successful bins are certified, and neighbors
   are unlocked for the next phase.  Current Brax `restore_params` restores the
   normalizer/policy/value parameters but not Adam moments, so optimizer moments
   restart at phase boundaries.
5. For fast MJX batching, training randomizes directly batchable friction and
   actuator-gain fields.  Mass/COM/restitution are retained as paper metadata;
   mass and motor/friction mismatch are handled explicitly by the race-time
   model-adaptation experiment.

These differences are recorded in every checkpoint `metadata.json`.

## Environment

Use the environment that is already running MuJoCo/MJX-Warp:

```bash
cd /home/alexis/Desktop/Transfer
source mujoco_env/bin/activate
```

The current JAX release requires current upstream Brax rather than PyPI 0.14.2
on this setup:

```bash
python -m pip uninstall -y brax
python -m pip install --no-cache-dir "brax @ git+https://github.com/google/brax.git"
```

Check the stack:

```bash
python - <<'PY'
import jax, brax, mujoco
print("jax:", jax.__version__)
print("brax:", getattr(brax, "__version__", "git/source"))
print("mujoco:", mujoco.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY
```

For GPUs where reduced-precision matrix multiplication hurts training
reproducibility, MuJoCo Playground recommends:

```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

## 1. Smoke-train Ant

Use one curriculum phase for a short end-to-end check:

```bash
python -m racing.policies.train_velocity_policy \
  --robot ant \
  --impl warp \
  --num-envs 1024 \
  --steps 1000000 \
  --phase-steps 1000000 \
  --contacts-per-env 8 \
  --output racing/policies/checkpoints/ant_rapid_test
```

The Warp broadphase capacity is scaled with the environment count.  If a run
still reports `broadphase overflow`, retry with `--contacts-per-env 12`.
Checkpoint paths are converted to absolute paths before Orbax is called.

The 1M smoke run is only a software test; do not expect it to learn the high
speed envelope.

## 2. Full paper-style Ant training

```bash
python -m racing.policies.train_velocity_policy \
  --robot ant \
  --impl warp \
  --num-envs 4096 \
  --steps 400000000 \
  --phase-steps 20000000 \
  --contacts-per-env 8 \
  --output racing/policies/checkpoints/ant_rapid
```

This gives 20 curriculum updates over the 400M-step budget.  The command grid
extends to +/-6 m/s and +/-6 rad/s, but it expands only where the current policy
passes the tracking thresholds.

To keep refining the forward-speed envelope until it stops improving, use
`--until-failure`.  This keeps the same PPO phase size, reward, 0.5 m/s grid
spacing, and frontier certification thresholds.  Once the current positive
straight-line edge is certified, one new +0.5 m/s row is appended.  Training
stops after `--stall-patience` consecutive full PPO phases without a higher
certified straight speed (with `--max-phases` as a safety guard).

For the included `ant_rapid` checkpoint, this can continue directly beyond its
already-completed 400M-step run:

```bash
python -m racing.policies.train_velocity_policy \
  --robot ant \
  --impl warp \
  --num-envs 4096 \
  --phase-steps 20000000 \
  --contacts-per-env 8 \
  --output racing/policies/checkpoints/ant_rapid \
  --resume \
  --until-failure \
  --stall-patience 4 \
  --max-phases 100
```

Optionally add `--max-forward-speed 12` (or another value) as an explicit safety
ceiling.  Without it, the dynamic +vx grid is limited only by the stall criterion
and `--max-phases`.  `--steps` is ignored in `--until-failure` mode.

Resume an interrupted fixed-budget run with the same settings plus `--resume`:

```bash
python -m racing.policies.train_velocity_policy \
  --robot ant \
  --impl warp \
  --num-envs 4096 \
  --steps 400000000 \
  --phase-steps 20000000 \
  --contacts-per-env 8 \
  --output racing/policies/checkpoints/ant_rapid \
  --resume
```

For diagnosing a custom MJX/Warp installation, domain randomization and pushes
can be disabled independently with `--no-domain-randomization` and
`--no-pushes`.  Those flags are debugging aids, not the recommended full run.

The final checkpoint contains at least:

```text
racing/policies/checkpoints/ant_rapid/
├── params.pkl
├── metadata.json
├── curriculum.json
├── frontier_phase_001.json
├── frontier_phase_002.json
├── ...
└── brax_checkpoints/
```

`metadata.json` records the final certified velocity/yaw envelope and all
training settings.

## 3. Visualize the learned locomotion policy

```bash
python -m racing.policies.test_velocity_policy \
  --robot ant \
  --policy racing/policies/checkpoints/ant_rapid
```

The minimal MuJoCo viewer is active by default.  The test cycles through
standing, increasing forward speeds, turning, and lateral commands.  It prints
the maximum certified straight speed and yaw rate before playback.

A smoke checkpoint can be visualized by replacing `ant_rapid` with
`ant_rapid_test`.

## 4. Policy-only fastest-feasible stadium run

The race command generator looks ahead at stadium curvature.  For each lookahead
location it asks the checkpoint's certified `(v_x, omega_z)` grid for the
largest forward velocity compatible with the required yaw rate.  The most
restrictive lookahead speed is used, so the robot slows before the semicircle
rather than after entering it.

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --variant policy_nominal
```

`--policy auto` loads:

```text
racing/policies/checkpoints/ant_rapid
```

Omit `--policy-speed` to use the fastest certified envelope.  During debugging,
you can impose a ceiling such as:

```bash
--policy-speed 2.0
```

## 5. Native-joint SPG-MPPI

Once policy-only locomotion is stable, run direct-joint SPG-MPPI. SPG is now the default, so `--variant` can be omitted:

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --rollouts 128 \
  --horizon 15 \
  --joint-noise 0.05
```

For Ant, the sampled population is exactly:

```text
[128, 15, 8]
```

The 8 dimensions are Ant's native MuJoCo controls. The learned policy constructs `U_bar`; SPG computes `J_t = d p_xy(t+L) / d u_t` around that nominal, maps the 2-D prior covariance through `J_t^dagger`, and MPPI/LBPS evaluates the resulting joint-control population.

For first debugging use a smaller controller:

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --rollouts 32 \
  --horizon 8 \
  --joint-noise 0.03
```

## 6. New-environment adaptation

Change the physical MuJoCo plant without directly changing the MPPI planning
model:

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --rollouts 128 \
  --horizon 15 \
  --friction-scale 0.55 \
  --mass-scale 1.10 \
  --motor-scale 0.90
```

Then enable online model identification:

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --rollouts 128 \
  --horizon 15 \
  --friction-scale 0.55 \
  --mass-scale 1.10 \
  --motor-scale 0.90 \
  --adapt-model \
  --sysid-history 12 \
  --sysid-interval 8 \
  --save racing/results/ant_adaptive.npz
```

The online identifier only receives recent `(x_t, u_t, x_{t+1})` transitions;
it is not given the true plant multipliers.  Its estimate updates the separate
MuJoCo model used for subsequent MPPI rollouts.

## 7. Run the adaptation ablation

```bash
python -m racing.experiments.compare_adaptation \
  --robot ant \
  --policy auto \
  --rollouts 128 \
  --horizon 15 \
  --friction-scale 0.55 \
  --mass-scale 1.10 \
  --motor-scale 0.90
```

This compares:

1. policy only;
2. policy-seeded MPPI with a fixed nominal model;
3. policy-seeded MPPI with online model identification.

Results are written to `racing/results/adaptation_comparison.csv`.

## Humanoid

The same training/controller interfaces work with the classic MuJoCo Humanoid:

```bash
python -m racing.policies.train_velocity_policy \
  --robot humanoid \
  --impl warp \
  --num-envs 4096 \
  --steps 400000000 \
  --phase-steps 20000000 \
  --output racing/policies/checkpoints/humanoid_rapid
```

Then visualize or race with `--robot humanoid`.  Do not expect Ant hyperparameters
to be optimal for Humanoid; the point of the common interface is to make those
morphology-specific comparisons explicit.

## Recommended experiment order

```text
A. 1M Ant smoke training
B. full Ant high-speed training
C. visualize command tracking
D. policy-only fastest-feasible stadium race
E. policy + direct-joint SPG-MPPI
F. repeat E under low friction / mass / motor changes
G. add online MuJoCo system identification
H. use standard MPPI only as the ablation
```


## Replay a completed run

Every `racing.experiments.race` invocation now saves a replay recording by default to:

```text
racing/results/last_run.npz
```

The recording contains the full MuJoCo `qpos`, `qvel`, actuator state, controls, model-parameter perturbations, track transform, and state timestamps. Replay therefore does not rerun PPO, SPG, MPPI, or system identification; it simply plays back the exact recorded MuJoCo states.

```bash
python -m racing.experiments.replay
```

Replay another result:

```bash
python -m racing.experiments.replay \
  --file racing/results/ant_adaptive.npz
```

Useful playback options:

```bash
# Half speed
python -m racing.experiments.replay --speed 0.5

# 2x speed and loop
python -m racing.experiments.replay --speed 2 --loop

# Only replay simulation seconds 3 through 8
python -m racing.experiments.replay --start 3 --end 8

# Lower camera angle (closer to ground; -90 is top-down)
python -m racing.experiments.replay --camera-elevation -58

# Export the exact saved states to a 30 fps GIF instead of opening the viewer
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --gif racing/results/last_run.gif \
  --gif-fps 30 \
  --gif-width 960 \
  --gif-height 540

# Export just a time interval, at 2x playback speed
python -m racing.experiments.replay \
  --start 3 --end 8 --speed 2 \
  --gif racing/results/clip.gif
```

Replay now defaults to a lower-angle camera (`-64` degrees instead of the race
viewer's `-76` degree overview) while keeping the complete stadium in frame.
Use `--camera-elevation` and `--camera-distance-scale` to tune it.  GIF export
uses off-screen MuJoCo rendering of the recorded states and includes the same
visual track overlay; it does not rerun PPO, MPPI, or system identification.
Add `--viewer-ui` only when the normal MuJoCo side panels are wanted. Use
`--no-save` during racing if no replay recording should be written.

## SPG defaults

The default race command is now equivalent to:

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --variant sensitivity_projected_gaussian_prior_mppi \
  --rollouts 128 \
  --horizon 15 \
  --spg-lookahead 3 \
  --spg-mix 1.0 \
  --spg-null-std 0.15
```

`--spg-mix 1.0` means the informed task-space component plus Jacobian-null-space exploration is used without a full-space standard-MPPI mixture. `--joint-noise` still controls the null-space exploration scale. For an ablation, explicitly pass `--variant standard_mppi`.

## Default race visualization

The race and replay viewers now use the same fixed overview presentation by default:

- plain texture-free grey MuJoCo ground plane,
- dark blue/teal visual race surface,
- alternating red/white curbs,
- dashed yellow centerline and bright start/finish stripe,
- fixed race overview camera framed to contain the complete stadium track; replay defaults to a slightly lower viewing angle,
- MuJoCo side UI panels hidden unless `--viewer-ui` is passed.

The track ribbon is viewer-only geometry and does not change contact physics, SPG sensitivities, policy observations, or MPPI rollouts.
