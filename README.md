# Racing: Rapid-Locomotion policy + direct-joint SPG-MPPI + online MuJoCo adaptation

This project uses the MuJoCo locomotion robots, with **Ant** as the primary 2-D stadium racer. The default policy trainer is an adaptation of Margolis et al., *Rapid Locomotion via Reinforcement Learning* (RSS 2022 / IJRR), and an MPPI controller refines the learned policy online for racing.

The trainer keeps the morphology-independent ideas from Rapid Locomotion—command-conditioned locomotion, PPO, a joint forward-velocity/yaw-rate curriculum, domain randomization and push disturbances—but adapts them to the classic MuJoCo Ant/Humanoid models and **direct native MuJoCo actuator controls** rather than the Mini Cheetah PD joint-position interface.

- Paper: https://doi.org/10.1177/02783649231224053
- Released reference code: https://github.com/Improbable-AI/rapid-locomotion-rl

<p align="center"><b>Nominal policy (19.38 s)</b></p>

<p align="center">
  <img src="racing/results/nominal.gif" alt="Nominal policy" width="100%">
</p>

<p align="center"><b>Standard MPPI (17.06 s)</b></p>

<p align="center">
  <img src="racing/results/mppi.gif" alt="Standard MPPI" width="100%">
</p>

<p align="center"><b>Sensitivity Projected Gaussian MPPI (10.52 s)</b></p>

<p align="center">
  <img src="racing/results/spg_3.gif" alt="Sensitivity Projected Gaussian MPPI" width="100%">
</p>

## Overview

The pipeline has four main stages:

1. **Train a velocity-conditioned locomotion policy** with PPO and a Grid Adaptive Curriculum.
2. **Use the learned policy as the nominal controller** for a stadium racing task.
3. **Refine the nominal online with MPPI**, optionally using Sensitivity Projected Gaussian (SPG) exploration and online model adaptation.
4. **Save and replay exact MuJoCo states**, either interactively or as a GIF.

The racing code supports three controller variants:

- `policy_nominal`: execute the policy-derived nominal without MPPI refinement.
- `standard_mppi`: standard direct-joint MPPI around the policy nominal.
- `sensitivity_projected_gaussian_prior_mppi`: SPG-MPPI; this is the main racing controller.

For low-latency racing, the project also includes an optional **fused C++ MuJoCo rollout evaluator**. It preserves the MuJoCo model, integrator, physics timestep, control period, substeps, MPPI horizon and rollout count, but evaluates candidate trajectories and their racing cost directly in C++ instead of materializing every full MuJoCo state back into Python.

---

## Requirements

### Platform

- Python **3.10+**.
- Git, because `requirements.txt` installs the current Brax source directly from GitHub.
- MuJoCo **3.3+**.
- A GPU is strongly recommended for PPO training.
- The fused C++ rollout evaluator currently builds on **Linux and macOS** and requires a C++17 compiler.
- The standard `native` and `python` rollout backends do not require compiling the fused extension.

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

### 2. Install JAX for your accelerator

JAX installation is hardware-specific. Install the appropriate JAX build **before** the rest of the requirements.

CPU:

```bash
python -m pip install -U jax
```

NVIDIA GPU using CUDA 13 wheels:

```bash
python -m pip install -U "jax[cuda13]"
```

NVIDIA GPU using CUDA 12 wheels:

```bash
python -m pip install -U "jax[cuda12]"
```

See the current JAX installation guide for other platforms and locally installed CUDA/ROCm configurations:

https://docs.jax.dev/en/latest/installation.html

### 3. Install the project dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

The training stack uses MuJoCo MJX / MuJoCo Warp and MuJoCo Playground. The current MuJoCo packages document `mujoco-mjx[warp]` as the Warp-enabled MJX install. MuJoCo Playground is distributed as the `playground` package but is imported as `mujoco_playground`.

Useful upstream references:

- MuJoCo MJX: https://mujoco.readthedocs.io/en/latest/mjx.html
- MuJoCo Warp: https://mujoco.readthedocs.io/en/latest/mjwarp/index.html
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground

Verify the environment:

```bash
python - <<'PY'
import jax
import mujoco
from mujoco import mjx
import mujoco_playground

print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())
print("MuJoCo:", mujoco.__version__)
print("MJX import: OK")
print("MuJoCo Playground import: OK")
PY
```

For GPU training, `jax.default_backend()` should normally report `gpu`.

### 4. Build the fused C++ rollout backend

This step is optional but recommended for real-time MPPI.

On Debian/Ubuntu, make sure a C++ compiler is installed:

```bash
sudo apt install build-essential
```

Then build the extension in-place:

```bash
python racing/setup_native.py build_ext --inplace
```

Verify it:

```bash
python - <<'PY'
from racing import _fused_mujoco
print("Fused MuJoCo extension:", _fused_mujoco.__file__)
PY
```

The build script links against the MuJoCo shared library and headers shipped with the currently installed Python `mujoco` package.

---

## Training the Rapid-Locomotion policy

The main trainer is:

```bash
python -m racing.policies.train_velocity_policy
```

The default output directory for Ant is:

```text
racing/policies/checkpoints/ant_rapid
```

This is also the checkpoint automatically loaded by racing with `--policy auto`.

### Default paper-style training

A typical Ant training run is:

```bash
python -m racing.policies.train_velocity_policy \
  --robot ant \
  --impl warp \
  --num-envs 4096 \
  --steps 400000000 \
  --phase-steps 20000000 \
  --contacts-per-env 8 \
  --njmax 512 \
  --output racing/policies/checkpoints/ant_rapid
```

The default controller period is 0.02 s / 50 Hz. The trainer starts the curriculum around `v_x = +/-1 m/s` and yaw rate `+/-1 rad/s`, with 0.5-unit grid spacing, and updates the shared curriculum between PPO phases using native-MuJoCo frontier evaluation.

The PPO defaults implemented in the project are:

| Setting | Default |
| --- | ---: |
| Environments | 4096 |
| Total training steps | 400,000,000 |
| Curriculum phase | 20,000,000 steps |
| Discount | 0.99 |
| GAE lambda | 0.95 |
| PPO rollout length | 21 |
| PPO epochs per rollout | 5 |
| Minibatches | 4 |
| Batch size | 1024 |
| Entropy cost | 0.01 |
| PPO clip epsilon | 0.2 |
| Learning rate | 1e-3 |
| Max gradient norm | 1.0 |
| Policy network | 512, 256, 128 / ELU |
| Value network | 512, 256, 128 / ELU |
| Initial action noise std | 1.0 |

### Continue the curriculum until speed stops improving

To keep extending the forward-speed curriculum beyond the original `+6 m/s` grid edge, use `--until-failure`:

```bash
python -m racing.policies.train_velocity_policy \
  --robot ant \
  --impl warp \
  --num-envs 4096 \
  --phase-steps 20000000 \
  --contacts-per-env 8 \
  --njmax 512 \
  --output racing/policies/checkpoints/ant_rapid \
  --resume \
  --until-failure \
  --stall-patience 4 \
  --max-phases 100
```

In this mode the forward edge is extended by the same 0.5 m/s curriculum spacing whenever the straight outer edge is certified. Training stops after the configured number of PPO phases without a higher certified straight speed, or when a safety guard is reached.

Optionally set an absolute safety ceiling:

```bash
--max-forward-speed 10.0
```

When `--until-failure` is active, `--steps` is not the overall stopping criterion; stopping is controlled by speed progress, `--stall-patience`, `--max-phases`, and optionally `--max-forward-speed`.

### Resume a fixed-budget run

```bash
python -m racing.policies.train_velocity_policy \
  --robot ant \
  --output racing/policies/checkpoints/ant_rapid \
  --resume
```

A checkpoint directory contains at least:

```text
params.pkl
metadata.json
curriculum.json
brax_checkpoints/
frontier_phase_XXX.json
```

### Policy-training options

| Option | Default | Description |
| --- | --- | --- |
| `--robot {ant,humanoid}` | `ant` | Robot to train. |
| `--output PATH` | `racing/policies/checkpoints/<robot>_rapid` | Output checkpoint directory. |
| `--impl {warp,jax}` | `warp` | MJX implementation used by the training environment. |
| `--num-envs N` | `4096` | Number of parallel training environments. |
| `--steps N` | `400000000` | Fixed total training budget. |
| `--phase-steps N` | `20000000` | PPO steps per curriculum phase. |
| `--episode-length N` | `1000` | Episode length in control steps. |
| `--seed N` | `1` | Random seed. |
| `--ctrl-dt SEC` | policy default (`0.02` for Ant/Humanoid) | Override policy/control period. |
| `--contacts-per-env N` | `8` | MuJoCo-Warp broadphase/contact capacity per parallel world. |
| `--njmax N` | `512` | MuJoCo-Warp constraint capacity per world. |
| `--no-domain-randomization` | off | Disable friction/motor domain randomization. |
| `--no-pushes` | off | Disable periodic push disturbances. |
| `--resume` | off | Resume `params.pkl`, metadata and curriculum from the output directory. |
| `--frontier-eval-seconds SEC` | `4.0` | Native-MuJoCo evaluation duration for each frontier command. |
| `--until-failure` | off | Keep adding curriculum phases and forward-speed rows until straight-speed progress stalls. |
| `--stall-patience N` | `4` | Consecutive PPO phases without a higher certified straight speed before stopping. |
| `--max-phases N` | `100` | Safety cap for `--until-failure`. |
| `--max-forward-speed MPS` | none | Optional absolute forward curriculum ceiling. |

### Visualize/test a trained velocity policy

```bash
python -m racing.policies.test_velocity_policy \
  --robot ant \
  --policy racing/policies/checkpoints/ant_rapid
```

Useful test options:

```text
--segment-seconds 3.0
--max-speed <m/s>
--headless
```

---

## Racing / MPPI

The racing entry point is:

```bash
python -m racing.experiments.race
```

A run saves exact MuJoCo state history to `racing/results/last_run.npz` by default. This file can be replayed later without rerunning the controller.

### Nominal learned policy

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --variant policy_nominal
```

### Standard MPPI

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --variant standard_mppi \
  --rollouts 32 \
  --horizon 50 \
  --joint-noise 0.5 \
  --warm-start
```

### SPG-MPPI

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --variant sensitivity_projected_gaussian_prior_mppi \
  --rollouts 32 \
  --horizon 50 \
  --joint-noise 0.5 \
  --warm-start
```

### Real-time fused SPG-MPPI

After building `racing._fused_mujoco`, use:

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --variant sensitivity_projected_gaussian_prior_mppi \
  --rollouts 32 \
  --horizon 50 \
  --joint-noise 0.5 \
  --warm-start \
  --workers 8 \
  --rollout-chunk-size 1 \
  --planner-integrator model \
  --spg-refresh 0 \
  --disable-gc \
  --rollout-backend fused \
  --profile
```

On a Linux machine with 8 physical cores / 16 SMT threads, pinning to one hardware thread per physical core may reduce jitter:

```bash
taskset -c 0-7 python -m racing.experiments.race ...
```

Do not copy that CPU mask blindly: inspect your topology first with:

```bash
lscpu -e=CPU,CORE,SOCKET,MAXMHZ
```

For latency benchmarking, a performance-oriented CPU governor/EPP can also help.

### Fused-backend verification

By default, the first fused candidate batch is checked against the stock `mujoco.rollout` path. After equivalence has been established on a machine, disable the one-time check with:

```bash
RACING_FUSED_VERIFY=0 python -m racing.experiments.race ... --rollout-backend fused
```

The fused backend changes the implementation of candidate evaluation, not the simulated dynamics. With `--planner-integrator model`, it uses the model's original integrator and timestep.

### Online model adaptation

The physical/plant model can be perturbed without telling the planner the true perturbation:

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --variant sensitivity_projected_gaussian_prior_mppi \
  --friction-scale 0.8 \
  --mass-scale 1.15 \
  --motor-scale 0.9 \
  --slope-deg 2.0 \
  --adapt-model
```

When `--adapt-model` is enabled, recent plant transitions update the planning model. If the fused backend is active, the C++ evaluator reloads the updated planning model automatically.

### Empirical spatial prior

Without `--prior`, racing uses the geometric prior. To use a saved empirical prior:

```bash
--prior path/to/prior.npz
```

### Racing / MPPI options

| Option | Default | Description |
| --- | --- | --- |
| `--robot NAME` | `ant` | Stadium robot; current main targets are Ant and Humanoid. |
| `--policy SPEC` | `auto` | `auto` loads `racing/policies/checkpoints/<robot>_rapid`; also accepts a checkpoint directory, `neutral`, or `module:function`. |
| `--policy-speed MPS` | none | Optional cap on the learned maximum racing speed; otherwise use the learned curriculum envelope. |
| `--prior PATH` | geometric prior | Load an empirical `.npz` spatial prior. |
| `--laps N` | `1` | Requested number of laps. |
| `--rollouts N` | `128` | MPPI candidate trajectories per control update. |
| `--horizon N` | `15` | MPPI horizon in control steps. |
| `--dt SEC` | trained policy dt | MPPI/policy control period. |
| `--lbps-delta FLOAT` | `0.9` | LBPS adaptive-temperature target parameter. |
| `--nominal-refine-iters N` | `0` | Optional policy-nominal refinement iterations. |
| `--spg-lookahead N` | `3` | Future-position lookahead used for SPG sensitivity. |
| `--spg-mix FLOAT` | `1.0` | `1.0` = pure SPG task/null-space proposal; lower values blend standard joint noise. |
| `--spg-null-std FLOAT` | `0.15` | Uninformed exploration scale restricted to the Jacobian null space. |
| `--spg-damping FLOAT` | `1e-6` | Pseudoinverse damping for `J^T (J J^T + lambda I)^-1`. |
| `--spg-epsilon FLOAT` | `1e-3` | Finite-difference fraction of actuator range for SPG sensitivity. |
| `--joint-noise FLOAT` | `0.08` | Actuator-range noise scale; for SPG this controls null-space/default exploration. |
| `--variant {policy_nominal,standard_mppi,sensitivity_projected_gaussian_prior_mppi}` | SPG-MPPI | Controller variant. |
| `--seed N` | `1` | Controller random seed. |
| `--workers N` | `0` | Native rollout threads; `0` uses all logical CPU cores automatically. |
| `--rollout-chunk-size N` | `0` | Native/fused thread-pool chunk size; `0` selects automatic behavior. |
| `--rollout-backend {fused,native,python}` | `native` | Candidate-evaluation backend. `fused` requires the C++ extension. |
| `--warm-start` | off | Shift the previous optimized MPPI sequence instead of rebuilding `H` policy actions every tick. |
| `--spg-refresh N` | `1` | Full SPG Jacobian refresh interval: `1` every tick, `0` initial only, `N>1` every N ticks. |
| `--spg-refresh-prefix N` | `0` | When reusing a shifted SPG Jacobian, freshly finite-difference this many leading horizon steps. |
| `--planner-integrator {model,euler,implicitfast}` | `model` | Planning-copy integrator. `model` preserves the XML/model integrator. The physical plant always keeps its own model integrator. |
| `--profile` | off | Print MPPI timing breakdown for early steps and every 50 updates. |
| `--disable-gc` | off | Disable Python cyclic GC during the race loop to reduce latency jitter. |
| `--max-steps N` | none | Optional maximum number of control steps. |
| `--headless` | off | Disable the MuJoCo viewer. |
| `--viewer-ui` | off | Show the MuJoCo left/right UI panels. |
| `--controller-overlay` | off | Draw controller diagnostics in the viewer. |
| `--friction-scale FLOAT` | `1.0` | Scale plant friction. |
| `--mass-scale FLOAT` | `1.0` | Scale plant mass. |
| `--motor-scale FLOAT` | `1.0` | Scale plant actuator strength. |
| `--slope-deg FLOAT` | `0.0` | Add plant ground slope in degrees. |
| `--adapt-model` | off | Enable online system identification of the SPG-MPPI planning model. |
| `--sysid-history N` | `12` | Number of recent transitions used by online system identification. |
| `--sysid-interval N` | `8` | Control-step interval between system-identification updates. |
| `--sysid-estimate-slope` | off | Also estimate slope during online system identification. |
| `--save PATH` | `racing/results/last_run.npz` | Save metrics and complete MuJoCo states for replay. |
| `--no-save` | off | Disable replay-file saving. |

### Profiling real-time performance

Add:

```bash
--profile
```

The profiler reports the main controller stages, including policy/warm-start, SPG sensitivity, sampling, candidate rollout, update, p50/p95 latency, the 20 ms deadline and deadline misses.

For a 50 Hz controller, the target is not just mean real time. Prefer a configuration with:

```text
p95 total < 20 ms
deadline misses close to 0%
```

---

## Replay

The replay command restores the exact saved MuJoCo states. It does **not** rerun PPO or MPPI.

Basic replay:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz
```

Replay at 2x speed:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --speed 2
```

Replay only a time interval:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --start 2.0 \
  --end 8.0
```

Use a lower camera:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --camera-elevation -15 \
  --camera-distance-scale 0.60
```

`--camera-elevation -90` is top-down; values closer to zero put the camera closer to the ground.

### Replay options

| Option | Default | Description |
| --- | --- | --- |
| `--file PATH` | `racing/results/last_run.npz` | Recording generated by `racing.experiments.race`. |
| `--speed FLOAT` | `1.0` | Playback speed multiplier. |
| `--loop` | off | Repeat until the viewer is closed. |
| `--viewer-ui` | off | Show MuJoCo side panels. |
| `--start SEC` | `0.0` | Start time in recorded simulation seconds. |
| `--end SEC` | end of recording | Optional end time. |
| `--pause-at-end SEC` | `0.25` | Pause after a non-looping playback. |
| `--camera-elevation DEG` | `-20.0` | Free-camera elevation; `-90` is top-down and values closer to `0` are lower. |
| `--camera-distance-scale FLOAT` | `0.60` | Multiplier on the automatically computed full-track camera distance. |
| `--gif PATH` | none | Export a GIF instead of opening the viewer. |
| `--gif-fps FPS` | recorded control frequency | GIF output frame rate. By default this is `1 / control_dt`. |
| `--gif-width PX` | `960` | GIF width. |
| `--gif-height PX` | `540` | GIF height. |

---

## GIF export

Export the latest race:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --gif racing/results/spg.gif
```

The default GIF frame rate is the control frequency recorded in the replay. With the default 0.02 s controller period, this is 50 FPS.

Change playback speed with `--speed`:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --gif racing/results/spg.gif \
  --speed 2
```

Change GIF temporal resolution explicitly:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --gif racing/results/spg.gif \
  --gif-fps 50 \
  --gif-width 960 \
  --gif-height 540
```

`--gif-fps` controls the encoded/output frame rate. `--speed` controls how quickly recorded simulation time is played back.

The exporter uses the recorded state timeline, resamples it at the requested output rate, renders the stadium overlay off-screen, and writes the GIF with explicit frame timing.

---

## Typical end-to-end workflow

### 1. Train

```bash
python -m racing.policies.train_velocity_policy \
  --robot ant \
  --impl warp \
  --num-envs 4096 \
  --phase-steps 20000000 \
  --contacts-per-env 8 \
  --output racing/policies/checkpoints/ant_rapid \
  --until-failure \
  --stall-patience 4 \
  --max-phases 100
```

### 2. Build the fused evaluator

```bash
python racing/setup_native.py build_ext --inplace
```

### 3. Race with SPG-MPPI

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --variant sensitivity_projected_gaussian_prior_mppi \
  --rollouts 64 \
  --horizon 30 \
  --joint-noise 0.5 \
  --warm-start \
  --workers 8 \
  --rollout-chunk-size 1 \
  --planner-integrator model \
  --spg-refresh 0 \
  --disable-gc \
  --rollout-backend fused \
  --profile
```

### 4. Replay

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz
```

### 5. Save a GIF

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --gif racing/results/spg.gif
```

---

## Notes on reproducibility and performance

- Use `--seed` when comparing controller settings.
- The first JAX policy call includes JIT compilation and is much slower than steady-state controller updates.
- `--warm-start` removes repeated sequential PPO nominal generation after initialization by shifting the previous optimized control sequence.
- `--rollout-backend fused` is intended for low-latency candidate evaluation. `native` uses MuJoCo's stock batched rollout implementation; `python` is the legacy fallback.
- Thread count and chunk size are CPU-dependent. More logical threads are not necessarily faster than one worker per physical core.
- `--planner-integrator model` is the fidelity-preserving choice. `euler` and `implicitfast` are available as planning-only approximations; the plant itself is not changed by this option.
- `--spg-refresh 1` recomputes the full SPG Jacobian every tick. `--spg-refresh 0` computes it initially and then reuses the receding-horizon Jacobian, with the tail refreshed by the implementation. Use the setting that is stable for your experiment.
- GIF export uses saved states, so replay rendering does not change the racing result.

## Reference

Margolis, G. B., Yang, G., Paigwar, K., Chen, T., & Agrawal, P. *Rapid Locomotion via Reinforcement Learning*. International Journal of Robotics Research. https://doi.org/10.1177/02783649231224053