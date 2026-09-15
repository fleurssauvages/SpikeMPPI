# Spike-MPPI: motoneuron-inspired MPPI

<p align="center"><strong>
<a href="#spike-mppi">Spike-MPPI</a> ·
<a href="#results">Results</a> ·
<a href="#overview">Overview</a> ·
<a href="#requirements">Requirements</a> ·
<a href="#racing">Racing</a> ·
<a href="#terrain-transfer">Terrain Transfer</a> ·
<a href="#task-transfer">Task Transfer</a> ·
<a href="#ant-geometry-transfer">Geometry Transfer</a> ·
<a href="#plant-model-mismatch">Model Mismatch</a> ·
<a href="#replay">Replay</a> ·
<a href="#references">References</a>
</strong></p>

This project studies **Ant locomotion with Model Predictive Path Integral control (MPPI)** in MuJoCo.

The main novelty of this repository is **Spike-MPPI**, a motoneuron-inspired MPPI sampling method. Instead of perturbing the nominal control sequence with independent Gaussian noise, Spike-MPPI samples sparse marked motor events, maps them through coordinated actuator synergies, and converts them into smooth control perturbations with causal twitch kernels. The proposal distribution itself adapts online from MPPI rollout weights by learning when a synergy should fire, which sign is useful, and which recruitment amplitudes are favored.

The same controller is evaluated across stadium racing, novel terrain, new tasks, morphology changes, and plant-model mismatch. The remaining MPPI samplers in this repository are retained primarily as controlled comparison baselines.

## Spike-MPPI

Spike-MPPI replaces direct Gaussian control noise with a **marked point-process proposal**. For rollout \(i\), horizon step \(t\), and motor synergy \(m\), positive and negative spike counts are sampled from learned intensities. Each event also carries a recruitment level.

## Results

### Same-task refinement

<table align="center">
  <tr>
    <th align="center">MPPI</th>
    <th align="center">Spike-MPPI</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/mppi.gif" width="460" alt="MPPI racing on the obstacle-free track">
    </td>
    <td align="center">
      <img src="assets/spike.gif" width="460" alt="Spike-MPPI racing on the obstacle-free track">
    </td>
  </tr>
</table>

### Adaptation to novel terrain

<table align="center">
  <tr>
    <th align="center">MPPI</th>
    <th align="center">Spike-MPPI</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/mppi_rocky.gif" width="460" alt="MPPI on rocky terrain">
    </td>
    <td align="center">
      <img src="assets/spike_rocky.gif" width="460" alt="Spike-MPPI on rocky terrain">
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <th align="center">MPPI</th>
    <th align="center">Spike-MPPI</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/mppi_mixed.gif" width="460" alt="MPPI on mixed terrain">
    </td>
    <td align="center">
      <img src="assets/spike_mixed.gif" width="460" alt="Spike-MPPI on mixed terrain">
    </td>
  </tr>
</table>

### Adaptation to new tasks

<table align="center">
  <tr>
    <th align="center">MPPI</th>
    <th align="center">Spike-MPPI</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/mppi_box.gif" width="460" alt="MPPI pushing a box">
    </td>
    <td align="center">
      <img src="assets/spike_box.gif" width="460" alt="Spike-MPPI pushing a box">
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <th align="center">MPPI</th>
    <th align="center">Spike-MPPI</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/mppi_sled.gif" width="460" alt="MPPI towing a sled">
    </td>
    <td align="center">
      <img src="assets/spike_sled.gif" width="460" alt="Spike-MPPI towing a sled">
    </td>
  </tr>
</table>

### Adaptation to modified Ant geometry

<table align="center">
  <tr>
    <th align="center">MPPI</th>
    <th align="center">Spike-MPPI</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/mppi_diagonal.gif" width="460" alt="MPPI with diagonal leg-length mismatch">
    </td>
    <td align="center">
      <img src="assets/spike_diagonal.gif" width="460" alt="Spike-MPPI with diagonal leg-length mismatch">
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <th align="center">MPPI</th>
    <th align="center">Spike-MPPI</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/mppi_sameside.gif" width="460" alt="MPPI with same-side leg-length mismatch">
    </td>
    <td align="center">
      <img src="assets/spike_sameside.gif" width="460" alt="Spike-MPPI with same-side leg-length mismatch">
    </td>
  </tr>
</table>


---

## Overview

The project supports one robot, **Ant**, and MPPI control:

| Controller variant | Description |
| --- | --- |
| `mppi` | Optimize a control sequence online with MPPI. Candidate generation is selected independently with `--sampling`. |

Spike-MPPI is the primary sampling contribution. Other samplers are included as baselines and ablations:

| Sampling option | Role | Description |
| --- | --- | --- |
| `spike` | **Main method** | Motoneuron-inspired marked spike events, multi-joint synergies, causal twitch decoding, and online adaptation of firing rate, sign and recruitment statistics. By default, it runs with online adaption of its parameters, use `--no-online`  to deactivate online adaptation |
| `standard` | Baseline | Fixed-scale direct-joint Gaussian MPPI sampling. |
| `guided` | Baseline | Low-rank history-guided sampling from recent successful MPPI update directions. |
| `diag-lowrank` | Baseline | Time/joint-dependent diagonal variance adaptation plus a history-guided low-rank component. |
| `spline` | Baseline | Smooth low-dimensional cubic B-spline perturbations. |
| `icem` | Baseline | Shifted elite control sequences from the previous update are reused in the next population. |

The main pipeline is:

1. Evaluate transfer to new terrain, new tasks, leg-length mismatch, and model-parameter mismatch.
2. Save exact MuJoCo states for deterministic replay and GIF export.

The MPPI controller acts directly on Ant's eight actuator controls.

---

## CPU MuJoCo rollout backend

Online MPPI planning is CPU-only. The controller automatically selects the fastest available MuJoCo rollout path in this order:

1. fused C++ MuJoCo physics + MPPI cost evaluation,

2. stock batched `mujoco.rollout`,

3. the legacy Python rollout loop as a compatibility fallback.

There is no rollout-backend command-line selector. When the fused extension is available it is preferred automatically.

The fused evaluator combines MuJoCo stepping with racing-cost evaluation and uses persistent MuJoCo data plus persistent worker threads to reduce Python overhead and temporary allocations. It changes the implementation of candidate evaluation, not the MPPI objective.

The active CPU path is printed at startup, for example:

```text
planner=cpu/fused/16t
```

Planner fidelity is selected independently with `--planner-mode`:

| Mode | Integrator | Planner timestep | Solver/contact profile |
| --- | --- | --- | --- |
| `rk4` | RK4 | source MuJoCo timestep | source settings |
| `fast-rk4` | RK4 | one step per control interval | fast profile |
| `implicitfast` | implicitfast | source MuJoCo timestep | fast profile |

For the default 20 ms controller and 10 ms source MuJoCo timestep, this means:

```text
rk4          -> 2 x 10 ms RK4 steps / control update
fast-rk4     -> 1 x 20 ms RK4 step  / control update
implicitfast -> 2 x 10 ms implicitfast steps / control update
```

The fast solver/contact profile caps solver iterations at 20, line-search iterations at 10, loosens tolerance to at least `1e-6`, and disables noslip iterations. `fast-rk4` is intended as a lower-cost RK4 planner while keeping the physical plant independent.

By default, the first fused candidate batch is verified against the reference rollout path before steady-state use. To disable the one-time verification after equivalence has already been established on a machine:

```bash
RACING_FUSED_VERIFY=0 python -m racing.experiments.race ...
```

---

## Requirements

### Platform

* Python **3.10+**.

* Git.

* MuJoCo **3.3+**.

* The fused C++ rollout evaluator currently targets Linux/macOS and requires a C++17 compiler.

* If the fused extension is unavailable, online MPPI falls back automatically to stock `mujoco.rollout` and then to the Python compatibility path.

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

### 2. Install project dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

### 3. Build the fused C++ rollout backend

This step is optional but recommended for real-time MPPI.

On Debian/Ubuntu:

```bash
sudo apt install build-essential
```

Build the extension in place:

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

---

## Racing

The main entry point is:

```bash
python -m racing.experiments.race
```

A run saves exact MuJoCo state history to `racing/results/last_run.npz` by default.

### Spike-MPPI

The main controller configuration is:

```bash
python -m racing.experiments.race \
  --robot ant \
  --variant mppi \
  --sampling spike \
  --rollouts 32 \
  --horizon 50 \
  --profile
```

`--sampling spike` keeps the nominal, MuJoCo rollout physics, task cost, LBPS temperature selection, actuator clipping and exponentially weighted MPPI control update unchanged. Only the stochastic proposal is replaced.

At each update, Spike-MPPI samples signed, marked events in a motor-synergy space. Causal twitch kernels transform those sparse events into smooth horizon-length actuator perturbations. MPPI rollout weights then adapt the firing-rate map, positive/negative preference and recruitment-level distribution online. The learned proposal is warm-started by shifting these statistics with the receding horizon.

Primary spike parameters:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--spike-rate-hz FLOAT` | `16.0` | Base event rate per synergy. With 8 synergies and the default 1 s horizon this gives about 128 expected events/rollout. |
| `--spike-rate-update FLOAT` | `0.02` | Online firing-rate adaptation rate. |
| `--spike-rate-prior FLOAT` | `0.5` | Prior strength pulling learned rates toward the base rate. |
| `--spike-recruitment-levels N` | `6` | Number of discrete recruitment amplitudes. |
| `--spike-twitch-rise SEC` | `0.016` | Twitch rise time. |
| `--spike-twitch-decay SEC` | `0.064` | Twitch decay time. |
| `--spike-twitch-duration SEC` | `0.200` | Causal twitch support. |
| `--spike-sign-update FLOAT` | `0.10` | Online adaptation rate of positive/negative event preference. |
| `--spike-sign-prior FLOAT` | `0.50` | Prior strength toward balanced event signs. |
| `--spike-sign-min-prob FLOAT` | `0.10` | Minimum probability assigned to either sign. |
| `--spike-mark-update FLOAT` | `0.10` | Online adaptation rate of the recruitment-level distribution. |
| `--spike-mark-prior FLOAT` | `0.50` | Prior strength on the recruitment-level distribution. |
| `--spike-mark-min-prob FLOAT` | `0.01` | Minimum probability assigned to any recruitment level. |

For the flat-ground HPO result, `--joint-noise 0.3` is a useful starting point; transfer experiments should retune or validate that value rather than assuming it is universal.

### Comparison MPPI samplers

The samplers below are retained for controlled comparisons. `--sampling` changes **only candidate generation inside MPPI**, so comparisons can keep the nominal, rollout count, horizon, task objective and physics fixed.

The non-standard comparison samplers are implementation-specific adaptations inspired by the cited methods; they are not exact reproductions of Guided ES, CMA-ES, Model Tensor Planning or iCEM.

#### Standard Gaussian sampling (`--sampling standard`)

`standard` perturbs each actuator trajectory directly with fixed-scale Gaussian MPPI noise. It is the reference baseline for measuring whether structured spike-based exploration improves sample efficiency, transfer behavior, or control quality.

#### Guided low-rank sampling (`--sampling guided`)

`guided` stores recent full-horizon MPPI update directions, shifts them with the receding horizon, and orthonormalizes the most recent directions into a low-rank basis. The next proposal mixes ordinary full-space MPPI noise with noise inside that history subspace. The expected proposal trace is renormalized so the method reallocates the standard exploration budget instead of simply adding more noise.

This is inspired by Guided Evolutionary Strategies, which elongates a random-search distribution along a low-dimensional guiding subspace [1]. Here the guiding vectors are previous MPPI update directions rather than external surrogate gradients.

```bash
python -m racing.experiments.race \
  --variant mppi \
  --sampling guided \
  --rollouts 32 \
  --horizon 50 \
  --joint-noise 0.5 \
  --guided-rank 6 \
  --guided-fraction 0.5 \
  --profile
```

Key parameters:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--guided-rank N` | `6` | Maximum number of recent update directions retained in the guiding subspace. |
| `--guided-fraction FLOAT` | `0.50` | Fraction of proposal energy assigned to the learned low-rank component before trace renormalization. |

#### Diagonal + low-rank sampling (`--sampling diag-lowrank`)

`diag-lowrank` extends `guided` with an adaptive normalized variance for every `(horizon step, actuator)` pair. The diagonal target is estimated from the MPPI-weighted applied perturbations. With a small rollout population, the update is deliberately conservative: low ESS shrinks the estimate back toward standard MPPI, the variance is clipped, smoothed with an EMA, renormalized to preserve the average exploration scale, and shifted with the warm start.

The diagonal covariance-adaptation idea is inspired by the covariance adaptation principles of CMA-ES [2], while the low-rank history component follows the guided-subspace idea in [1]. This controller does not implement the full CMA-ES algorithm.

```bash
python -m racing.experiments.race \
  --variant mppi \
  --sampling diag-lowrank \
  --rollouts 32 \
  --horizon 50 \
  --joint-noise 0.5 \
  --guided-rank 6 \
  --guided-fraction 0.5 \
  --diag-lowrank-rate 0.08 \
  --diag-lowrank-min 0.25 \
  --diag-lowrank-max 4.0 \
  --profile
```

Key parameters:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--diag-lowrank-rate FLOAT` | `0.08` | EMA rate for the time/joint variance update. |
| `--diag-lowrank-min FLOAT` | `0.25` | Minimum normalized variance factor before renormalization. |
| `--diag-lowrank-max FLOAT` | `4.0` | Maximum normalized variance factor before renormalization. |

#### B-spline latent sampling (`--sampling spline`)

`spline` samples a small set of cubic B-spline coefficients per actuator and expands them into the full horizon. With the default `H=50`, eight Ant actuators, and six spline modes, the stochastic proposal is parameterized by `6 x 8 = 48` latent coefficients rather than 400 independent time/joint values. The basis is row-normalized so `joint_noise` remains approximately comparable to the standard MPPI scale. Because the spline perturbation is already temporally smooth, the standard AR temporal-noise filter is not applied a second time.

This sampling option is inspired by structured spline control-trajectory sampling, including the B-spline trajectory parameterization used in Model Tensor Planning [3]. It implements only the low-dimensional B-spline proposal, not the tensor-sampling algorithm of that paper.

```bash
python -m racing.experiments.race \
  --variant mppi \
  --sampling spline \
  --rollouts 32 \
  --horizon 50 \
  --joint-noise 0.5 \
  --spline-modes 6 \
  --profile
```

| Option | Default | Meaning |
| --- | ---: | --- |
| `--spline-modes N` | `6` | Number of cubic B-spline latent modes per actuator; values below four are raised to four. |

#### iCEM-style elite reuse (`--sampling icem`)

`icem` retains the best finite-cost control sequences from the previous controller update. At the next MPC step those sequences are shifted by one horizon step and inserted into the new candidate population; the remaining candidates are sampled using the standard MPPI proposal. The controller still uses LBPS and the standard MPPI weighted update rather than a CEM elite-mean update.

This is inspired by the memory/sample-reuse mechanism in sample-efficient iCEM [4].

```bash
python -m racing.experiments.race \
  --variant mppi \
  --sampling icem \
  --rollouts 32 \
  --horizon 50 \
  --joint-noise 0.5 \
  --icem-elites 4 \
  --profile
```

| Option | Default | Meaning |
| --- | ---: | --- |
| `--icem-elites N` | `4` | Number of best previous control sequences shifted and reused in the next rollout population. |

For controlled sampling comparisons, keep `--variant mppi`, `--seed`, `--rollouts`, `--horizon`, `--joint-noise`, planner physics, plant physics, task and terrain fixed and change only `--sampling` plus sampling-specific parameters.

Warm-starting is enabled by default. Use:

```text
--no-warm-start
```

to disable it.

### MPPI planner physics modes

The default planner uses full RK4 with the source MuJoCo timestep and solver/contact settings:

```bash
python -m racing.experiments.race \
  --robot ant \
  --variant mppi \
  --rollouts 32 \
  --horizon 50 \
  --joint-noise 0.5 \
  --workers 16 \
  --planner-mode rk4 \
  --profile
```

For a cheaper RK4 planner, use:

```text
--planner-mode fast-rk4
```

`fast-rk4` keeps RK4 but sets the planner timestep equal to the control period, so the default 20 ms control interval requires one planner `mj_step` instead of two 10 ms steps. It also uses the fast solver/contact profile.

For the cheapest supported MuJoCo planner profile, use:

```text
--planner-mode implicitfast
```

`implicitfast` keeps the source planner timestep and uses the same fast solver/contact profile. The physical plant remains independent; use `--plant-integrator` only when intentionally changing the plant integrator.

At startup, the program prints both plant and planner stepping, for example:

```text
plant=model:2x0.01s  planner_mode=fast-rk4:1x0.02s
```

---

## Terrain transfer

The MuJoCo plant/planner terrain changes at test time.

```bash
python -m racing.experiments.race \
  --robot ant \
  --variant mppi \
  --terrain ramps
```

Available terrain modes:

```text
flat
ramps
stairs
rocky
mixed
```

Useful controls:

```text
--terrain-seed N
--terrain-scale FLOAT
```

---

## Task transfer

Additional flat-ground tasks are supported.

### Push a box

```bash
python -m racing.experiments.race \
  --robot ant \
  --variant mppi \
  --task push_box
```

The push task uses a **box only**.

Useful box parameters include:

```text
--box-distance FLOAT
--box-size FLOAT
--box-height FLOAT
--box-mass FLOAT
--box-friction FLOAT
--push-box-progress-weight FLOAT
--push-robot-progress-weight FLOAT
--push-approach-weight FLOAT
--push-box-max-lift FLOAT
--push-box-min-up FLOAT
```

### Tow a sled

```bash
python -m racing.experiments.race \
  --robot ant \
  --variant mppi \
  --task tow_sled
```

Useful sled parameters include:

```text
--sled-distance FLOAT
--sled-length FLOAT
--sled-width FLOAT
--sled-height FLOAT
--sled-mass FLOAT
--sled-friction FLOAT
--sled-rope-length FLOAT
--sled-progress-weight FLOAT
--sled-robot-progress-weight FLOAT
--sled-max-lift FLOAT
--sled-min-up FLOAT
```

`push_box` and `tow_sled` require flat terrain.

---

## Ant geometry transfer

Known leg-length mismatches can be applied to both the physical plant and planning model.

Same-side mismatch:

```bash
python -m racing.experiments.race \
  --robot ant \
  --variant mppi \
  --leg-mismatch same_side
```

Diagonal mismatch:

```bash
python -m racing.experiments.race \
  --robot ant \
  --variant mppi \
  --leg-mismatch diagonal
```

The short/long leg scales can be changed with:

```text
--short-leg-scale 0.75
--long-leg-scale 1.25
```

---

## Plant-model mismatch

The physical plant can be perturbed without giving those perturbations directly to the planning model. This is useful for evaluating model mismatch between the online MPPI planner and the simulated plant.

```bash
python -m racing.experiments.race \
  --robot ant \
  --variant mppi \
  --friction-scale 0.8 \
  --mass-scale 1.15 \
  --motor-scale 0.9 \
  --slope-deg 2.0
```

The available plant perturbations are:

```text
--friction-scale FLOAT
--mass-scale FLOAT
--motor-scale FLOAT
--slope-deg FLOAT
```

These parameters modify the physical plant only. The planning copy starts from the nominal model, so the run measures transfer under unobserved model mismatch.

---

## Empirical spatial prior

Without `--prior`, racing uses the geometric prior.

To load a saved empirical prior:

```text
--prior path/to/prior.npz
```

---

## Core racing options

| Option | Default | Description |
| --- | --- | --- |
| `--robot ant` | `ant` | The only supported racing robot. |
| `--laps N` | `1` | Requested laps. |
| `--variant mppi` | `mppi` | MPPI controller. |
| `--sampling {spike,standard,guided,diag-lowrank,spline,icem}` | `standard` | Candidate-generation strategy used by MPPI. `spike` is the primary method in this repository. |
| `--rollouts N` | `32` | MPPI candidate trajectories per update. |
| `--horizon N` | `50` | MPPI horizon in control steps. |
| `--dt SEC` | control dt | Control period. |
| `--lbps-delta FLOAT` | `0.95` | Adaptive-temperature target. |
| `--joint-noise FLOAT` | `0.5` | Actuator-range MPPI exploration scale. |
| `--plant-integrator {model,euler,implicitfast}` | `model` | Physical-plant integrator; `model` preserves the source XML setting. |
| `--planner-mode {rk4,fast-rk4,implicitfast}` | `rk4` | Planner physics profile. `fast-rk4` uses one RK4 step per control interval; `implicitfast` uses the source timestep. Both fast modes use the cheaper solver/contact profile. |
| `--task {run,push_box,tow_sled}` | `run` | Test-time task. |
| `--terrain {flat,ramps,stairs,rocky,mixed}` | `flat` | Test-time terrain. |
| `--leg-mismatch {none,same_side,diagonal}` | `none` | Known Ant leg-length mismatch. |
| `--profile` | off | Print compact controller timing statistics. |
| `--headless` | off | Disable viewer. |

For all options:

```bash
python -m racing.experiments.race --help
```

---

## Profiling real-time performance

Add:

```text
--profile
```

The per-update profiler prints only the active timing components: nominal construction, sampling, rollout evaluation, MPPI update, total latency, the control deadline, and an `OK`/`MISS` deadline status. `refine` and real-time-factor (`xRT`) fields are not printed.

Example steady-state output:

```text
MPPI [    2]  nominal    2.93 ms (warm 2.76, prior 0.16)  |  sample   0.45 ms  |  rollout    9.15 ms (fused 8.72)  |  update   0.49 ms  |  total   13.02 / 20.00 ms  [OK]
```

The first controller update is treated as warm-up: it remains visible in the live trace but is excluded from p50/p95 timing, deadline-miss statistics, and HPO latency scoring. Steps 2 onward determine the steady-state profile.

For a 50 Hz controller, a useful target is approximately:

```text
p95 total < 20 ms
deadline misses close to 0%
```

When comparing planner configurations, record at least:

* lap time or task progress,

* fall/off-track outcome,

* MPPI effective sample size,

* rollout and total p50/p95 latency,

* deadline misses.

For CPU rollout tuning, benchmark `--workers` and `--rollout-chunk-size` on the target machine. The fused evaluator is selected automatically when the extension is available.

---

## Replay

Replay restores exact saved MuJoCo states. It does **not** rerun MPPI.

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz
```

At 2x speed:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --speed 2
```

Replay an interval:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --start 2.0 \
  --end 8.0
```

---

## GIF export

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --gif racing/results/mppi.gif
```

Set output properties explicitly:

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --gif racing/results/mppi.gif \
  --gif-fps 50 \
  --gif-width 960 \
  --gif-height 540
```

---

## Typical end-to-end workflow

### 1. Build the fused evaluator

```bash
python racing/setup_native.py build_ext --inplace
```

### 2. Run Spike-MPPI

```bash
python -m racing.experiments.race \
  --robot ant \
  --variant mppi \
  --sampling spike \
  --rollouts 32 \
  --horizon 50 \
  --workers 16 \
  --planner-mode rk4 \
  --profile
```

### 3. Replay

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz
```

### 4. Export a GIF

```bash
python -m racing.experiments.replay \
  --file racing/results/last_run.npz \
  --gif racing/results/mppi.gif
```

---

## Notes on reproducibility and performance

* Use the same `--seed`, horizon, rollout count, `--joint-noise`, plant/planner physics settings, task, and terrain when comparing MPPI sampling options.

* The first controller update warms native/controller paths and can be much slower than steady state. It is printed for diagnostics but excluded from profiling and HPO latency statistics.

* MPPI warm start is enabled by default.

* `--planner-mode rk4` is the fidelity-oriented planner profile: explicit RK4, source planner timestep, and source solver/contact settings.

* `--planner-mode fast-rk4` keeps RK4 but uses one planner step per control interval plus the fast solver/contact profile. With the default 20 ms control period and 10 ms source timestep, this changes the planner from `2 x 10 ms` to `1 x 20 ms` while leaving the plant unchanged.

* `--planner-mode implicitfast` uses MuJoCo `implicitfast` at the source timestep with the same fast solver/contact profile.

* Plant and planner substep counts are computed independently. Changing planner mode does not change how far the physical plant advances per control update.

* Thread count and rollout chunk size are machine dependent; benchmark them on the target CPU.

* GIF export uses saved states, so rendering does not alter the racing result.

## References

The non-standard sampling options above are implementation-specific adaptations of the following ideas; the citations identify the main methodological inspiration rather than claiming exact reproduction.

[1] Maheswaranathan, N., Metz, L., Tucker, G., Choi, D., & Sohl-Dickstein, J. **Guided evolutionary strategies: augmenting random search with surrogate gradients**. Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97:4264-4273, 2019. https://proceedings.mlr.press/v97/maheswaranathan19a.html

[2] Hansen, N. **The CMA Evolution Strategy: A Tutorial**. arXiv:1604.00772, 2016. https://arxiv.org/abs/1604.00772

[3] Le, A. T., Nguyen, K., Vu, M. N., Carvalho, J., & Peters, J. **Model Tensor Planning**. Transactions on Machine Learning Research, 2025. https://arxiv.org/abs/2505.01059

[4] Pinneri, C., Sawant, S., Blaes, S., Achterhold, J., Stueckler, J., Rolinek, M., & Martius, G. **Sample-efficient Cross-Entropy Method for Real-time Planning**. Proceedings of the 2020 Conference on Robot Learning, PMLR 155:1049-1065, 2021. https://proceedings.mlr.press/v155/pinneri21a.html