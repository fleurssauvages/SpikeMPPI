# Real-time SPG-MPPI

The trained Ant policy uses `control_dt = 0.02 s`, so the online controller has a **20 ms deadline (50 Hz)**. The goal is therefore not just high aggregate simulation throughput: the complete `controller.step()` should have low p95 latency below 20 ms.

## Optimizations in this version

1. **Native batched MuJoCo rollouts**
   - Candidate MPPI trajectories are evaluated with `mujoco.rollout.Rollout` instead of a Python loop over rollouts.
   - The native C++ thread pool is persistent across controller ticks.
   - `--workers 0` chooses an automatic CPU thread count; use `--workers N` to benchmark explicitly.
   - The old Python evaluator remains available with `--rollout-backend python`.

2. **Batched SPG finite differences**
   - All control perturbations used to estimate the joint-to-XY Jacobians are sent to the same native rollout backend in batches.
   - This removes the original Python loop that launched a separate MuJoCo rollout for every `(horizon step, actuator)` perturbation.

3. **Fewer Python-to-MuJoCo calls**
   - Repeated physics substeps now use `mujoco.mj_step(..., nstep=substeps)`.

4. **Vectorized rollout cost / track projection**
   - Root XY, height and upright values are extracted from the returned state tensor.
   - The complete `[num_rollouts * horizon]` XY array is projected onto the stadium in one vectorized call.
   - The scalar stadium path used by the PPO nominal was also rewritten to avoid repeated transforms and temporary NumPy allocations.

5. **Lower allocation overhead**
   - Control scales/bounds are cached on hot paths.
   - The rapid-policy command envelope is pre-sorted instead of filtering/sorting curriculum cells for every curvature query.
   - Geometric/empirical prior covariance construction is vectorized.

6. **SPG sampling optimization for larger robots**
   - For larger actuator counts, null-space noise applies `(I - J^dagger J)z` in factorized form rather than a dense `nu x nu` multiply.
   - Ant (`nu=8`) intentionally keeps the dense path because it is faster at that small dimension.

7. **Real-time profiler**
   - `--profile` reports policy, SPG Jacobian, prior, sampling, candidate rollout, update and total controller latency.
   - The summary excludes the first five warm-up steps and prints p50/p95 plus deadline-miss percentage.

## Benchmark the real-time path

From the repository root:

```bash
python -m racing.experiments.race \
  --robot ant \
  --policy auto \
  --headless \
  --no-save \
  --max-steps 300 \
  --rollout-backend native \
  --workers 0 \
  --profile
```

The important final line is similar to:

```text
MPPI profile (warm-up excluded): ... total=p50 .../p95 ...ms
deadline=20.00ms  misses=...%
```

Target:

- `total p95 < 20 ms`
- `deadline misses` close to `0%`
- final race `xRT >= 1.0`

Do not judge the first few controller steps: JAX policy compilation/warm-up can make them outliers.

## Find the best CPU thread count

Automatic thread count is a starting point, not a guarantee. Benchmark the same race with:

```text
--workers 1
--workers 2
--workers 4
--workers 8
--workers 16
--workers <logical-core-count>
```

Choose the configuration with the lowest **p95 total**, not merely the best average. On hybrid CPUs it can also be worth pinning the process to performance cores.

For a baseline comparison, run the original-style evaluator:

```bash
python -m racing.experiments.race \
  --robot ant --policy auto --headless --no-save --max-steps 300 \
  --rollout-backend python --workers 0 --profile
```

## What to optimize next based on the profile

### If `rollouts` dominates

First tune `--workers`. If the native backend is still above budget:

1. Reduce `--rollouts` from 128 to 96 or 64 and verify racing quality.
2. Reduce `--horizon` from 15 to 12 or 10 on straights.
3. Make rollout count/horizon adaptive: use more samples in corners / low-ESS situations and fewer on easy straights.
4. If you eventually need hundreds or thousands of simultaneous rollouts, benchmark MJWarp/MJX with the *entire* sample-cost-update path on device. Avoid a CPU<->GPU transfer every 20 ms.

### If `spg_jac` dominates

The exact implementation recomputes all SPG sensitivities every 20 ms. The next high-value approximation is temporal reuse:

- shift the previous Jacobian forward one horizon step;
- recompute the full Jacobian every 2-4 controller ticks;
- optionally refresh only the first few horizon entries on intermediate ticks.

This can cut the sensitivity workload substantially, but it is no longer mathematically identical to the current controller, so validate lap time/off-track rate.

### If `policy` dominates

The current nominal performs a closed-loop PPO inference at every horizon step. Two strong options are:

1. **Shift/warm-start the previous optimized control sequence** and ask PPO only for the new tail control. This changes the nominal but is a very standard receding-horizon optimization.
2. Export the deterministic PPO MLP to a low-overhead CPU inference path (e.g. a small NumPy/BLAS or compiled implementation) if JAX launch/device-transfer overhead is measurable.

Avoid moving only the PPO network to GPU while MuJoCo remains on CPU; repeated host/device synchronization can erase the gain.

### If `sampling` or `update` dominates

That would be unusual at `128 x 15 x 8`. These parts are already NumPy-vectorized. Only then consider Numba for a remaining pure-Python kernel.

## Why Numba is not the first optimization

Numba cannot remove the dominant Python/MuJoCo boundary by JIT-compiling calls into MuJoCo's Python bindings. The original hot path was thousands of physics calls and per-rollout Python bookkeeping. Native `mujoco.rollout` attacks that directly. Numba is still useful later if profiling identifies a custom pure-Python cost, geometry, or sampling kernel that remains significant.

## More aggressive real-time options

If the exact controller still cannot meet 20 ms after CPU tuning:

- **25 Hz planner / 50 Hz actuator loop:** recompute MPPI every other control tick and execute the shifted plan between updates.
- **Deadline-aware MPPI:** dynamically shrink `N`/`H` when recent p95 approaches 20 ms.
- **Two-stage sampling:** short/coarse evaluation of all candidates, then extend only the best fraction.
- **C++ planner extension:** keep state extraction, rollout, cost and reduction in one native call, following the architecture used by MuJoCo MPC.
- **Solver benchmark:** use MuJoCo's performance tools to compare solver/iteration settings for this exact model before changing them; contact fidelity can change racing behavior.

## Online system identification

`--adapt-model` adds periodic work outside the controller timing breakdown. If hard real-time is required, use the final race `xRT` as well as the MPPI timing summary. If system identification causes spikes, move it to a slower update rate or a separate model copy rather than sharing mutable planner state across threads.
