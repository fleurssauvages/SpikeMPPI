from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
import time
from typing import Sequence
import numpy as np

from .fast_kernels import NUMBA_AVAILABLE, stadium_rollout_cost_from_states



@dataclass
class RolloutCostConfig:
    hard_collision_clearance: float = 0.02
    fall_height_fraction: float = 0.45
    min_root_up: float = 0.15
    upright_weight: float = 0.05
    control_deviation_weight: float = 1e-4




class NativeRolloutBatcher:
    """Fast batched open-loop rollouts using MuJoCo's native C++ rollout module.

    Besides parallel MPPI evaluation, this class also owns the hot-path state
    buffers used by warm-start propagation and SPG finite differences.  Keeping
    those operations in the same native rollout backend avoids Python ``mj_step``
    loops and repeated state packing.
    """

    def __init__(self, robot, *, workers: int = 0, batch_hint: int = 128, chunk_size: int = 0, fused: bool = False) -> None:
        from mujoco import rollout as mj_rollout
        import inspect

        self.robot = robot
        self.mj = robot.mujoco
        self.model = robot.model
        cpu = max(1, int(os.cpu_count() or 1))
        requested = int(workers)
        if requested <= 0:
            requested = min(cpu, max(1, int(batch_hint)))
        self.nthread = max(1, requested)
        self.chunk_size = max(0, int(chunk_size))
        runner_threads = self.nthread if self.nthread > 1 else 0
        self.runner = mj_rollout.Rollout(nthread=runner_threads)
        self.data = (
            [self.mj.MjData(self.model) for _ in range(self.nthread)]
            if self.nthread > 1
            else self.mj.MjData(self.model)
        )
        self.state_spec = self.mj.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = int(self.mj.mj_stateSize(self.model, self.state_spec))
        self._state_scratch = self.mj.MjData(self.model)
        self._state_pack = np.empty(self.nstate, dtype=np.float64)
        self._state_buffers: dict[tuple[int, int], np.ndarray] = {}
        self._sensor_buffers: dict[tuple[int, int], np.ndarray] = {}
        self._model_batches: dict[int, list] = {}
        self._root_qadr = self._find_world_free_root_qadr()
        self.supports_vectorized_cost = self._root_qadr is not None
        self._ctrl_scale = np.maximum(robot.control_scale(), 1e-6)
        self._fd_scale = np.maximum(robot.control_scale(fraction=1.0), 1e-6)
        self._ctrl_low, self._ctrl_high = robot.control_bounds()
        self._cost_buffers: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.last_rollout_physics_ms = 0.0
        self.last_rollout_cost_ms = 0.0
        self.last_rollout_fused_ms = 0.0
        self.fused_evaluator = None
        self.fused_requested = bool(fused)
        self._fused_verified = False
        self._verify_fused = os.environ.get('RACING_FUSED_VERIFY', '1').strip().lower() not in {'0', 'false', 'no'}

        # ``skip_checks`` and caller-provided output arrays are supported by
        # current MuJoCo rollout wrappers.  Feature-detect so mujoco>=3.3
        # installations remain compatible if their wrapper signature differs.
        try:
            params = inspect.signature(self.runner.rollout).parameters
        except (TypeError, ValueError):
            params = {}
        self._supports_skip_checks = 'skip_checks' in params
        self._supports_state_output = 'state' in params
        self._supports_chunk_size = 'chunk_size' in params

        if self.fused_requested:
            self._init_fused_evaluator()

    def _init_fused_evaluator(self) -> None:
        if self._root_qadr is None:
            raise RuntimeError('fused rollout requires a world free-joint root')
        try:
            from racing import _fused_mujoco
        except ImportError as exc:
            raise RuntimeError(
                'fused rollout extension is not built. Run: '
                'python racing/setup_native.py build_ext --inplace '
                '(or python setup_native.py build_ext --inplace from inside racing/)'
            ) from exc
        import tempfile
        from pathlib import Path

        # Loading a temporary MJB keeps this extension independent of private
        # Python-binding pointer wrappers. mj_loadModel deep-loads the model, so
        # the file can be removed immediately after construction.
        with tempfile.NamedTemporaryFile(suffix='.mjb', delete=False) as tmp:
            model_path = tmp.name
        try:
            self.mj.mj_saveModel(self.model, model_path)
            chunk = self.chunk_size if self.chunk_size > 0 else 1
            self.fused_evaluator = _fused_mujoco.FusedRolloutEvaluator(
                model_path, int(self.nthread), int(self._root_qadr), int(chunk)
            )
        finally:
            Path(model_path).unlink(missing_ok=True)

    def sync_fused_model(self) -> None:
        """Reload the fused model after explicit online model-parameter updates."""
        if not self.fused_requested:
            return
        # Destroying the old object joins its persistent workers before loading
        # the newly serialized planning model. This is outside the MPPI hot path.
        self.fused_evaluator = None
        self._init_fused_evaluator()

    @property
    def uses_fused(self) -> bool:
        return self.fused_evaluator is not None

    def close(self) -> None:
        self.fused_evaluator = None
        runner = getattr(self, 'runner', None)
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
            self.runner = None

    def __del__(self):
        self.close()

    def _find_world_free_root_qadr(self) -> int | None:
        m = self.model
        root = int(self.robot.root_body_id)
        if int(m.body_parentid[root]) != 0:
            return None
        free_type = int(self.mj.mjtJoint.mjJNT_FREE)
        for j in range(int(m.njnt)):
            if int(m.jnt_bodyid[j]) == root and int(m.jnt_type[j]) == free_type:
                return int(m.jnt_qposadr[j])
        return None

    def snapshot_to_state(self, snapshot, out: np.ndarray | None = None) -> np.ndarray:
        d = self._state_scratch
        d.time = float(snapshot.time)
        d.qpos[:] = snapshot.qpos
        d.qvel[:] = snapshot.qvel
        if self.model.na:
            d.act[:] = snapshot.act
        target = self._state_pack if out is None else out
        self.mj.mj_getState(self.model, d, target, self.state_spec)
        if out is None:
            return target.copy()
        return target

    def snapshots_to_states(self, snapshots: Sequence) -> np.ndarray:
        out = np.empty((len(snapshots), self.nstate), dtype=np.float64)
        for i, snap in enumerate(snapshots):
            self.snapshot_to_state(snap, out=out[i])
        return out

    def _root_from_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._root_qadr is None:
            raise RuntimeError('root pose is not directly available from qpos')
        q = 1 + int(self._root_qadr)
        pos = state[..., q:q + 3]
        quat = state[..., q + 3:q + 7]
        up = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
        return pos[..., :2], pos[..., 2], up

    def _state_buffer(self, nbatch: int, nstep: int) -> np.ndarray:
        key = (int(nbatch), int(nstep))
        buf = self._state_buffers.get(key)
        if buf is None:
            buf = np.empty((key[0], key[1], self.nstate), dtype=np.float64)
            self._state_buffers[key] = buf
        return buf

    def _sensor_buffer(self, nbatch: int, nstep: int) -> np.ndarray:
        key = (int(nbatch), int(nstep))
        buf = self._sensor_buffers.get(key)
        if buf is None:
            buf = np.empty((key[0], key[1], int(self.model.nsensordata)), dtype=np.float64)
            self._sensor_buffers[key] = buf
        return buf

    def _cost_buffer(self, nbatch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(nbatch)
        buf = self._cost_buffers.get(n)
        if buf is None:
            buf = (
                np.empty(n, dtype=np.float64),
                np.empty(n, dtype=np.float64),
                np.empty(n, dtype=np.bool_),
            )
            self._cost_buffers[n] = buf
        return buf

    def _model_batch(self, nbatch: int):
        n = int(nbatch)
        models = self._model_batches.get(n)
        if models is None:
            models = [self.model] * n
            self._model_batches[n] = models
        return models

    def rollout_states(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        initial = np.ascontiguousarray(initial_state, dtype=np.float64)
        if initial.ndim == 1:
            initial = initial[None, :]
        ctrl = np.ascontiguousarray(controls, dtype=np.float64)
        if ctrl.ndim == 2:
            ctrl = ctrl[None, :, :]
        nbatch = max(int(initial.shape[0]), int(ctrl.shape[0]))
        nstep = int(ctrl.shape[1])
        chunk = None
        if self._supports_chunk_size and self.nthread > 1:
            chunk = (
                self.chunk_size
                if self.chunk_size > 0
                else max(1, int(math.ceil(nbatch / (2.0 * self.nthread))))
            )

        # Fast path: do the wrapper's singleton expansion once ourselves and
        # provide both output arrays, then skip repeated Python validation and
        # allocation.  The low-level rollout requires exact batch dimensions.
        if self._supports_skip_checks and self._supports_state_output:
            if initial.shape[0] == 1 and nbatch > 1:
                initial = np.repeat(initial, nbatch, axis=0)
            if ctrl.shape[0] == 1 and nbatch > 1:
                ctrl = np.repeat(ctrl, nbatch, axis=0)
            state_buf = self._state_buffer(nbatch, nstep)
            sensor_buf = self._sensor_buffer(nbatch, nstep)
            kwargs = {
                'skip_checks': True,
                'nstep': nstep,
                'state': state_buf,
                'sensordata': sensor_buf,
            }
            if chunk is not None:
                kwargs['chunk_size'] = chunk
            state, _ = self.runner.rollout(
                self._model_batch(nbatch), self.data, initial, ctrl, **kwargs
            )
            return state

        kwargs = {}
        if self._supports_state_output:
            kwargs['state'] = self._state_buffer(nbatch, nstep)
        if chunk is not None:
            kwargs['chunk_size'] = chunk
        state, _ = self.runner.rollout(self.model, self.data, initial, ctrl, **kwargs)
        return state

    @staticmethod
    def _progress_from_s(track, progress_s: np.ndarray, current_s: float) -> np.ndarray:
        s = np.asarray(progress_s, dtype=np.float64)
        prev = np.empty_like(s)
        prev[0] = float(current_s)
        if len(s) > 1:
            prev[1:] = s[:-1]
        ds = s - prev
        half = 0.5 * float(track.length)
        ds[ds > half] -= float(track.length)
        ds[ds < -half] += float(track.length)
        return np.cumsum(ds)

    def rollout_nominal(
        self,
        start_snapshot,
        controls: np.ndarray,
        track,
        current_s: float,
        *,
        control_substeps: int,
    ):
        """Propagate one nominal sequence entirely inside native rollout.

        The physics, timestep, integrator and number of substeps are identical to
        candidate evaluation.  We merely remove H Python calls around ``mj_step``.
        """
        u = np.asarray(controls, dtype=np.float64)
        clipped = np.clip(u, self._ctrl_low, self._ctrl_high)
        h = int(clipped.shape[0])
        substeps = max(1, int(control_substeps))
        expanded = np.repeat(clipped[None, :, :], substeps, axis=1)
        start_state = self.snapshot_to_state(start_snapshot)
        states = self.rollout_states(start_state[None, :], expanded)
        sampled = states[0, substeps - 1::substeps, :]
        positions, _, _ = self._root_from_state(sampled)
        progress_s, _ = track.project(positions)
        progress_s = np.asarray(progress_s, dtype=np.float64)
        cumulative = self._progress_from_s(track, progress_s, current_s)

        # Sensitivity at time t starts immediately before u[t].  These states are
        # already present in the nominal trajectory, so do not repack snapshots.
        initial_states = np.empty((h, self.nstate), dtype=np.float64)
        initial_states[0] = start_state
        if h > 1:
            initial_states[1:] = sampled[:-1]
        return NominalRollout(
            clipped,
            [],
            np.asarray(positions, dtype=np.float64).copy(),
            progress_s.copy(),
            cumulative,
            native_initial_states=initial_states,
            native_states=np.asarray(sampled, dtype=np.float64).copy(),
        )

    def nominal_lookahead_endpoints(self, nominal_states: np.ndarray, lookahead_steps: int) -> np.ndarray:
        states = np.asarray(nominal_states, dtype=np.float64)
        h = int(states.shape[0])
        L = max(1, int(lookahead_steps))
        idx = np.minimum(np.arange(h, dtype=np.int64) + L - 1, h - 1)
        xy, _, _ = self._root_from_state(states[idx])
        return np.asarray(xy, dtype=np.float64).copy()

    def evaluate(
        self,
        start_snapshot,
        control_batch: np.ndarray,
        track,
        current_s: float,
        *,
        control_substeps: int,
        nominal_controls: np.ndarray,
        cost_cfg: RolloutCostConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.supports_vectorized_cost:
            raise RuntimeError('vectorized rollout cost is unsupported for this root layout')
        batch = np.ascontiguousarray(control_batch, dtype=np.float64)
        n, h, _ = batch.shape
        substeps = max(1, int(control_substeps))

        if self.fused_evaluator is not None:
            required = (
                '_origin', '_rot', '_canonical_start', 'centerline_radius',
                'left_arc_x', 'right_arc_x', 'center_y', 'straight_length',
                'length', 'road_width',
            )
            if not all(hasattr(track, name) for name in required):
                raise RuntimeError('fused rollout currently supports StadiumTrack only')
            initial = np.ascontiguousarray(self.snapshot_to_state(start_snapshot), dtype=np.float64)
            nominal = np.ascontiguousarray(nominal_controls, dtype=np.float64)
            allowed = max(
                0.0, 0.5 * float(track.road_width) - float(cost_cfg.hard_collision_clearance)
            )
            params = np.asarray([
                float(track._origin[0]), float(track._origin[1]),
                float(track._rot[0, 0]), float(track._rot[0, 1]),
                float(track._rot[1, 0]), float(track._rot[1, 1]),
                float(track._canonical_start[0]), float(track._canonical_start[1]),
                float(track.centerline_radius), float(track.left_arc_x),
                float(track.right_arc_x), float(track.center_y),
                float(track.straight_length), float(track.length),
                allowed * allowed,
                float(cost_cfg.fall_height_fraction) * max(self.robot.initial_root_height, 1e-6),
                float(cost_cfg.min_root_up), float(cost_cfg.upright_weight),
                float(cost_cfg.control_deviation_weight), float(current_s),
                float(start_snapshot.time),
            ], dtype=np.float64)
            t_fused = time.perf_counter()
            positions, costs, terminal_progress, failed = self.fused_evaluator.evaluate(
                initial, batch, nominal, self._ctrl_scale, params, substeps
            )
            elapsed_ms = 1e3 * (time.perf_counter() - t_fused)
            positions = np.asarray(positions)
            costs = np.asarray(costs)
            terminal_progress = np.asarray(terminal_progress)
            failed = np.asarray(failed, dtype=bool)

            if self._verify_fused and not self._fused_verified:
                # First-call semantic guard. The policy JIT dominates the first
                # control update anyway, so paying for one stock rollout here is
                # preferable to benchmarking a subtly different controller.
                fused_obj = self.fused_evaluator
                self.fused_evaluator = None
                try:
                    ref_pos, ref_cost, ref_progress, ref_failed = self.evaluate(
                        start_snapshot, batch, track, current_s,
                        control_substeps=substeps, nominal_controls=nominal,
                        cost_cfg=cost_cfg,
                    )
                finally:
                    self.fused_evaluator = fused_obj

                ref_failed = np.asarray(ref_failed, dtype=bool)
                if not np.array_equal(failed, ref_failed):
                    raise RuntimeError('fused rollout verification failed: failure flags differ')
                finite = np.isfinite(ref_cost) & np.isfinite(costs)
                if not np.allclose(costs[finite], np.asarray(ref_cost)[finite], rtol=1e-9, atol=1e-10):
                    err = float(np.max(np.abs(costs[finite] - np.asarray(ref_cost)[finite]))) if np.any(finite) else math.nan
                    raise RuntimeError(f'fused rollout verification failed: cost max_abs={err:g}')
                if not np.allclose(terminal_progress, ref_progress, rtol=1e-9, atol=1e-10):
                    err = float(np.max(np.abs(terminal_progress - np.asarray(ref_progress))))
                    raise RuntimeError(f'fused rollout verification failed: progress max_abs={err:g}')
                good = ~failed
                if np.any(good) and not np.allclose(positions[good], np.asarray(ref_pos)[good], rtol=1e-9, atol=1e-10):
                    err = float(np.max(np.abs(positions[good] - np.asarray(ref_pos)[good])))
                    raise RuntimeError(f'fused rollout verification failed: XY max_abs={err:g}')
                self._fused_verified = True

            # Physics and race-cost accumulation are deliberately fused and cannot
            # be timed separately without perturbing the hot loop. Restore fused
            # timings because the one-time verifier invokes the stock path.
            self.last_rollout_fused_ms = elapsed_ms
            self.last_rollout_physics_ms = elapsed_ms
            self.last_rollout_cost_ms = 0.0
            return positions, costs, terminal_progress, failed
        expanded = np.repeat(batch, substeps, axis=1)
        initial = self.snapshot_to_state(start_snapshot)[None, :]

        self.last_rollout_fused_ms = 0.0
        t_physics = time.perf_counter()
        states = self.rollout_states(initial, expanded)
        self.last_rollout_physics_ms = 1e3 * (time.perf_counter() - t_physics)

        t_cost = time.perf_counter()
        sampled = states[:, substeps - 1::substeps, :]

        # Exact, allocation-light StadiumTrack fast path.  The rollout itself is
        # unchanged: same RK4 model, timestep, substeps, horizon and candidates.
        # We only fuse projection + failure checks + cost accumulation and read
        # the root pose directly from FULLPHYSICS output.
        use_fused_cost = bool(
            NUMBA_AVAILABLE
            and self._root_qadr is not None
            and all(hasattr(track, name) for name in (
                '_origin', '_rot', '_canonical_start', 'centerline_radius',
                'left_arc_x', 'right_arc_x', 'center_y', 'straight_length',
                'length', 'road_width',
            ))
        )
        if use_fused_cost:
            q = 1 + int(self._root_qadr)
            positions = sampled[..., q:q + 2]
            costs, terminal_progress, failed = self._cost_buffer(n)
            allowed = max(
                0.0, 0.5 * float(track.road_width) - float(cost_cfg.hard_collision_clearance)
            )
            stadium_rollout_cost_from_states(
                sampled,
                batch,
                np.asarray(nominal_controls, dtype=np.float64),
                self._ctrl_scale,
                q,
                float(start_snapshot.time),
                float(current_s),
                float(track._origin[0]),
                float(track._origin[1]),
                float(track._rot[0, 0]),
                float(track._rot[0, 1]),
                float(track._rot[1, 0]),
                float(track._rot[1, 1]),
                float(track._canonical_start[0]),
                float(track._canonical_start[1]),
                float(track.centerline_radius),
                float(track.left_arc_x),
                float(track.right_arc_x),
                float(track.center_y),
                float(track.straight_length),
                float(track.length),
                allowed * allowed,
                float(cost_cfg.fall_height_fraction) * max(self.robot.initial_root_height, 1e-6),
                float(cost_cfg.min_root_up),
                float(cost_cfg.upright_weight),
                float(cost_cfg.control_deviation_weight),
                costs,
                terminal_progress,
                failed,
            )
            self.last_rollout_cost_ms = 1e3 * (time.perf_counter() - t_cost)
            return positions, costs, terminal_progress, failed

        positions, height, up = self._root_from_state(sampled)

        flat_s, flat_d2 = track.project(positions.reshape(-1, 2))
        progress_s = np.asarray(flat_s, dtype=np.float64).reshape(n, h)
        d2 = np.asarray(flat_d2, dtype=np.float64).reshape(n, h)
        allowed = max(0.0, 0.5 * float(track.road_width) - float(cost_cfg.hard_collision_clearance))
        failure = d2 > allowed * allowed
        failure |= height < float(cost_cfg.fall_height_fraction) * max(self.robot.initial_root_height, 1e-6)
        failure |= up < float(cost_cfg.min_root_up)

        times = sampled[..., 0]
        prev_t = np.empty_like(times)
        prev_t[:, 0] = float(start_snapshot.time)
        if h > 1:
            prev_t[:, 1:] = times[:, :-1]
        failure |= times <= prev_t + 1e-15

        prev_s = np.empty_like(progress_s)
        prev_s[:, 0] = float(current_s)
        if h > 1:
            prev_s[:, 1:] = progress_s[:, :-1]
        ds = progress_s - prev_s
        half = 0.5 * float(track.length)
        ds[ds > half] -= float(track.length)
        ds[ds < -half] += float(track.length)

        failed = np.any(failure, axis=1)
        # Most racing batches are fully feasible.  Avoid the accumulate/where
        # temporaries on that common path while preserving the exact cost.
        if not np.any(failed):
            terminal_progress = np.sum(ds, axis=1)
            cumulative = np.cumsum(ds, axis=1)
            progress_cost = -np.sum(cumulative, axis=1) / max(1, h)
            upright_cost = float(cost_cfg.upright_weight) * np.sum((1.0 - up) ** 2, axis=1)
            du = batch - np.asarray(nominal_controls, dtype=np.float64)[None, :, :]
            control_cost = float(cost_cfg.control_deviation_weight) * np.sum(
                np.mean((du / self._ctrl_scale[None, None, :]) ** 2, axis=2), axis=1
            )
            costs = progress_cost + upright_cost + control_cost
        else:
            alive = np.logical_and.accumulate(~failure, axis=1)
            cumulative_alive = np.cumsum(np.where(alive, ds, 0.0), axis=1)
            terminal_progress = cumulative_alive[:, -1]
            costs = np.full(n, math.inf, dtype=np.float64)
            finite = ~failed
            if np.any(finite):
                cumulative = np.cumsum(ds[finite], axis=1)
                progress_cost = -np.sum(cumulative, axis=1) / max(1, h)
                upright_cost = float(cost_cfg.upright_weight) * np.sum((1.0 - up[finite]) ** 2, axis=1)
                du = batch[finite] - np.asarray(nominal_controls, dtype=np.float64)[None, :, :]
                control_cost = float(cost_cfg.control_deviation_weight) * np.sum(
                    np.mean((du / self._ctrl_scale[None, None, :]) ** 2, axis=2), axis=1
                )
                costs[finite] = progress_cost + upright_cost + control_cost

        self.last_rollout_cost_ms = 1e3 * (time.perf_counter() - t_cost)
        return positions, costs, terminal_progress, failed

    def estimate_joint_task_jacobians(
        self,
        snapshots: Sequence | None,
        nominal_controls: np.ndarray,
        *,
        control_substeps: int,
        lookahead_steps: int,
        epsilon_fraction: float = 1e-3,
        initial_states: np.ndarray | None = None,
        nominal_states: np.ndarray | None = None,
        time_indices: np.ndarray | Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Finite-difference J using one vectorized native rollout dispatch.

        Full-horizon refresh preserves the previous numerical semantics: each
        finite-difference row includes a restarted unperturbed base trajectory.

        For a *subset* of time indices (used by shifted-Jacobian refresh), the
        continuous nominal trajectory supplies endpoints for untouched rows, so
        only the requested restarted bases and actuator perturbations are simulated.
        """
        if not self.supports_vectorized_cost:
            raise RuntimeError('vectorized Jacobians are unsupported for this root layout')
        u_nom = np.asarray(nominal_controls, dtype=np.float64)
        h, nu = u_nom.shape
        L = max(1, int(lookahead_steps))
        substeps = max(1, int(control_substeps))

        if initial_states is None:
            if snapshots is None:
                raise ValueError('snapshots or initial_states are required')
            initial_all = self.snapshots_to_states(snapshots)
        else:
            initial_all = np.asarray(initial_states, dtype=np.float64)
        if initial_all.shape != (h, self.nstate):
            raise ValueError(f'initial_states must have shape {(h, self.nstate)}, got {initial_all.shape}')

        if time_indices is None:
            ids = np.arange(h, dtype=np.int64)
        else:
            ids = np.asarray(time_indices, dtype=np.int64).reshape(-1)
            ids = ids[(ids >= 0) & (ids < h)]
            ids = np.unique(ids)

        all_ids = np.arange(h, dtype=np.int64)
        ell_all = np.minimum(L, h - all_ids)
        future_idx_all = np.minimum(all_ids[:, None] + np.arange(L)[None, :], h - 1)
        base_controls_all = u_nom[future_idx_all]  # [H, L, nu]

        # The continuous nominal endpoints are exact for the proposal center.
        # They are also enough for untouched tail rows. Requested FD rows still
        # get a restarted base trajectory, preserving the previous derivative
        # semantics despite qfrc_warmstart not being in FULLPHYSICS.
        endpoints = None
        if nominal_states is not None:
            nominal_states_arr = np.asarray(nominal_states, dtype=np.float64)
            if nominal_states_arr.shape[0] == h:
                idx = np.minimum(all_ids + L - 1, h - 1)
                xy, _, _ = self._root_from_state(nominal_states_arr[idx])
                endpoints = np.asarray(xy, dtype=np.float64).copy()

        if endpoints is None:
            # Compatibility path: without nominal states we must resimulate all
            # bases in order to return endpoints for the complete horizon.
            base_ids = all_ids
            endpoints = np.empty((h, 2), dtype=np.float64)
        else:
            # With nominal endpoints available, only requested derivative rows
            # need restarted bases.
            base_ids = ids

        b = int(base_ids.size)
        m = int(ids.size)
        controls_parts = []
        init_parts = []

        if b:
            controls_parts.append(base_controls_all[base_ids])
            init_parts.append(initial_all[base_ids])

        denom = np.empty((m, nu), dtype=np.float64)
        if m:
            future = base_controls_all[ids]
            perturbed = np.repeat(future[:, None, :, :], nu, axis=1)
            actuator = np.arange(nu, dtype=np.int64)
            eps = np.maximum(1e-7, float(epsilon_fraction) * self._fd_scale)
            perturbed[:, actuator, 0, actuator] += eps[None, :]
            np.clip(
                perturbed[:, :, 0, :], self._ctrl_low, self._ctrl_high,
                out=perturbed[:, :, 0, :],
            )
            denom[:] = perturbed[:, actuator, 0, actuator] - future[:, 0, :]
            controls_parts.append(perturbed.reshape(m * nu, L, nu))
            init_parts.append(np.repeat(initial_all[ids], nu, axis=0))

        jac = np.zeros((h, 2, nu), dtype=np.float64)
        if not controls_parts:
            return jac, endpoints

        controls = np.concatenate(controls_parts, axis=0)
        init_batch = np.concatenate(init_parts, axis=0)
        expanded = np.repeat(controls, substeps, axis=1)
        out = self.rollout_states(init_batch, expanded)

        if b:
            ell_base = np.minimum(L, h - base_ids)
            base_step = ell_base * substeps - 1
            base_terminal = out[np.arange(b), base_step]
            base_xy, _, _ = self._root_from_state(base_terminal)
            base_xy = np.asarray(base_xy, dtype=np.float64)
            endpoints[base_ids] = base_xy

        if m:
            # Requested ids are always present among base_ids. With nominal
            # states base_ids == ids; without them base_ids == all horizon rows.
            if b == m and np.array_equal(base_ids, ids):
                fd_base = endpoints[ids]
            else:
                fd_base = endpoints[ids]
            ell = np.minimum(L, h - ids)
            terminal_step = np.repeat(ell * substeps - 1, nu)
            pert_terminal = out[b + np.arange(m * nu), terminal_step]
            xy, _, _ = self._root_from_state(pert_terminal)
            xy = np.asarray(xy, dtype=np.float64).reshape(m, nu, 2)
            jac_local = np.zeros((m, nu, 2), dtype=np.float64)
            valid = np.abs(denom) > 1e-12
            delta = xy - fd_base[:, None, :]
            jac_local[valid] = delta[valid] / denom[valid, None]
            jac[ids] = np.swapaxes(jac_local, 1, 2)
        return jac, endpoints


@dataclass
class NominalRollout:
    controls: np.ndarray
    snapshots: list
    positions: np.ndarray
    progress_s: np.ndarray
    cumulative_progress: np.ndarray
    # Native full-physics states at control boundaries.  These are optional so
    # the legacy Python rollout path remains unchanged.
    native_initial_states: np.ndarray | None = None
    native_states: np.ndarray | None = None


def rollout_control_nominal(
    robot,
    start_snapshot,
    controls: np.ndarray,
    track,
    current_s: float,
    *,
    control_substeps: int,
    native_batcher: NativeRolloutBatcher | None = None,
) -> NominalRollout:
    """Roll out a supplied warm-start sequence without invoking the policy.

    With the native backend this is one C++ rollout call.  The fallback keeps
    the original Python loop exactly for unsupported MuJoCo/root layouts.
    """
    if native_batcher is not None and native_batcher.supports_vectorized_cost:
        return native_batcher.rollout_nominal(
            start_snapshot, controls, track, current_s,
            control_substeps=control_substeps,
        )

    controls = np.asarray(controls, dtype=np.float64)
    d = robot.new_data(start_snapshot)
    h = int(controls.shape[0])
    clipped = np.empty_like(controls)
    snapshots = []
    positions = np.empty((h, 2), dtype=np.float64)
    progress_s = np.empty(h, dtype=np.float64)
    cumulative = np.empty(h, dtype=np.float64)
    s_prev = float(current_s)
    cum = 0.0
    for t in range(h):
        snapshots.append(robot.snapshot(d))
        u = robot.clip_ctrl(controls[t])
        clipped[t] = u
        robot.step_control(u, substeps=control_substeps, data=d)
        p = robot.xy(d)
        s_new, _ = track.project(p)
        cum += track.signed_progress_delta(float(s_new), s_prev)
        s_prev = float(s_new)
        positions[t] = p
        progress_s[t] = float(s_new)
        cumulative[t] = cum
    return NominalRollout(clipped, snapshots, positions, progress_s, cumulative)


def rollout_policy_nominal(
    robot,
    start_snapshot,
    policy,
    track,
    prior,
    current_s: float,
    *,
    horizon: int,
    control_substeps: int,
) -> NominalRollout:
    """Roll out the robot's default policy in native MuJoCo to seed the nominal."""
    d = robot.new_data(start_snapshot)
    controls = np.empty((horizon, robot.nu), dtype=np.float64)
    snapshots = []
    positions = np.empty((horizon, 2), dtype=np.float64)
    progress_s = np.empty(horizon, dtype=np.float64)
    cumulative = np.empty(horizon, dtype=np.float64)
    s_prev = float(current_s)
    cum = 0.0
    for t in range(horizon):
        snapshots.append(robot.snapshot(d))
        s_now, _ = track.project(robot.xy(d))
        u = np.asarray(policy.action(robot, d, track=track, prior=prior, current_s=float(s_now)), dtype=np.float64)
        controls[t] = robot.clip_ctrl(u)
        robot.step_control(controls[t], substeps=control_substeps, data=d)
        p = robot.xy(d)
        s_new, _ = track.project(p)
        cum += track.signed_progress_delta(float(s_new), s_prev)
        s_prev = float(s_new)
        positions[t] = p
        progress_s[t] = float(s_new)
        cumulative[t] = cum
    return NominalRollout(controls, snapshots, positions, progress_s, cumulative)


def rollout_controls(
    robot,
    start_snapshot,
    controls: np.ndarray,
    track,
    current_s: float,
    *,
    control_substeps: int,
    nominal_controls: np.ndarray | None = None,
    cost_cfg: RolloutCostConfig | None = None,
) -> tuple[np.ndarray, float, float, bool]:
    """Evaluate one direct joint-control sequence entirely in native MuJoCo."""
    cfg = cost_cfg or RolloutCostConfig()
    d = robot.new_data(start_snapshot)
    controls = np.asarray(controls, dtype=np.float64)
    nominal_controls = controls if nominal_controls is None else np.asarray(nominal_controls, dtype=np.float64)
    positions = np.empty((len(controls), 2), dtype=np.float64)
    s_prev = float(current_s)
    cumulative = 0.0
    prefix_sum = 0.0
    control_cost = 0.0
    upright_cost = 0.0
    allowed = max(0.0, 0.5 * float(track.road_width) - float(cfg.hard_collision_clearance))
    off_track = False
    ctrl_scale = np.maximum(robot.control_scale(), 1e-6) if robot.nu else None

    for t, u in enumerate(controls):
        robot.step_control(u, substeps=control_substeps, data=d)
        p = robot.xy(d)
        positions[t] = p
        s_new, d2 = track.project(p)
        if float(d2) > allowed * allowed:
            off_track = True
            break
        if robot.root_height(d) < cfg.fall_height_fraction * max(robot.initial_root_height, 1e-6):
            off_track = True
            break
        up = robot.root_up(d)
        if up < cfg.min_root_up:
            off_track = True
            break
        ds = track.signed_progress_delta(float(s_new), s_prev)
        cumulative += ds
        s_prev = float(s_new)
        prefix_sum += cumulative
        upright_cost += float(cfg.upright_weight) * (1.0 - up) ** 2
        if robot.nu:
            du = u - nominal_controls[min(t, len(nominal_controls) - 1)]
            control_cost += float(cfg.control_deviation_weight) * float(np.mean((du / ctrl_scale) ** 2))

    if off_track:
        return positions, math.inf, cumulative, True
    progress_cost = -prefix_sum / max(1, len(controls))
    return positions, float(progress_cost + upright_cost + control_cost), cumulative, False


def evaluate_control_batch(
    robot,
    start_snapshot,
    control_batch: np.ndarray,
    track,
    current_s: float,
    *,
    control_substeps: int,
    nominal_controls: np.ndarray,
    cost_cfg: RolloutCostConfig,
    workers: int = 0,
    native_batcher: NativeRolloutBatcher | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if native_batcher is not None and native_batcher.supports_vectorized_cost:
        return native_batcher.evaluate(
            start_snapshot, control_batch, track, current_s,
            control_substeps=control_substeps,
            nominal_controls=nominal_controls,
            cost_cfg=cost_cfg,
        )
    batch = np.asarray(control_batch, dtype=np.float64)
    n, h, _ = batch.shape
    positions = np.empty((n, h, 2), dtype=np.float64)
    costs = np.empty(n, dtype=np.float64)
    progress = np.empty(n, dtype=np.float64)
    failed = np.zeros(n, dtype=bool)

    def one(i: int):
        pos, cost, prog, fail = rollout_controls(
            robot,
            start_snapshot,
            batch[i],
            track,
            current_s,
            control_substeps=control_substeps,
            nominal_controls=nominal_controls,
            cost_cfg=cost_cfg,
        )
        return i, pos, cost, prog, fail

    if int(workers) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            iterator = pool.map(one, range(n))
            for i, pos, cost, prog, fail in iterator:
                positions[i] = pos
                costs[i] = cost
                progress[i] = prog
                failed[i] = fail
    else:
        for i in range(n):
            _, pos, cost, prog, fail = one(i)
            positions[i] = pos
            costs[i] = cost
            progress[i] = prog
            failed[i] = fail
    return positions, costs, progress, failed


def _endpoint_from_snapshot(robot, snapshot, controls: np.ndarray, control_substeps: int) -> np.ndarray:
    d = robot.new_data(snapshot)
    for u in controls:
        robot.step_control(u, substeps=control_substeps, data=d)
    return robot.xy(d)


def estimate_joint_task_jacobians(
    robot,
    snapshots: Sequence,
    nominal_controls: np.ndarray,
    *,
    control_substeps: int,
    lookahead_steps: int,
    epsilon_fraction: float = 1e-3,
    centered: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct MuJoCo finite-difference J_t = d p_xy(t+L) / d ctrl_t.

    This is model agnostic and includes contacts, actuator dynamics and the full
    robot morphology. It replaces the two-control A/B projection in the source
    controller with a joint-dimensional raw local sensitivity.
    """
    u_nom = np.asarray(nominal_controls, dtype=np.float64)
    h, nu = u_nom.shape
    jac = np.zeros((h, 2, nu), dtype=np.float64)
    endpoints = np.empty((h, 2), dtype=np.float64)
    scale = np.maximum(robot.control_scale(fraction=1.0), 1e-6)
    L = max(1, int(lookahead_steps))

    for t in range(h):
        end = min(h, t + L)
        future = u_nom[t:end]
        base = _endpoint_from_snapshot(robot, snapshots[t], future, control_substeps)
        endpoints[t] = base
        for j in range(nu):
            eps = max(1e-7, float(epsilon_fraction) * float(scale[j]))
            plus = future.copy()
            plus[0, j] += eps
            plus[0] = robot.clip_ctrl(plus[0])
            p_plus = _endpoint_from_snapshot(robot, snapshots[t], plus, control_substeps)
            if centered:
                minus = future.copy()
                minus[0, j] -= eps
                minus[0] = robot.clip_ctrl(minus[0])
                p_minus = _endpoint_from_snapshot(robot, snapshots[t], minus, control_substeps)
                denom = float(plus[0, j] - minus[0, j])
                if abs(denom) > 1e-12:
                    jac[t, :, j] = (p_plus - p_minus) / denom
            else:
                denom = float(plus[0, j] - future[0, j])
                if abs(denom) > 1e-12:
                    jac[t, :, j] = (p_plus - base) / denom
    return jac, endpoints


def refine_policy_nominal(
    robot,
    start_snapshot,
    policy_rollout: NominalRollout,
    track,
    prior,
    *,
    control_substeps: int,
    lookahead_steps: int,
    iterations: int = 1,
    damping: float = 1e-4,
    step_size: float = 0.35,
    max_control_step_fraction: float = 0.15,
    epsilon_fraction: float = 1e-3,
    native_batcher: NativeRolloutBatcher | None = None,
) -> tuple[NominalRollout, np.ndarray, np.ndarray]:
    """Policy-seeded iLQR-like task-space refinement.

    A full contact-rich iLQR implementation that works identically for every
    robot actuator model is brittle. This uses the same local principle:
    linearize MuJoCo around the policy rollout, solve a damped least-squares
    task correction, update the joint sequence, and re-rollout. The resulting
    sensitivity matrices are also the exact matrices consumed by SPG.
    """
    controls = np.asarray(policy_rollout.controls, dtype=np.float64).copy()
    current = policy_rollout
    jac = np.zeros((len(controls), 2, robot.nu), dtype=np.float64)
    endpoints = current.positions.copy()
    max_step = max_control_step_fraction * np.maximum(robot.control_scale(fraction=1.0), 1e-6)

    def _estimate(rollout, ctrls):
        if native_batcher is not None and native_batcher.supports_vectorized_cost:
            return native_batcher.estimate_joint_task_jacobians(
                rollout.snapshots if rollout.native_initial_states is None else None,
                ctrls,
                control_substeps=control_substeps,
                lookahead_steps=lookahead_steps,
                epsilon_fraction=epsilon_fraction,
                initial_states=rollout.native_initial_states,
                nominal_states=rollout.native_states,
            )
        return estimate_joint_task_jacobians(
            robot, rollout.snapshots, ctrls,
            control_substeps=control_substeps,
            lookahead_steps=lookahead_steps,
            epsilon_fraction=epsilon_fraction,
        )

    for _ in range(max(0, int(iterations))):
        jac, endpoints = _estimate(current, controls)
        for t in range(len(controls)):
            s_target, _ = track.project(endpoints[t])
            mean, _ = prior.sample(track, float(s_target))
            error = np.asarray(mean, dtype=np.float64) - endpoints[t]
            j = jac[t]
            pinv = j.T @ np.linalg.inv(j @ j.T + float(damping) * np.eye(2))
            du = float(step_size) * (pinv @ error)
            du = np.clip(du, -max_step, max_step)
            controls[t] = robot.clip_ctrl(controls[t] + du)

        # Re-rollout the corrected open-loop sequence.  On the native path this
        # remains one batched C++ call and preserves the same physics exactly.
        current_s0 = float(track.project(robot.xy(robot.new_data(start_snapshot)))[0])
        current = rollout_control_nominal(
            robot, start_snapshot, controls, track, current_s0,
            control_substeps=control_substeps,
            native_batcher=native_batcher,
        )

    # Ensure sensitivities correspond to the final nominal.
    jac, endpoints = _estimate(current, current.controls)
    return current, jac, endpoints
