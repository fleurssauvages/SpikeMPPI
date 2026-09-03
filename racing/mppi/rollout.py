from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Sequence
import numpy as np

from .fast_kernels import NUMBA_AVAILABLE, stadium_rollout_cost_from_states



@dataclass
class RolloutCostConfig:
    hard_collision_clearance: float = 0.02
    # Exact race/fused fall rule: root_up < min_root_up AND torso touches ground.
    # fall_height_fraction is used only by the state-only fallback as a contact proxy.
    fall_height_fraction: float = 0.45
    min_root_up: float = 0.0
    upright_weight: float = 0.05
    control_deviation_weight: float = 1e-4

    # Push-task shaping.  These are ignored when task body == robot root.
    # Box progress remains the primary objective.  The extra potential terms
    # make approaching the box informative before the first contact.
    box_progress_weight: float = 1.0
    robot_progress_weight: float = 0.35
    robot_box_approach_weight: float = 1.00
    box_max_lift: float = 0.12
    box_min_up: float = 0.75




class NativeRolloutBatcher:
    supports_vectorized_jacobians = True
    """Fast batched open-loop rollouts using MuJoCo's native C++ rollout module.

    Besides parallel MPPI evaluation, this class also owns the hot-path state
    buffers used by warm-start propagation and SPG finite differences.  Keeping
    those operations in the same native rollout backend avoids Python ``mj_step``
    loops and repeated state packing.
    """

    def __init__(self, robot, *, workers: int = 16, batch_hint: int = 32, chunk_size: int = 0, fused: bool = False) -> None:
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
        self._initial_buffers: dict[int, np.ndarray] = {}
        self._expanded_control_buffers: dict[tuple[int, int, int, int], np.ndarray] = {}
        self._model_batches: dict[int, list] = {}
        self._root_qadr = self._find_world_free_root_qadr()
        self._task_qadr = getattr(robot, "task_qpos_adr", self._root_qadr)
        self.supports_vectorized_cost = self._root_qadr is not None and self._task_qadr is not None
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
        self._fused_params: np.ndarray | None = None
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
        if self._root_qadr is None or self._task_qadr is None:
            raise RuntimeError('fused rollout requires free-joint robot and task roots')
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
            try:
                self.fused_evaluator = _fused_mujoco.FusedRolloutEvaluator(
                    model_path, int(self.nthread), int(self._root_qadr),
                    int(self._task_qadr), int(chunk)
                )
            except TypeError as exc:
                raise RuntimeError(
                    "fused rollout extension has an old ABI. Rebuild it with: "
                    "python racing/setup_native.py build_ext --inplace"
                ) from exc
            required_methods = ('evaluate', 'rollout_nominal', 'estimate_spg_jacobian')
            if not all(hasattr(self.fused_evaluator, name) for name in required_methods):
                self.fused_evaluator = None
                raise RuntimeError(
                    "fused rollout extension has an old ABI. Rebuild it with: "
                    "python racing/setup_native.py build_ext --inplace"
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

    @staticmethod
    def _pose_from_state(state: np.ndarray, qadr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = 1 + int(qadr)  # FULLPHYSICS packs time before qpos.
        pos = state[..., q:q + 3]
        quat = state[..., q + 3:q + 7]
        up = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
        return pos[..., :2], pos[..., 2], up

    def _root_from_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._root_qadr is None:
            raise RuntimeError('root pose is not directly available from qpos')
        return self._pose_from_state(state, int(self._root_qadr))

    def _task_xy_from_state(self, state: np.ndarray) -> np.ndarray:
        if self._task_qadr is None:
            raise RuntimeError('task pose is not directly available from qpos')
        xy, _, _ = self._pose_from_state(state, int(self._task_qadr))
        return xy

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

    def _broadcast_initial(self, initial: np.ndarray, nbatch: int) -> np.ndarray:
        """Expand a singleton state into a persistent native-rollout buffer."""
        n = int(nbatch)
        buf = self._initial_buffers.get(n)
        if buf is None:
            buf = np.empty((n, self.nstate), dtype=np.float64)
            self._initial_buffers[n] = buf
        buf[:] = initial[0]
        return buf

    def _expand_controls(self, controls: np.ndarray, substeps: int) -> np.ndarray:
        """Repeat controls in time without allocating on every MPPI update."""
        ctrl = np.ascontiguousarray(controls, dtype=np.float64)
        if ctrl.ndim == 2:
            ctrl = ctrl[None, :, :]
        sub = max(1, int(substeps))
        if sub == 1:
            return ctrl
        n, h, nu = ctrl.shape
        key = (int(n), int(h), sub, int(nu))
        buf = self._expanded_control_buffers.get(key)
        if buf is None:
            buf = np.empty((n, h * sub, nu), dtype=np.float64)
            self._expanded_control_buffers[key] = buf
        buf.reshape(n, h, sub, nu)[:] = ctrl[:, :, None, :]
        return buf

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
                initial = self._broadcast_initial(initial, nbatch)
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
        start_state = self.snapshot_to_state(start_snapshot, out=self._state_pack)

        if self.fused_evaluator is not None:
            boundaries, positions = self.fused_evaluator.rollout_nominal(
                start_state, np.ascontiguousarray(clipped, dtype=np.float64), substeps
            )
            boundaries = np.asarray(boundaries, dtype=np.float64)
            # Keep the public diagnostic trajectory stable across subsequent
            # control ticks even though the C++ evaluator reuses its XY buffer.
            positions = np.asarray(positions, dtype=np.float64).copy()
            progress_s, _ = track.project(positions)
            progress_s = np.asarray(progress_s, dtype=np.float64)
            cumulative = self._progress_from_s(track, progress_s, current_s)
            return NominalRollout(
                clipped,
                [],
                positions,
                progress_s.copy(),
                cumulative,
                native_initial_states=boundaries[:-1],
                native_states=boundaries[1:],
            )

        expanded = self._expand_controls(clipped, substeps)
        states = self.rollout_states(start_state[None, :], expanded)
        sampled = states[0, substeps - 1::substeps, :]
        positions = self._task_xy_from_state(sampled)
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
        xy = self._task_xy_from_state(states[idx])
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

        # Pushing uses the box as the task body, but dense shaping also rewards
        # robot advancement and reduction of robot-box distance.  Compute the
        # initial potentials once so fused/native/Python cost paths agree.
        pushing = self._task_qadr != self._root_qadr
        qpos0 = np.asarray(start_snapshot.qpos, dtype=np.float64)
        root_xy0 = qpos0[int(self._root_qadr):int(self._root_qadr) + 2]
        task_xy0 = qpos0[int(self._task_qadr):int(self._task_qadr) + 2]
        current_root_s = float(track.project(root_xy0)[0])
        initial_task_root_distance = float(np.linalg.norm(task_xy0 - root_xy0))
        initial_task_height = float(self.robot.task_rest_height) if pushing else 0.0

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
            if self._fused_params is None:
                self._fused_params = np.empty(29, dtype=np.float64)
            params = self._fused_params
            params[:] = [
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
                float(cost_cfg.control_deviation_weight),
                float(cost_cfg.box_progress_weight),
                float(cost_cfg.robot_progress_weight),
                float(cost_cfg.robot_box_approach_weight),
                float(cost_cfg.box_max_lift), float(cost_cfg.box_min_up),
                float(current_s), float(current_root_s),
                float(initial_task_root_distance), float(initial_task_height),
                float(start_snapshot.time),
            ]
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
                # First-call semantic guard.  Keep continuous verification on
                # the original vectorized MuJoCo rollout path so fused and
                # reference trajectories are compared without introducing a
                # second independently stepped scalar trajectory.  FULLPHYSICS
                # does not contain contact pairs, however, so use direct scalar
                # MuJoCo stepping *only* to verify the exact discrete fall flag
                # (flipped torso AND torso-ground contact).

                # 1) Continuous reference: stock vectorized MuJoCo rollout.
                expanded_verify = self._expand_controls(batch, substeps)
                initial_verify = self.snapshot_to_state(start_snapshot)[None, :]
                verify_states = self.rollout_states(initial_verify, expanded_verify)
                verify_sampled = verify_states[:, substeps - 1::substeps, :]
                ref_pos = np.asarray(self._task_xy_from_state(verify_sampled), dtype=np.float64)
                ref_root_pos, _, ref_up = self._root_from_state(verify_sampled)

                flat_s, _ = track.project(ref_pos.reshape(-1, 2))
                ref_s = np.asarray(flat_s, dtype=np.float64).reshape(n, h)
                prev_s = np.empty_like(ref_s)
                prev_s[:, 0] = float(current_s)
                if h > 1:
                    prev_s[:, 1:] = ref_s[:, :-1]
                ref_ds = ref_s - prev_s
                half_length = 0.5 * float(track.length)
                ref_ds[ref_ds > half_length] -= float(track.length)
                ref_ds[ref_ds < -half_length] += float(track.length)
                ref_progress = np.sum(ref_ds, axis=1)

                ref_cumulative = np.cumsum(ref_ds, axis=1)
                ref_cost = (
                    -float(cost_cfg.box_progress_weight)
                    * np.sum(ref_cumulative, axis=1) / max(1, h)
                )
                if pushing:
                    root_s_arr, _ = track.project(np.asarray(ref_root_pos).reshape(-1, 2))
                    root_s_arr = np.asarray(root_s_arr, dtype=np.float64).reshape(n, h)
                    root_prev = np.empty_like(root_s_arr)
                    root_prev[:, 0] = float(current_root_s)
                    if h > 1:
                        root_prev[:, 1:] = root_s_arr[:, :-1]
                    root_ds = root_s_arr - root_prev
                    root_ds[root_ds > half_length] -= float(track.length)
                    root_ds[root_ds < -half_length] += float(track.length)
                    root_cumulative = np.cumsum(root_ds, axis=1)
                    coupled = np.minimum(root_cumulative, np.maximum(ref_cumulative, 0.0))
                    approach = float(initial_task_root_distance) - np.linalg.norm(
                        ref_pos - np.asarray(ref_root_pos), axis=2
                    )
                    ref_cost -= (
                        float(cost_cfg.robot_progress_weight)
                        * np.sum(coupled, axis=1) / max(1, h)
                    )
                    ref_cost -= (
                        float(cost_cfg.robot_box_approach_weight)
                        * np.sum(approach, axis=1) / max(1, h)
                    )
                ref_cost += float(cost_cfg.upright_weight) * np.sum((1.0 - ref_up) ** 2, axis=1)
                du = batch - nominal[None, :, :]
                ref_cost += float(cost_cfg.control_deviation_weight) * np.sum(
                    np.mean((du / self._ctrl_scale[None, None, :]) ** 2, axis=2), axis=1
                )

                # 2) Discrete reference: exact scalar contact-aware failure flags.
                ref_failed = np.zeros_like(failed, dtype=bool)
                for i in range(n):
                    _, _, _, fail_i = rollout_controls(
                        self.robot, start_snapshot, batch[i], track, current_s,
                        control_substeps=substeps, nominal_controls=nominal,
                        cost_cfg=cost_cfg,
                    )
                    ref_failed[i] = fail_i

                if not np.array_equal(failed, ref_failed):
                    mismatch = np.flatnonzero(failed != ref_failed)
                    preview = ','.join(map(str, mismatch[:8]))
                    raise RuntimeError(
                        'fused rollout verification failed: exact failure flags differ '
                        f'at rollout indices [{preview}]'
                    )

                # Compare continuous values only for trajectories that did not
                # fail. Failed trajectories terminate at contact/off-track time
                # in the fused evaluator, whereas the vectorized state rollout
                # intentionally continues through the full horizon.
                good = ~failed
                if np.any(good):
                    numeric_rtol = 1.0e-7
                    state_atol = 1.0e-7
                    cost_atol = 1.0e-6
                    if not np.allclose(
                        terminal_progress[good], ref_progress[good],
                        rtol=numeric_rtol, atol=state_atol,
                    ):
                        err = float(np.max(np.abs(
                            terminal_progress[good] - ref_progress[good]
                        )))
                        raise RuntimeError(
                            'fused rollout verification failed: '
                            f'progress max_abs={err:g}'
                        )
                    if not np.allclose(
                        positions[good], ref_pos[good],
                        rtol=numeric_rtol, atol=state_atol,
                    ):
                        err = float(np.max(np.abs(positions[good] - ref_pos[good])))
                        raise RuntimeError(
                            'fused rollout verification failed: '
                            f'XY max_abs={err:g}'
                        )
                    finite_good = good & np.isfinite(costs) & np.isfinite(ref_cost)
                    if np.any(finite_good) and not np.allclose(
                        costs[finite_good], ref_cost[finite_good],
                        rtol=numeric_rtol, atol=cost_atol,
                    ):
                        diff = np.abs(costs[finite_good] - ref_cost[finite_good])
                        err = float(np.max(diff))
                        denom = np.maximum(
                            np.maximum(
                                np.abs(costs[finite_good]),
                                np.abs(ref_cost[finite_good]),
                            ),
                            1.0,
                        )
                        rel = float(np.max(diff / denom))
                        raise RuntimeError(
                            'fused rollout verification failed: '
                            f'cost max_abs={err:g}, max_rel={rel:g}'
                        )
                self._fused_verified = True

            # Physics and race-cost accumulation are deliberately fused and cannot
            # be timed separately without perturbing the hot loop. Restore fused
            # timings because the one-time verifier invokes the stock path.
            self.last_rollout_fused_ms = elapsed_ms
            self.last_rollout_physics_ms = elapsed_ms
            self.last_rollout_cost_ms = 0.0
            return positions, costs, terminal_progress, failed
        expanded = self._expand_controls(batch, substeps)
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
            task_q = 1 + int(self._task_qadr)
            root_q = 1 + int(self._root_qadr)
            positions = sampled[..., task_q:task_q + 2]
            costs, terminal_progress, failed = self._cost_buffer(n)
            allowed = max(
                0.0, 0.5 * float(track.road_width) - float(cost_cfg.hard_collision_clearance)
            )
            stadium_rollout_cost_from_states(
                sampled,
                batch,
                np.asarray(nominal_controls, dtype=np.float64),
                self._ctrl_scale,
                task_q,
                root_q,
                float(start_snapshot.time),
                float(current_s),
                float(current_root_s),
                float(initial_task_root_distance),
                float(cost_cfg.box_progress_weight),
                float(cost_cfg.robot_progress_weight),
                float(cost_cfg.robot_box_approach_weight),
                float(cost_cfg.box_max_lift),
                float(cost_cfg.box_min_up),
                float(initial_task_height),
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

        positions = self._task_xy_from_state(sampled)
        root_positions, height, up = self._root_from_state(sampled)

        flat_s, flat_d2 = track.project(positions.reshape(-1, 2))
        progress_s = np.asarray(flat_s, dtype=np.float64).reshape(n, h)
        d2 = np.asarray(flat_d2, dtype=np.float64).reshape(n, h)
        allowed = max(0.0, 0.5 * float(track.road_width) - float(cost_cfg.hard_collision_clearance))
        failure = d2 > allowed * allowed
        if self._task_qadr != self._root_qadr:
            _, root_flat_d2 = track.project(np.asarray(root_positions).reshape(-1, 2))
            root_d2 = np.asarray(root_flat_d2, dtype=np.float64).reshape(n, h)
            failure |= root_d2 > allowed * allowed
        # FULLPHYSICS state output does not contain the contact list.  The
        # stock vectorized fallback therefore uses a conservative proxy that
        # still requires BOTH inversion and a very low torso.  The fused
        # evaluator and the real race use exact torso-ground contact pairs.
        failure |= (
            (up < float(cost_cfg.min_root_up))
            & (height < float(cost_cfg.fall_height_fraction) * max(self.robot.initial_root_height, 1e-6))
        )
        if pushing:
            task_q = 1 + int(self._task_qadr)
            task_z = sampled[..., task_q + 2]
            task_qx = sampled[..., task_q + 4]
            task_qy = sampled[..., task_q + 5]
            task_up = 1.0 - 2.0 * (task_qx * task_qx + task_qy * task_qy)
            failure |= task_z > float(initial_task_height) + float(cost_cfg.box_max_lift)
            failure |= task_up < float(cost_cfg.box_min_up)

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

        root_ds = None
        approach = None
        if pushing:
            root_s_arr, _ = track.project(np.asarray(root_positions).reshape(-1, 2))
            root_s_arr = np.asarray(root_s_arr, dtype=np.float64).reshape(n, h)
            root_prev = np.empty_like(root_s_arr)
            root_prev[:, 0] = float(current_root_s)
            if h > 1:
                root_prev[:, 1:] = root_s_arr[:, :-1]
            root_ds = root_s_arr - root_prev
            root_ds[root_ds > half] -= float(track.length)
            root_ds[root_ds < -half] += float(track.length)
            approach = float(initial_task_root_distance) - np.linalg.norm(
                positions - np.asarray(root_positions), axis=2
            )

        failed = np.any(failure, axis=1)
        # Most racing batches are fully feasible.  Avoid the accumulate/where
        # temporaries on that common path while preserving the exact cost.
        if not np.any(failed):
            terminal_progress = np.sum(ds, axis=1)
            cumulative = np.cumsum(ds, axis=1)
            progress_cost = -float(cost_cfg.box_progress_weight) * np.sum(cumulative, axis=1) / max(1, h)
            if pushing:
                root_cumulative = np.cumsum(root_ds, axis=1)
                # Reward robot advancement only while it remains coupled to the
                # box's forward progress. Running past a stationary box cannot
                # earn this shaping term; the approach potential handles reach.
                coupled = np.minimum(root_cumulative, np.maximum(cumulative, 0.0))
                progress_cost -= float(cost_cfg.robot_progress_weight) * np.sum(coupled, axis=1) / max(1, h)
                progress_cost -= float(cost_cfg.robot_box_approach_weight) * np.sum(approach, axis=1) / max(1, h)
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
                progress_cost = -float(cost_cfg.box_progress_weight) * np.sum(cumulative, axis=1) / max(1, h)
                if pushing:
                    root_cumulative = np.cumsum(root_ds[finite], axis=1)
                    coupled = np.minimum(root_cumulative, np.maximum(cumulative, 0.0))
                    progress_cost -= float(cost_cfg.robot_progress_weight) * np.sum(coupled, axis=1) / max(1, h)
                    progress_cost -= float(cost_cfg.robot_box_approach_weight) * np.sum(approach[finite], axis=1) / max(1, h)
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
        """Finite-difference task Jacobians using the fastest available backend.

        The fused backend restores the nominal control-boundary FULLPHYSICS
        state together with ``qacc_warmstart`` and therefore needs only the
        actuator perturbations.  The stock ``mujoco.rollout`` fallback retains
        the historical restarted-baseline trajectories because FULLPHYSICS by
        itself does not contain solver warm-start state.
        """
        if not self.supports_vectorized_cost:
            raise RuntimeError('vectorized Jacobians are unsupported for this root layout')
        u_nom = np.asarray(nominal_controls, dtype=np.float64)
        h, nu = u_nom.shape
        L = max(1, int(lookahead_steps))
        substeps = max(1, int(control_substeps))

        if time_indices is None:
            ids = None
            fused_ids = None
        else:
            ids = np.asarray(time_indices, dtype=np.int64).reshape(-1)
            ids = ids[(ids >= 0) & (ids < h)]
            ids = np.unique(ids)
            fused_ids = np.ascontiguousarray(ids, dtype=np.int64)

        # Fused nominal rollouts cache both FULLPHYSICS control-boundary states
        # and qacc_warmstart.  Restoring both for each actuator perturbation
        # makes the continuous nominal endpoint the exact FD baseline, removing
        # one redundant restarted baseline rollout per refreshed horizon row.
        if self.fused_evaluator is not None and nominal_states is not None:
            jac, endpoints = self.fused_evaluator.estimate_spg_jacobian(
                np.ascontiguousarray(u_nom, dtype=np.float64),
                L,
                float(epsilon_fraction),
                substeps,
                fused_ids,
            )
            return (
                np.asarray(jac, dtype=np.float64),
                np.asarray(endpoints, dtype=np.float64),
            )

        if ids is None:
            ids = np.arange(h, dtype=np.int64)

        if initial_states is None:
            if snapshots is None:
                raise ValueError('snapshots or initial_states are required')
            initial_all = self.snapshots_to_states(snapshots)
        else:
            initial_all = np.asarray(initial_states, dtype=np.float64)
        if initial_all.shape != (h, self.nstate):
            raise ValueError(f'initial_states must have shape {(h, self.nstate)}, got {initial_all.shape}')

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
                xy = self._task_xy_from_state(nominal_states_arr[idx])
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
        expanded = self._expand_controls(controls, substeps)
        out = self.rollout_states(init_batch, expanded)

        if b:
            ell_base = np.minimum(L, h - base_ids)
            base_step = ell_base * substeps - 1
            base_terminal = out[np.arange(b), base_step]
            base_xy = self._task_xy_from_state(base_terminal)
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
            xy = self._task_xy_from_state(pert_terminal)
            xy = np.asarray(xy, dtype=np.float64).reshape(m, nu, 2)
            jac_local = np.zeros((m, nu, 2), dtype=np.float64)
            valid = np.abs(denom) > 1e-12
            delta = xy - fd_base[:, None, :]
            jac_local[valid] = delta[valid] / denom[valid, None]
            jac[ids] = np.swapaxes(jac_local, 1, 2)
        return jac, endpoints

    def estimate_joint_task_time_sensitivities(
        self,
        snapshots: Sequence | None,
        nominal_controls: np.ndarray,
        *,
        control_substeps: int,
        future_steps: int,
        epsilon_fraction: float = 1e-3,
        initial_states: np.ndarray | None = None,
        nominal_states: np.ndarray | None = None,
        time_indices: np.ndarray | Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Finite-difference future task sensitivity for time-dependent SPG.

        Returns the forward tensor

            H[k, ell] = d y[k+ell+1] / d z[k]

        with shape ``[H, W, 2, nu]`` together with the matching nominal future
        task positions ``[H, W, 2]``.  One actuator perturbation is propagated
        once and every control-boundary XY value in the requested future window
        is retained.  The SPG layer subsequently forms the damped local inverse
        ``G[k] ~= d z[k] / d [y[k+1], ...]`` used to move spatial variance into
        actuator space.
        """
        if not self.supports_vectorized_cost:
            raise RuntimeError('vectorized time sensitivities are unsupported for this root layout')
        u_nom = np.asarray(nominal_controls, dtype=np.float64)
        h, nu = u_nom.shape
        W = max(1, int(future_steps))
        substeps = max(1, int(control_substeps))

        if time_indices is None:
            ids = np.arange(h, dtype=np.int64)
            fused_ids = None
        else:
            ids = np.asarray(time_indices, dtype=np.int64).reshape(-1)
            ids = ids[(ids >= 0) & (ids < h)]
            ids = np.unique(ids)
            fused_ids = np.ascontiguousarray(ids, dtype=np.int64)

        # New fused builds retain every future XY endpoint produced by the same
        # persistent worker perturbation used for classic SPG.  Older fused
        # builds simply fall through to the stock persistent rollout path, so
        # adding this controller variant does not invalidate the classic ABI.
        if (
            self.fused_evaluator is not None
            and nominal_states is not None
            and hasattr(self.fused_evaluator, 'estimate_spg_time_sensitivity')
        ):
            sensitivity, future_positions = self.fused_evaluator.estimate_spg_time_sensitivity(
                np.ascontiguousarray(u_nom, dtype=np.float64),
                W,
                float(epsilon_fraction),
                substeps,
                fused_ids,
            )
            return (
                np.asarray(sensitivity, dtype=np.float64),
                np.asarray(future_positions, dtype=np.float64),
            )

        if initial_states is None:
            if snapshots is None:
                raise ValueError('snapshots or initial_states are required')
            initial_all = self.snapshots_to_states(snapshots)
        else:
            initial_all = np.asarray(initial_states, dtype=np.float64)
        if initial_all.shape != (h, self.nstate):
            raise ValueError(f'initial_states must have shape {(h, self.nstate)}, got {initial_all.shape}')

        # Matching nominal points y[k+1], ..., y[k+W].  Nominal states from the
        # fused nominal path already contain these exact control-boundary states.
        future_idx = np.minimum(
            np.arange(h, dtype=np.int64)[:, None]
            + np.arange(W, dtype=np.int64)[None, :],
            h - 1,
        )
        future_positions = np.empty((h, W, 2), dtype=np.float64)
        if nominal_states is not None:
            states = np.asarray(nominal_states, dtype=np.float64)
            if states.shape[0] != h:
                raise ValueError(f'nominal_states must have first dimension {h}')
            future_positions[:] = self._task_xy_from_state(states[future_idx])
        else:
            future_positions.fill(np.nan)

        sensitivity = np.zeros((h, W, 2, nu), dtype=np.float64)
        if ids.size == 0:
            return sensitivity, future_positions

        # Fixed-width sequences keep mujoco.rollout rectangular.  Tail controls
        # are repeated only outside the valid MPC range; the SPG factor builder
        # ignores those padded lags via valid_lengths[k] = min(W, H-k).
        ctrl_idx = np.minimum(
            ids[:, None] + np.arange(W, dtype=np.int64)[None, :], h - 1
        )
        base_controls = u_nom[ctrl_idx]
        m = int(ids.size)
        actuator = np.arange(nu, dtype=np.int64)
        eps = np.maximum(1e-7, float(epsilon_fraction) * self._fd_scale)

        perturbed = np.repeat(base_controls[:, None, :, :], nu, axis=1)
        perturbed[:, actuator, 0, actuator] += eps[None, :]
        np.clip(
            perturbed[:, :, 0, :], self._ctrl_low, self._ctrl_high,
            out=perturbed[:, :, 0, :],
        )
        denom = perturbed[:, actuator, 0, actuator] - base_controls[:, 0, :]

        controls = np.concatenate(
            (base_controls, perturbed.reshape(m * nu, W, nu)), axis=0
        )
        init_batch = np.concatenate(
            (initial_all[ids], np.repeat(initial_all[ids], nu, axis=0)), axis=0
        )
        expanded = self._expand_controls(controls, substeps)
        out = self.rollout_states(init_batch, expanded)
        sampled = out[:, substeps - 1::substeps, :]
        base_xy = np.asarray(self._task_xy_from_state(sampled[:m]), dtype=np.float64)
        pert_xy = np.asarray(
            self._task_xy_from_state(sampled[m:]), dtype=np.float64
        ).reshape(m, nu, W, 2)

        if nominal_states is None:
            future_positions[ids] = base_xy

        for row, t in enumerate(ids):
            valid_len = min(W, h - int(t))
            delta = pert_xy[row, :, :valid_len, :] - base_xy[row, None, :valid_len, :]
            valid = np.abs(denom[row]) > 1e-12
            if np.any(valid):
                local = np.zeros((nu, valid_len, 2), dtype=np.float64)
                local[valid] = delta[valid] / denom[row, valid, None, None]
                sensitivity[t, :valid_len] = np.transpose(local, (1, 2, 0))

        return sensitivity, future_positions


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
        p = robot.task_xy(d)
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
        # The transferred policy is still a *running* policy.  Its track
        # command therefore follows the robot root, not the pushed object's
        # location.  MPPI alone sees and optimizes the box objective.
        s_now, _ = track.project(robot.xy(d))
        u = np.asarray(policy.action(robot, d, track=track, prior=prior, current_s=float(s_now)), dtype=np.float64)
        controls[t] = robot.clip_ctrl(u)
        robot.step_control(controls[t], substeps=control_substeps, data=d)
        p = robot.task_xy(d)
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
    pushing = int(robot.task_body_id) != int(robot.root_body_id)
    root_s_prev = float(track.project(robot.xy(d))[0])
    initial_task_root_distance = float(np.linalg.norm(robot.task_xy(d) - robot.xy(d)))
    initial_task_height = float(robot.task_rest_height) if pushing else 0.0
    cumulative = 0.0
    root_cumulative = 0.0
    prefix_sum = 0.0
    root_prefix_sum = 0.0
    approach_prefix_sum = 0.0
    control_cost = 0.0
    upright_cost = 0.0
    allowed = max(0.0, 0.5 * float(track.road_width) - float(cfg.hard_collision_clearance))
    off_track = False
    ctrl_scale = np.maximum(robot.control_scale(), 1e-6) if robot.nu else None

    for t, u in enumerate(controls):
        robot.step_control(u, substeps=control_substeps, data=d)
        p = robot.task_xy(d)
        positions[t] = p
        s_new, d2 = track.project(p)
        _, robot_d2 = track.project(robot.xy(d))
        if float(d2) > allowed * allowed or float(robot_d2) > allowed * allowed:
            off_track = True
            break
        up = robot.root_up(d)
        if robot.has_fallen(d, flipped_threshold=cfg.min_root_up):
            off_track = True
            break
        if pushing and (
            robot.task_height(d) > initial_task_height + cfg.box_max_lift
            or robot.task_up(d) < cfg.box_min_up
        ):
            off_track = True
            break
        ds = track.signed_progress_delta(float(s_new), s_prev)
        cumulative += ds
        s_prev = float(s_new)
        prefix_sum += cumulative
        if pushing:
            root_s_new, _ = track.project(robot.xy(d))
            root_ds = track.signed_progress_delta(float(root_s_new), root_s_prev)
            root_cumulative += root_ds
            root_s_prev = float(root_s_new)
            root_prefix_sum += min(root_cumulative, max(cumulative, 0.0))
            approach_prefix_sum += initial_task_root_distance - float(np.linalg.norm(robot.task_xy(d) - robot.xy(d)))
        upright_cost += float(cfg.upright_weight) * (1.0 - up) ** 2
        if robot.nu:
            du = u - nominal_controls[min(t, len(nominal_controls) - 1)]
            control_cost += float(cfg.control_deviation_weight) * float(np.mean((du / ctrl_scale) ** 2))

    if off_track:
        return positions, math.inf, cumulative, True
    inv_h = 1.0 / max(1, len(controls))
    progress_cost = -float(cfg.box_progress_weight) * prefix_sum * inv_h
    if pushing:
        progress_cost -= float(cfg.robot_progress_weight) * root_prefix_sum * inv_h
        progress_cost -= float(cfg.robot_box_approach_weight) * approach_prefix_sum * inv_h
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
    return robot.task_xy(d)


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


def estimate_joint_task_time_sensitivities(
    robot,
    snapshots: Sequence,
    nominal_controls: np.ndarray,
    *,
    control_substeps: int,
    future_steps: int,
    epsilon_fraction: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Python compatibility path for ``d y[k+ell+1] / d z[k]``.

    The expensive path is intentionally simple and exact: each actuator is
    perturbed only at row ``k`` and the perturbation is propagated through the
    same nominal future controls while every intermediate task XY is recorded.
    """
    u_nom = np.asarray(nominal_controls, dtype=np.float64)
    h, nu = u_nom.shape
    W = max(1, int(future_steps))
    sensitivity = np.zeros((h, W, 2, nu), dtype=np.float64)
    future_positions = np.empty((h, W, 2), dtype=np.float64)
    scale = np.maximum(robot.control_scale(fraction=1.0), 1e-6)

    for k in range(h):
        valid_len = min(W, h - k)
        future = u_nom[k:k + valid_len]

        d_base = robot.new_data(snapshots[k])
        for ell, u in enumerate(future):
            robot.step_control(u, substeps=control_substeps, data=d_base)
            future_positions[k, ell] = robot.task_xy(d_base)
        if valid_len < W:
            future_positions[k, valid_len:] = future_positions[k, valid_len - 1]

        for actuator in range(nu):
            eps = max(1e-7, float(epsilon_fraction) * float(scale[actuator]))
            plus = future.copy()
            plus[0, actuator] += eps
            plus[0] = robot.clip_ctrl(plus[0])
            denom = float(plus[0, actuator] - future[0, actuator])
            if abs(denom) <= 1e-12:
                continue

            d_plus = robot.new_data(snapshots[k])
            for ell, u in enumerate(plus):
                robot.step_control(u, substeps=control_substeps, data=d_plus)
                sensitivity[k, ell, :, actuator] = (
                    np.asarray(robot.task_xy(d_plus), dtype=np.float64)
                    - future_positions[k, ell]
                ) / denom

    return sensitivity, future_positions


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
        if (
            native_batcher is not None
            and getattr(native_batcher, "supports_vectorized_jacobians", False)
        ):
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
        current_s0 = float(track.project(robot.task_xy(robot.new_data(start_snapshot)))[0])
        current = rollout_control_nominal(
            robot, start_snapshot, controls, track, current_s0,
            control_substeps=control_substeps,
            native_batcher=native_batcher,
        )

    # Ensure sensitivities correspond to the final nominal.
    jac, endpoints = _estimate(current, current.controls)
    return current, jac, endpoints
