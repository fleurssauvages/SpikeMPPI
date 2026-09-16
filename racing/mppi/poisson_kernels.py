"""Optional, exact-model acceleration for the existing Poisson-MPPI sampler.

The small-mean Poisson generator uses the same product-of-uniforms algorithm
as NumPy, but caches exp(-lambda) across the rollout population. Larger means
use Generator.poisson. Draw order remains level -> sign -> rollout -> time ->
channel. There is no Bernoulli approximation, event cap, parallel RNG, altered
mark law, or changed twitch kernel. See NumPy's random_poisson_mult at:
https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c

Only implementation changes: compile scalar work, avoid broadcast temporaries,
and omit event-history arrays when firing plasticity is disabled. The NumPy
reference path remains selectable with --poisson-sampler numpy.
"""
from __future__ import annotations

import math
import warnings
import numpy as np

from .fast_kernels import NUMBA_AVAILABLE, njit


@njit(cache=True, nogil=True, fastmath=False)
def _poisson_block(rng, mean, threshold, out) -> None:
    """Fill one sign/mark block; keep the RNG loop separate for optimization."""
    n, h, m = out.shape
    for i in range(n):
        for t in range(h):
            for j in range(m):
                lam = mean[t, j]
                if lam == 0.0:
                    value = 0
                elif lam < 10.0:
                    value = 0
                    product = 1.0
                    while True:
                        product *= rng.random()
                        if product <= threshold[t, j]:
                            break
                        value += 1
                else:
                    value = rng.poisson(lam)
                out[i, t, j] = value


@njit(cache=True, nogil=True, fastmath=False)
def marked_impulses(rng, pos_rate, neg_rate, mark_prob, amplitudes,
                    dt: float, n: int, collect_counts: bool):
    """Generate the original signed marked events, in the original RNG order.

    Call with validated nonnegative finite Poisson means and float64 arrays.
    Returned count arrays are empty when collect_counts is false; event_total
    remains exact and is sufficient for poisson_events_mean diagnostics.
    """
    h, m = pos_rate.shape
    levels = len(amplitudes)
    impulses = np.zeros((n, h, m), dtype=np.float64)
    plus_scratch = np.empty((n, h, m), dtype=np.int64)
    minus_scratch = np.empty((n, h, m), dtype=np.int64)
    if collect_counts:
        pos_counts = np.zeros((n, h, m), dtype=np.int32)
        neg_counts = np.zeros((n, h, m), dtype=np.int32)
        level_counts = np.empty((n, h, m, levels), dtype=np.int32)
    else:
        pos_counts = np.empty((0, 0, 0), dtype=np.int32)
        neg_counts = np.empty((0, 0, 0), dtype=np.int32)
        level_counts = np.empty((0, 0, 0, 0), dtype=np.int32)
    mean_pos = np.empty((h, m), dtype=np.float64)
    mean_neg = np.empty((h, m), dtype=np.float64)
    threshold_pos = np.empty((h, m), dtype=np.float64)
    threshold_neg = np.empty((h, m), dtype=np.float64)
    event_total = 0
    for level in range(levels):
        amplitude = amplitudes[level]
        for t in range(h):
            for j in range(m):
                # Preserve the original multiplication order, including for
                # time-varying maps supplied by research/diagnostic callers.
                exposure = dt * mark_prob[t, j, level]
                lp = pos_rate[t, j] * exposure
                ln = neg_rate[t, j] * exposure
                if (not math.isfinite(lp) or not math.isfinite(ln)
                        or lp < 0.0 or ln < 0.0
                        or lp > 9.223372006484771e18 or ln > 9.223372006484771e18):
                    raise ValueError("invalid or excessively large Poisson mean")
                mean_pos[t, j] = lp
                mean_neg[t, j] = ln
                threshold_pos[t, j] = math.exp(-lp)
                threshold_neg[t, j] = math.exp(-ln)
        # NumPy previously drew the entire positive block before the negative
        # block at each recruitment level. Do not interleave these RNG calls.
        _poisson_block(rng, mean_pos, threshold_pos, plus_scratch)
        _poisson_block(rng, mean_neg, threshold_neg, minus_scratch)
        for i in range(n):
            for t in range(h):
                for j in range(m):
                    plus = plus_scratch[i, t, j]
                    minus = minus_scratch[i, t, j]
                    impulses[i, t, j] += amplitude * (plus - minus)
                    event_total += plus + minus
                    if collect_counts:
                        pos_counts[i, t, j] += plus
                        neg_counts[i, t, j] += minus
                        level_counts[i, t, j, level] = plus + minus
    return impulses, pos_counts, neg_counts, level_counts, event_total


@njit(cache=True, nogil=True, fastmath=False)
def twitch_filter(impulses: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Original finite causal FIR convolution, with identical lag order."""
    n, h, m = impulses.shape
    filtered = np.zeros_like(impulses)
    for lag in range(min(h, len(kernel))):
        k = kernel[lag]
        for i in range(n):
            for t in range(lag, h):
                for j in range(m):
                    filtered[i, t, j] += k * impulses[i, t - lag, j]
    return filtered


_prepared = False
_prepare_error: Exception | None = None


def resolve_poisson_sampler(requested: str) -> str:
    """Compile/probe before control starts, never using the controller RNG."""
    global _prepared, _prepare_error
    if requested == "numpy":
        return "numpy"
    if not NUMBA_AVAILABLE or not hasattr(marked_impulses, "signatures"):
        if requested == "numba":
            raise RuntimeError("--poisson-sampler numba requires a working Numba installation with JIT enabled")
        return "numpy"
    if not _prepared and _prepare_error is None:
        try:
            rng = np.random.default_rng(7319)
            rate = np.full((2, 2), 8.0)
            prob = np.full((2, 2, 2), 0.5)
            out = marked_impulses(rng, rate, rate, prob, np.array([0.5, 1.0]),
                                  0.02, 2, True)
            twitch_filter(out[0], np.array([1.0, 0.5]))
            # Probe the actual fixed-law production path as well as the audit helper.
            static = StaticPoissonSampler(rate, rate, prob, np.array([0.5, 1.0]),
                                        0.02, 2, np.array([1.0, 0.5]), backend="numba")
            static.sample(rng)
            static.sample_projected(rng, np.eye(2), collect_channel_energy=True)
            _prepared = True
        except Exception as exc:
            _prepare_error = exc
            if requested == "auto":
                warnings.warn(
                    f"Spike acceleration unavailable; using NumPy: {exc}",
                    RuntimeWarning, stacklevel=2,
                )
    if not _prepared:
        if requested == "numba":
            raise RuntimeError("Numba could not compile the Poisson sampler") from _prepare_error
        return "numpy"
    return "numba"


@njit(cache=True, nogil=True, fastmath=False)
def _static_impulses(rng, means, thresholds, amplitudes, impulses, plus, minus):
    """Same event draws as marked_impulses; only a scalar count is retained."""
    impulses[:] = 0.0
    n, h, m = impulses.shape
    total = 0
    for level in range(len(amplitudes)):
        _poisson_block(rng, means[level, 0], thresholds[level, 0], plus)
        _poisson_block(rng, means[level, 1], thresholds[level, 1], minus)
        amplitude = amplitudes[level]
        for i in range(n):
            for t in range(h):
                for j in range(m):
                    p, q = plus[i, t, j], minus[i, t, j]
                    impulses[i, t, j] += amplitude * (p - q)
                    total += p + q
    return total


@njit(cache=True, nogil=True, fastmath=False)
def _twitch_into(impulses, kernel, filtered):
    """Allocation-free version of the existing causal FIR, same lag order."""
    filtered[:] = 0.0
    n, h, m = impulses.shape
    for lag in range(min(h, len(kernel))):
        k = kernel[lag]
        for i in range(n):
            for t in range(lag, h):
                for j in range(m):
                    filtered[i, t, j] += k * impulses[i, t - lag, j]






@njit(cache=True, nogil=True, fastmath=False)
def _static_project_homogeneous_identity(rng, block_means, amplitudes,
                                         motor_impulses):
    """Exact fixed-Spike fast path for identity motor wiring.

    The production ``poisson`` sampler has one channel per actuator and an
    identity dictionary. The generic projector therefore checked all ``nu``
    wiring entries for every sparse event even though exactly one is nonzero.
    This specialization keeps the identical flattened Poisson process and RNG
    draw order, but writes the event directly to its matching motor channel.
    """
    motor_impulses[:] = 0.0
    n, h, nu = motor_impulses.shape
    cells = n * h * nu
    total = 0
    for level in range(len(amplitudes)):
        amp = amplitudes[level]
        for sign in range(2):
            lam = block_means[level, sign]
            if lam <= 0.0:
                continue
            signed_amp = amp if sign == 0 else -amp
            pos = rng.exponential(1.0 / lam)
            while pos < cells:
                idx = int(pos)
                j = idx % nu
                q = idx // nu
                t = q % h
                i = q // h
                motor_impulses[i, t, j] += signed_amp
                total += 1
                pos += rng.exponential(1.0 / lam)
    return total


@njit(cache=True, nogil=True, fastmath=False)
def _static_project_homogeneous_antagonistic(rng, block_means, amplitudes,
                                               muscle_impulses):
    """Map fixed signed Spike events to nonnegative antagonist muscles.

    Positive events go to channel ``2*j`` and negative events to ``2*j+1``.
    No event itself is negative; opposite joint torque is produced by the
    opposing muscle transmission in the MuJoCo model.
    """
    muscle_impulses[:] = 0.0
    n, h, nu = muscle_impulses.shape
    m = nu // 2
    cells = n * h * m
    total = 0
    for level in range(len(amplitudes)):
        amp = amplitudes[level]
        for sign in range(2):
            lam = block_means[level, sign]
            if lam <= 0.0:
                continue
            pos = rng.exponential(1.0 / lam)
            while pos < cells:
                idx = int(pos)
                j = idx % m
                q = idx // m
                t = q % h
                i = q // h
                muscle_impulses[i, t, 2 * j + sign] += amp
                total += 1
                pos += rng.exponential(1.0 / lam)
    return total


@njit(cache=True, nogil=True, fastmath=False)
def _static_project_homogeneous(rng, block_means, amplitudes, wiring,
                                motor_impulses, channel_energy):
    """Exact homogeneous marked-Poisson sampling with event skipping.

    Each level/sign block is a homogeneous Poisson process over the flattened
    rollout x time x neuron bins. Counts in disjoint unit bins are therefore
    independent Poisson(block_mean), exactly matching the fixed marked-Poisson
    law while avoiding one RNG call for every usually-empty bin. Events are
    projected immediately into motor space; the common twitch is applied later.
    """
    motor_impulses[:] = 0.0
    if channel_energy.size:
        channel_energy[:] = 0.0
    n, h, nu = motor_impulses.shape
    m = wiring.shape[0]
    cells = n * h * m
    total = 0
    for level in range(len(amplitudes)):
        amp = amplitudes[level]
        amp2 = amp * amp
        for sign in range(2):
            lam = block_means[level, sign]
            if lam <= 0.0:
                continue
            signed_amp = amp if sign == 0 else -amp
            pos = rng.exponential(1.0 / lam)
            while pos < cells:
                idx = int(pos)
                j = idx % m
                q = idx // m
                t = q % h
                i = q // h
                if channel_energy.size:
                    channel_energy[i, j] += amp2
                for u in range(nu):
                    wij = wiring[j, u]
                    if wij != 0.0:
                        motor_impulses[i, t, u] += signed_amp * wij
                total += 1
                pos += rng.exponential(1.0 / lam)
    return total



@njit(cache=True, nogil=True, fastmath=False)
def _project_impulses(impulses, wiring, motor_impulses, channel_energy):
    """Project neuron impulses to motors and optionally accumulate neuron energy.

    Projection is done before the common linear twitch filter. This is exactly
    equivalent to filtering every neuron and projecting afterwards, while the
    expensive FIR therefore scales with motor count rather than neuron count.
    """
    n, h, m = impulses.shape
    nu = wiring.shape[1]
    motor_impulses[:] = 0.0
    if channel_energy.size:
        channel_energy[:] = 0.0
    for i in range(n):
        for t in range(h):
            for j in range(m):
                a = impulses[i, t, j]
                if channel_energy.size:
                    channel_energy[i, j] += a * a
                if a != 0.0:
                    for u in range(nu):
                        motor_impulses[i, t, u] += a * wiring[j, u]

class StaticPoissonSampler:
    """Cached event-law plan and reusable workspace for fixed firing statistics.

    The public controller fixes rates, signs and recruitment. A general static
    time-varying law is accepted here to test equivalence of the implementation.
    Inputs are copied: changing them later cannot leave half of a plan stale.
    ``sample`` returns borrowed filtered storage, valid until the next call.
    It never retains per-event or per-mark history arrays.
    """

    def __init__(self, pos_rate, neg_rate, mark_prob, amplitudes, dt, n, kernel,
                 *, backend="numpy"):
        pos = np.asarray(pos_rate, dtype=np.float64)
        neg = np.asarray(neg_rate, dtype=np.float64)
        probs = np.asarray(mark_prob, dtype=np.float64)
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        kernel = np.asarray(kernel, dtype=np.float64)
        if pos.ndim != 2 or neg.shape != pos.shape or min(pos.shape) < 1:
            raise ValueError("rates must have matching nonempty (H,M) shapes")
        if amplitudes.ndim != 1 or not len(amplitudes):
            raise ValueError("amplitudes must be a nonempty vector")
        if probs.shape != (*pos.shape, len(amplitudes)):
            raise ValueError("mark_prob must have shape (H,M,L)")
        if (not np.all(np.isfinite(probs)) or np.any(probs < 0)
                or not np.allclose(probs.sum(axis=-1), 1.0, rtol=1e-12, atol=1e-12)):
            raise ValueError("mark probabilities must be finite, nonnegative and sum to one")
        if (not np.isfinite(dt) or dt <= 0 or int(n) != n or n < 1
                or not np.all(np.isfinite(amplitudes))
                or kernel.ndim != 1 or not len(kernel) or not np.all(np.isfinite(kernel))):
            raise ValueError("invalid dt, rollout count, amplitudes or kernel")
        if backend not in {"numpy", "numba"}:
            raise ValueError("backend must be numpy or numba")
        self.backend = backend
        self.dt = float(dt)
        self.amplitudes = amplitudes.copy()
        self.kernel = kernel.copy()
        self.means = np.empty((len(amplitudes), 2, *pos.shape), dtype=np.float64)
        for level in range(len(amplitudes)):
            exposure = float(dt) * probs[..., level]
            self.means[level, 0] = pos * exposure
            self.means[level, 1] = neg * exposure
        if (not np.all(np.isfinite(self.means)) or np.any(self.means < 0)
                or np.any(self.means > 9.223372006484771e18)):
            raise ValueError("invalid or excessively large Poisson mean")
        # math.exp matches the compiled reference threshold calculation exactly.
        self.thresholds = np.fromiter((math.exp(-float(x)) for x in self.means.flat),
                                      dtype=np.float64, count=self.means.size).reshape(self.means.shape)
        self.expected_signed = float(dt) * (pos - neg) * np.sum(
            probs * self.amplitudes[None, None, :], axis=-1)
        self.center = bool(np.any(self.expected_signed != 0))
        # Production Poisson-MPPI uses fixed homogeneous firing/mark laws. Detect
        # that exact case once and use event-skipping sampling in motor space.
        ref = self.means[:, :, :1, :1]
        self.homogeneous_blocks = (
            self.means[:, :, 0, 0].copy()
            if np.array_equal(self.means, np.broadcast_to(ref, self.means.shape))
            else None
        )
        self.n = int(n)
        self.h = int(pos.shape[0])
        self.m = int(pos.shape[1])
        # The racing hot path projects homogeneous events directly into motor
        # channels and never needs an N-neuron horizon tensor. Keep the large
        # general-law workspaces lazy so 32/64-neuron sampling does not carry
        # unused impulses/plus/minus/filter arrays in cache.
        self.impulses = None
        self.filtered = None
        self.plus = None
        self.minus = None
        if not (backend == "numba" and self.homogeneous_blocks is not None and not self.center):
            self._ensure_channel_workspace()
        self.motor_impulses = None
        self.motor_filtered = None
        self.channel_energy = np.empty((self.n, self.m), dtype=np.float64)
        self.last_event_total = 0

    def _ensure_channel_workspace(self):
        if self.impulses is not None:
            return
        shape = (self.n, self.h, self.m)
        self.impulses = np.zeros(shape, dtype=np.float64)
        self.filtered = np.zeros(shape, dtype=np.float64)
        if self.backend == "numba":
            self.plus = np.empty(shape, dtype=np.int64)
            self.minus = np.empty(shape, dtype=np.int64)

    @property
    def workspace_nbytes(self):
        """Persistent buffers only, excludes small fixed plans and NumPy draws."""
        return sum(
            a.nbytes
            for a in (
                self.impulses, self.filtered, self.motor_impulses,
                self.motor_filtered, self.channel_energy, self.plus, self.minus,
            )
            if a is not None
        )

    def sample_projected_identity(self, rng):
        """Specialized production path for one fixed neuron per actuator.

        Returns borrowed twitch-decoded motor storage and the scalar event
        count. It is distribution-equivalent to ``sample_projected(rng, I)``
        and preserves the event RNG sequence exactly.
        """
        target_shape = (self.n, self.h, self.m)
        if self.motor_impulses is None or self.motor_impulses.shape != target_shape:
            self.motor_impulses = np.zeros(target_shape, dtype=np.float64)
            self.motor_filtered = np.zeros(target_shape, dtype=np.float64)

        if self.backend == "numba" and self.homogeneous_blocks is not None and not self.center:
            total = _static_project_homogeneous_identity(
                rng, self.homogeneous_blocks, self.amplitudes, self.motor_impulses
            )
            _twitch_into(self.motor_impulses, self.kernel, self.motor_filtered)
            self.last_event_total = int(total)
            return self.motor_filtered, self.last_event_total

        # Non-production/general-law fallback remains the established generic
        # implementation; constructing I here is outside the Numba hot path.
        identity = np.eye(self.m, dtype=np.float64)
        out, _, total = self.sample_projected(
            rng, identity, collect_channel_energy=False
        )
        return out, int(total)

    def sample_projected_antagonistic_pairs(self, rng):
        """Decode fixed Poisson signs into separate nonnegative muscle channels.

        There are ``m`` original joint-level Spike channels and ``2*m`` muscle
        outputs ordered ``[positive, negative]`` per joint.  The point-process
        law and event budget are unchanged from the torque-space Poisson baseline;
        only the sign decoder changes.
        """
        target_shape = (self.n, self.h, 2 * self.m)
        if self.motor_impulses is None or self.motor_impulses.shape != target_shape:
            self.motor_impulses = np.zeros(target_shape, dtype=np.float64)
            self.motor_filtered = np.zeros(target_shape, dtype=np.float64)

        if self.backend == "numba" and self.homogeneous_blocks is not None and not self.center:
            total = _static_project_homogeneous_antagonistic(
                rng, self.homogeneous_blocks, self.amplitudes, self.motor_impulses
            )
            _twitch_into(self.motor_impulses, self.kernel, self.motor_filtered)
            self.last_event_total = int(total)
            return self.motor_filtered, self.last_event_total

        # NumPy/general-law reference path. Each sign is routed to its own
        # nonnegative muscle instead of subtracting the two event streams.
        self.motor_impulses.fill(0.0)
        total = 0
        for level, amplitude in enumerate(self.amplitudes):
            plus = rng.poisson(
                self.means[level, 0][None], size=(self.n, self.h, self.m)
            )
            minus = rng.poisson(
                self.means[level, 1][None], size=(self.n, self.h, self.m)
            )
            self.motor_impulses[..., 0::2] += float(amplitude) * plus
            self.motor_impulses[..., 1::2] += float(amplitude) * minus
            total += int(plus.sum(dtype=np.int64)) + int(minus.sum(dtype=np.int64))
        self.motor_filtered.fill(0.0)
        h = self.motor_filtered.shape[1]
        for lag, k in enumerate(self.kernel[:h]):
            self.motor_filtered[:, lag:, :] += (
                float(k) * self.motor_impulses[:, :h-lag, :]
            )
        self.last_event_total = int(total)
        return self.motor_filtered, self.last_event_total

    def sample_projected(self, rng, wiring, *, collect_channel_energy=False):
        """Sample neuron events, project to motors, then apply the common twitch.

        Because convolution is linear and every neuron uses the same fixed twitch
        kernel, twitch(project(impulses)) == project(twitch(impulses)). This avoids
        filtering many event channels when only the physical motor outputs are required.
        """
        wiring = np.asarray(wiring, dtype=np.float64)
        if wiring.ndim != 2 or wiring.shape[0] != self.m:
            raise ValueError("wiring must have shape (channels, motors)")
        nu = int(wiring.shape[1])
        target_shape = (self.n, self.h, nu)
        if self.motor_impulses is None or self.motor_impulses.shape != target_shape:
            self.motor_impulses = np.zeros(target_shape, dtype=np.float64)
            self.motor_filtered = np.zeros(target_shape, dtype=np.float64)

        energy = self.channel_energy if collect_channel_energy else self.channel_energy[:0, :0]
        if self.backend == "numba" and self.homogeneous_blocks is not None and not self.center:
            # Exact fast path for the fixed symmetric event law used by racing.
            total = _static_project_homogeneous(
                rng, self.homogeneous_blocks, self.amplitudes, wiring,
                self.motor_impulses, energy,
            )
            _twitch_into(self.motor_impulses, self.kernel, self.motor_filtered)
            self.last_event_total = int(total)
            return self.motor_filtered, (self.channel_energy if collect_channel_energy else None), self.last_event_total

        self._ensure_channel_workspace()
        if self.backend == "numba":
            total = _static_impulses(rng, self.means, self.thresholds, self.amplitudes,
                                     self.impulses, self.plus, self.minus)
        else:
            self.impulses.fill(0.0)
            total = 0
            for level, amplitude in enumerate(self.amplitudes):
                plus = rng.poisson(self.means[level, 0][None], size=self.impulses.shape)
                minus = rng.poisson(self.means[level, 1][None], size=self.impulses.shape)
                self.impulses += float(amplitude) * (plus - minus)
                total += int(plus.sum(dtype=np.int64)) + int(minus.sum(dtype=np.int64))
        if self.center:
            self.impulses -= self.expected_signed[None]

        if self.backend == "numba":
            _project_impulses(self.impulses, wiring, self.motor_impulses, energy)
            _twitch_into(self.motor_impulses, self.kernel, self.motor_filtered)
        else:
            self.motor_impulses[:] = (
                self.impulses.reshape(-1, wiring.shape[0]) @ wiring
            ).reshape(target_shape)
            if collect_channel_energy:
                np.einsum("ihm,ihm->im", self.impulses, self.impulses,
                          out=self.channel_energy, optimize=False)
            self.motor_filtered.fill(0.0)
            h = self.motor_filtered.shape[1]
            for lag, k in enumerate(self.kernel[:h]):
                self.motor_filtered[:, lag:, :] += float(k) * self.motor_impulses[:, :h-lag, :]

        self.last_event_total = int(total)
        return self.motor_filtered, (self.channel_energy if collect_channel_energy else None), self.last_event_total

    def sample(self, rng):
        self._ensure_channel_workspace()
        if self.backend == "numba":
            total = _static_impulses(rng, self.means, self.thresholds, self.amplitudes,
                                     self.impulses, self.plus, self.minus)
        else:
            self.impulses.fill(0.0)
            total = 0
            for level, amplitude in enumerate(self.amplitudes):
                plus = rng.poisson(self.means[level, 0][None], size=self.impulses.shape)
                minus = rng.poisson(self.means[level, 1][None], size=self.impulses.shape)
                self.impulses += float(amplitude) * (plus - minus)
                total += int(plus.sum(dtype=np.int64)) + int(minus.sum(dtype=np.int64))
        if self.center:
            self.impulses -= self.expected_signed[None]
        if self.backend == "numba":
            _twitch_into(self.impulses, self.kernel, self.filtered)
        else:
            self.filtered.fill(0.0)
            h = self.filtered.shape[1]
            for lag, k in enumerate(self.kernel[:h]):
                self.filtered[:, lag:, :] += float(k) * self.impulses[:, :h-lag, :]
        self.last_event_total = int(total)
        return self.filtered, self.last_event_total
