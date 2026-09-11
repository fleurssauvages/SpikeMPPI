"""Optional, exact-model acceleration for the existing Spike-MPPI sampler.

The small-mean Poisson generator uses the same product-of-uniforms algorithm
as NumPy, but caches exp(-lambda) across the rollout population. Larger means
use Generator.poisson. Draw order remains level -> sign -> rollout -> time ->
channel. There is no Bernoulli approximation, event cap, parallel RNG, altered
mark law, or changed twitch kernel. See NumPy's random_poisson_mult at:
https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c

Only implementation changes: compile scalar work, avoid broadcast temporaries,
and omit event-history arrays when firing plasticity is disabled. The NumPy
reference path remains selectable with --spike-sampler numpy.
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
    remains exact and is sufficient for spike_events_mean diagnostics.
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


def resolve_spike_sampler(requested: str) -> str:
    """Compile/probe before control starts, never using the controller RNG."""
    global _prepared, _prepare_error
    if requested == "numpy":
        return "numpy"
    if not NUMBA_AVAILABLE or not hasattr(marked_impulses, "signatures"):
        if requested == "numba":
            raise RuntimeError("--spike-sampler numba requires a working Numba installation with JIT enabled")
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
            static = StaticSpikeSampler(rate, rate, prob, np.array([0.5, 1.0]),
                                        0.02, 2, np.array([1.0, 0.5]), backend="numba")
            static.sample(rng)
            static.sample_projected(rng, np.eye(2), collect_channel_energy=True)
            static.sample_projected_adaptive(
                rng, np.eye(2), total_rate_hz=16.0,
                rate_prob=np.array([0.5, 0.5]),
                sign_prob=np.array([0.5, 0.5]),
                mark_prob=np.array([[0.5, 0.5], [0.5, 0.5]]),
                collect_stats=True,
            )
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
            raise RuntimeError("Numba could not compile the Spike sampler") from _prepare_error
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

    The production ``spike`` sampler has one channel per actuator and an
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
def _adaptive_project_budget(rng, total_rate_hz, dt, rate_cdf, sign_prob,
                             mark_cdf, amplitudes, wiring, motor_impulses,
                             event_counts, positive_counts, level_counts):
    """Exact fixed-budget marked-Poisson superposition with learned marks.

    The total population process is Poisson(total_rate_hz).  Conditional on an
    event, the neuron is drawn from rate_cdf, sign from sign_prob, recruitment
    from mark_cdf, and the discrete horizon bin uniformly.  Poisson thinning
    makes this exactly equivalent to independent homogeneous per-neuron Poisson
    processes whose rates sum to total_rate_hz.
    """
    motor_impulses[:] = 0.0
    event_counts[:] = 0
    positive_counts[:] = 0
    level_counts[:] = 0
    n, h, nu = motor_impulses.shape
    m = wiring.shape[0]
    levels = len(amplitudes)
    expected_total = total_rate_hz * dt * h
    total = 0

    for i in range(n):
        k = rng.poisson(expected_total)
        total += k
        for _ in range(k):
            r = rng.random()
            lo = 0
            hi = m - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if r <= rate_cdf[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            j = lo
            t = int(rng.random() * h)
            is_pos = rng.random() < sign_prob[j]
            q = rng.random()
            level = 0
            while level + 1 < levels and q > mark_cdf[j, level]:
                level += 1
            amp = amplitudes[level]
            signed_amp = amp if is_pos else -amp
            event_counts[i, j] += 1
            if is_pos:
                positive_counts[i, j] += 1
            level_counts[i, j, level] += 1
            for u in range(nu):
                wij = wiring[j, u]
                if wij != 0.0:
                    motor_impulses[i, t, u] += signed_amp * wij

    # Center the learned asymmetric event law exactly in expectation so every
    # candidate distribution still explores around the unchanged nominal.
    # rate probability is recovered from the CDF increments.
    for u in range(nu):
        mean_u = 0.0
        prev = 0.0
        for j in range(m):
            qj = rate_cdf[j] - prev
            prev = rate_cdf[j]
            mean_amp = 0.0
            prev_mark = 0.0
            for level in range(levels):
                pij = mark_cdf[j, level] - prev_mark
                prev_mark = mark_cdf[j, level]
                mean_amp += pij * amplitudes[level]
            mean_u += qj * (2.0 * sign_prob[j] - 1.0) * mean_amp * wiring[j, u]
        mean_u *= total_rate_hz * dt
        if mean_u != 0.0:
            for i in range(n):
                for t in range(h):
                    motor_impulses[i, t, u] -= mean_u
    return total



@njit(cache=True, nogil=True, fastmath=False)
def _temporal_project_budget(rng, total_rate_hz, dt, rate_cdf, sign_prob,
                             mark_cdf, amplitudes, wiring, mean_motor,
                             motor_impulses, event_counts, positive_counts,
                             level_counts):
    """Fixed-budget marked-Poisson sampling from a K-knot trajectory proposal.

    Event time is sampled from the unchanged homogeneous population Poisson
    process.  The event's latent proposal knot is drawn by linear interpolation
    between the two neighboring temporal knots; neuron/sign/recruitment are then
    sampled from that knot.  This gives a continuous-in-time mixture without
    constructing H x N probability tensors.
    """
    motor_impulses[:] = 0.0
    event_counts[:] = 0
    positive_counts[:] = 0
    level_counts[:] = 0
    n, h, nu = motor_impulses.shape
    knots, m = rate_cdf.shape
    levels = len(amplitudes)
    expected_total = total_rate_hz * dt * h
    total = 0

    for i in range(n):
        nevents = rng.poisson(expected_total)
        total += nevents
        for _ in range(nevents):
            t = int(rng.random() * h)
            if knots <= 1 or h <= 1:
                knot = 0
            else:
                x = (t * (knots - 1.0)) / (h - 1.0)
                k0 = int(x)
                if k0 >= knots - 1:
                    knot = knots - 1
                else:
                    a = x - k0
                    knot = k0 + 1 if rng.random() < a else k0

            r = rng.random()
            lo = 0
            hi = m - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if r <= rate_cdf[knot, mid]:
                    hi = mid
                else:
                    lo = mid + 1
            j = lo

            is_pos = rng.random() < sign_prob[knot, j]
            q = rng.random()
            level = 0
            while level + 1 < levels and q > mark_cdf[knot, j, level]:
                level += 1
            signed_amp = amplitudes[level] if is_pos else -amplitudes[level]
            event_counts[i, knot, j] += 1
            if is_pos:
                positive_counts[i, knot, j] += 1
            level_counts[i, knot, j, level] += 1
            for u in range(nu):
                wij = wiring[j, u]
                if wij != 0.0:
                    motor_impulses[i, t, u] += signed_amp * wij

    # Center the learned asymmetric proposal exactly in expectation at every
    # horizon bin.  This keeps the proposal centered on the unchanged nominal.
    for t in range(h):
        if knots <= 1 or h <= 1:
            k0 = 0
            k1 = 0
            a = 0.0
        else:
            x = (t * (knots - 1.0)) / (h - 1.0)
            k0 = int(x)
            if k0 >= knots - 1:
                k0 = knots - 1
                k1 = k0
                a = 0.0
            else:
                k1 = k0 + 1
                a = x - k0
        for u in range(nu):
            mu = (1.0 - a) * mean_motor[k0, u] + a * mean_motor[k1, u]
            if mu != 0.0:
                for i in range(n):
                    motor_impulses[i, t, u] -= mu
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

class StaticSpikeSampler:
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
        # Production Spike-MPPI uses fixed homogeneous firing/mark laws. Detect
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
        # The racing hot path projects homogeneous events directly into 8 motor
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
        self.event_counts = None
        self.positive_counts = None
        self.level_counts = None
        self.rate_cdf = None
        self.mark_cdf = None
        self.temporal_rate_cdf = None
        self.temporal_mark_cdf = None
        self.temporal_event_counts = None
        self.temporal_positive_counts = None
        self.temporal_level_counts = None
        self.temporal_mean_motor = None
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
        return sum(a.nbytes for a in (self.impulses, self.filtered, self.motor_impulses,
                                       self.motor_filtered, self.channel_energy,
                                       self.event_counts, self.positive_counts, self.level_counts,
                                       self.rate_cdf, self.mark_cdf,
                                       self.temporal_rate_cdf, self.temporal_mark_cdf,
                                       self.temporal_event_counts, self.temporal_positive_counts,
                                       self.temporal_level_counts, self.temporal_mean_motor,
                                       self.plus, self.minus)
                   if a is not None)

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

    def sample_projected(self, rng, wiring, *, collect_channel_energy=False):
        """Sample neuron events, project to motors, then apply the common twitch.

        Because convolution is linear and every neuron uses the same fixed twitch
        kernel, twitch(project(impulses)) == project(twitch(impulses)). This avoids
        filtering N=32/64 channels when only nu=8 motor outputs are required.
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

    def sample_projected_adaptive(self, rng, wiring, *, total_rate_hz, rate_prob,
                                  sign_prob, mark_prob, collect_stats=True):
        """Sample learned rate/sign/recruitment laws with a fixed event budget.

        ``sum(rate_prob)==1`` makes the expected number of population events
        exactly ``total_rate_hz * horizon * dt`` regardless of neuron count.
        The returned sufficient statistics are per rollout/neuron and contain no
        horizon tensor, so 256-1048 neuron learning remains compact.
        """
        wiring = np.asarray(wiring, dtype=np.float64)
        rate_prob = np.asarray(rate_prob, dtype=np.float64).reshape(-1)
        sign_prob = np.asarray(sign_prob, dtype=np.float64).reshape(-1)
        mark_prob = np.asarray(mark_prob, dtype=np.float64)
        if wiring.ndim != 2 or wiring.shape[0] != self.m:
            raise ValueError("wiring must have shape (channels, motors)")
        if rate_prob.shape != (self.m,) or sign_prob.shape != (self.m,):
            raise ValueError("rate_prob and sign_prob must have one entry per channel")
        if mark_prob.shape != (self.m, len(self.amplitudes)):
            raise ValueError("mark_prob must have shape (channels, levels)")
        if (not np.all(np.isfinite(rate_prob)) or np.any(rate_prob < 0.0)
                or abs(float(np.sum(rate_prob)) - 1.0) > 1e-8):
            raise ValueError("rate_prob must be finite, nonnegative and sum to one")
        if (not np.all(np.isfinite(sign_prob)) or np.any(sign_prob < 0.0)
                or np.any(sign_prob > 1.0)):
            raise ValueError("sign_prob must lie in [0,1]")
        if (not np.all(np.isfinite(mark_prob)) or np.any(mark_prob < 0.0)
                or not np.allclose(np.sum(mark_prob, axis=1), 1.0, atol=1e-8)):
            raise ValueError("mark_prob rows must be finite, nonnegative and sum to one")
        total_rate_hz = float(total_rate_hz)
        if not np.isfinite(total_rate_hz) or total_rate_hz <= 0.0:
            raise ValueError("total_rate_hz must be finite and positive")

        nu = int(wiring.shape[1])
        target_shape = (self.n, self.h, nu)
        if self.motor_impulses is None or self.motor_impulses.shape != target_shape:
            self.motor_impulses = np.zeros(target_shape, dtype=np.float64)
            self.motor_filtered = np.zeros(target_shape, dtype=np.float64)
        levels = len(self.amplitudes)
        if self.event_counts is None or self.event_counts.shape != (self.n, self.m):
            self.event_counts = np.zeros((self.n, self.m), dtype=np.int32)
            self.positive_counts = np.zeros((self.n, self.m), dtype=np.int32)
            self.level_counts = np.zeros((self.n, self.m, levels), dtype=np.int32)
            self.rate_cdf = np.empty(self.m, dtype=np.float64)
            self.mark_cdf = np.empty((self.m, levels), dtype=np.float64)
        np.cumsum(rate_prob, out=self.rate_cdf)
        self.rate_cdf[-1] = 1.0
        np.cumsum(mark_prob, axis=1, out=self.mark_cdf)
        self.mark_cdf[:, -1] = 1.0

        if self.backend == "numba":
            total = _adaptive_project_budget(
                rng, total_rate_hz, self.dt, self.rate_cdf, sign_prob,
                self.mark_cdf, self.amplitudes, wiring, self.motor_impulses,
                self.event_counts, self.positive_counts, self.level_counts,
            )
            _twitch_into(self.motor_impulses, self.kernel, self.motor_filtered)
        else:
            # Reference path with the same Poisson-superposition law.
            self.motor_impulses.fill(0.0)
            self.event_counts.fill(0)
            self.positive_counts.fill(0)
            self.level_counts.fill(0)
            expected_total = total_rate_hz * self.dt * self.h
            total = 0
            for i in range(self.n):
                k = int(rng.poisson(expected_total))
                total += k
                if k == 0:
                    continue
                neurons = rng.choice(self.m, size=k, p=rate_prob)
                times = rng.integers(0, self.h, size=k)
                for e in range(k):
                    j = int(neurons[e])
                    t = int(times[e])
                    is_pos = bool(rng.random() < sign_prob[j])
                    level = int(rng.choice(levels, p=mark_prob[j]))
                    amp = self.amplitudes[level] if is_pos else -self.amplitudes[level]
                    self.event_counts[i, j] += 1
                    if is_pos:
                        self.positive_counts[i, j] += 1
                    self.level_counts[i, j, level] += 1
                    self.motor_impulses[i, t] += amp * wiring[j]
            mean_amp = mark_prob @ self.amplitudes
            signed = total_rate_hz * self.dt * rate_prob * (2.0 * sign_prob - 1.0) * mean_amp
            self.motor_impulses -= (signed @ wiring)[None, None, :]
            self.motor_filtered.fill(0.0)
            h = self.motor_filtered.shape[1]
            for lag, kval in enumerate(self.kernel[:h]):
                self.motor_filtered[:, lag:, :] += float(kval) * self.motor_impulses[:, :h-lag, :]

        self.last_event_total = int(total)
        if collect_stats:
            return (self.motor_filtered, self.event_counts, self.positive_counts,
                    self.level_counts, self.last_event_total)
        return self.motor_filtered, None, None, None, self.last_event_total



    def sample_projected_temporal(self, rng, wiring, *, total_rate_hz, rate_prob,
                                  sign_prob, mark_prob, collect_stats=True):
        """Sample a state-conditioned K-knot spike-trajectory proposal.

        The global population Poisson rate is unchanged.  Temporal structure is
        represented by K knot distributions and sampled through a latent linear
        interpolation, avoiding any H x neurons probability expansion.
        """
        wiring = np.asarray(wiring, dtype=np.float64)
        rate_prob = np.asarray(rate_prob, dtype=np.float64)
        sign_prob = np.asarray(sign_prob, dtype=np.float64)
        mark_prob = np.asarray(mark_prob, dtype=np.float64)
        if rate_prob.ndim != 2:
            raise ValueError("rate_prob must have shape (knots, channels)")
        knots, channels = rate_prob.shape
        levels = len(self.amplitudes)
        if wiring.ndim != 2 or wiring.shape[0] != self.m or channels != self.m:
            raise ValueError("wiring/rate_prob channel count mismatch")
        if sign_prob.shape != (knots, self.m):
            raise ValueError("sign_prob must have shape (knots, channels)")
        if mark_prob.shape != (knots, self.m, levels):
            raise ValueError("mark_prob must have shape (knots, channels, levels)")
        if (not np.all(np.isfinite(rate_prob)) or np.any(rate_prob < 0.0)
                or not np.allclose(np.sum(rate_prob, axis=1), 1.0, atol=1e-8)):
            raise ValueError("each temporal rate_prob row must sum to one")
        if (not np.all(np.isfinite(sign_prob)) or np.any(sign_prob < 0.0)
                or np.any(sign_prob > 1.0)):
            raise ValueError("sign_prob must lie in [0,1]")
        if (not np.all(np.isfinite(mark_prob)) or np.any(mark_prob < 0.0)
                or not np.allclose(np.sum(mark_prob, axis=2), 1.0, atol=1e-8)):
            raise ValueError("temporal mark_prob rows must sum to one")
        total_rate_hz = float(total_rate_hz)
        if not np.isfinite(total_rate_hz) or total_rate_hz <= 0.0:
            raise ValueError("total_rate_hz must be finite and positive")

        nu = int(wiring.shape[1])
        target_shape = (self.n, self.h, nu)
        if self.motor_impulses is None or self.motor_impulses.shape != target_shape:
            self.motor_impulses = np.zeros(target_shape, dtype=np.float64)
            self.motor_filtered = np.zeros(target_shape, dtype=np.float64)

        stats_shape = (self.n, knots, self.m)
        if self.temporal_event_counts is None or self.temporal_event_counts.shape != stats_shape:
            self.temporal_event_counts = np.zeros(stats_shape, dtype=np.int32)
            self.temporal_positive_counts = np.zeros(stats_shape, dtype=np.int32)
            self.temporal_level_counts = np.zeros(stats_shape + (levels,), dtype=np.int32)
            self.temporal_rate_cdf = np.empty((knots, self.m), dtype=np.float64)
            self.temporal_mark_cdf = np.empty((knots, self.m, levels), dtype=np.float64)
            self.temporal_mean_motor = np.empty((knots, nu), dtype=np.float64)

        np.cumsum(rate_prob, axis=1, out=self.temporal_rate_cdf)
        self.temporal_rate_cdf[:, -1] = 1.0
        np.cumsum(mark_prob, axis=2, out=self.temporal_mark_cdf)
        self.temporal_mark_cdf[:, :, -1] = 1.0

        # Expected signed motor impulse per event-knot, then scale by the fixed
        # population event rate per control bin.  Only a K x nu matrix is kept.
        mean_amp = np.sum(mark_prob * self.amplitudes[None, None, :], axis=2)
        signed_mass = rate_prob * (2.0 * sign_prob - 1.0) * mean_amp
        self.temporal_mean_motor[:] = (
            float(total_rate_hz) * self.dt * (signed_mass @ wiring)
        )

        if self.backend == "numba":
            total = _temporal_project_budget(
                rng, total_rate_hz, self.dt, self.temporal_rate_cdf, sign_prob,
                self.temporal_mark_cdf, self.amplitudes, wiring,
                self.temporal_mean_motor, self.motor_impulses,
                self.temporal_event_counts, self.temporal_positive_counts,
                self.temporal_level_counts,
            )
            _twitch_into(self.motor_impulses, self.kernel, self.motor_filtered)
        else:
            self.motor_impulses.fill(0.0)
            self.temporal_event_counts.fill(0)
            self.temporal_positive_counts.fill(0)
            self.temporal_level_counts.fill(0)
            expected_total = total_rate_hz * self.dt * self.h
            total = 0
            for i in range(self.n):
                nevents = int(rng.poisson(expected_total))
                total += nevents
                for _ in range(nevents):
                    t = int(rng.integers(0, self.h))
                    if knots <= 1 or self.h <= 1:
                        knot = 0
                    else:
                        x = t * (knots - 1.0) / (self.h - 1.0)
                        k0 = min(int(x), knots - 1)
                        if k0 >= knots - 1:
                            knot = knots - 1
                        else:
                            knot = k0 + 1 if rng.random() < (x - k0) else k0
                    j = int(rng.choice(self.m, p=rate_prob[knot]))
                    is_pos = bool(rng.random() < sign_prob[knot, j])
                    level = int(rng.choice(levels, p=mark_prob[knot, j]))
                    signed_amp = self.amplitudes[level] if is_pos else -self.amplitudes[level]
                    self.temporal_event_counts[i, knot, j] += 1
                    if is_pos:
                        self.temporal_positive_counts[i, knot, j] += 1
                    self.temporal_level_counts[i, knot, j, level] += 1
                    self.motor_impulses[i, t] += signed_amp * wiring[j]
            for t in range(self.h):
                if knots <= 1 or self.h <= 1:
                    mu = self.temporal_mean_motor[0]
                else:
                    x = t * (knots - 1.0) / (self.h - 1.0)
                    k0 = min(int(x), knots - 1)
                    if k0 >= knots - 1:
                        mu = self.temporal_mean_motor[-1]
                    else:
                        a = x - k0
                        mu = (1.0 - a) * self.temporal_mean_motor[k0] + a * self.temporal_mean_motor[k0 + 1]
                self.motor_impulses[:, t, :] -= mu[None, :]
            self.motor_filtered.fill(0.0)
            for lag, kval in enumerate(self.kernel[:self.h]):
                self.motor_filtered[:, lag:, :] += float(kval) * self.motor_impulses[:, :self.h-lag, :]

        self.last_event_total = int(total)
        if collect_stats:
            return (self.motor_filtered, self.temporal_event_counts,
                    self.temporal_positive_counts, self.temporal_level_counts,
                    self.last_event_total)
        return self.motor_filtered, None, None, None, self.last_event_total

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
