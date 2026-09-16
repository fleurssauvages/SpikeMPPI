"""Biologically structured motor-pool proposal for Spike-MPPI.

This module intentionally models only a compact set of motor-pool mechanisms
that are useful as an MPPI proposal distribution:

* antagonistic (+/-) motor pools rather than signed spikes,
* common synaptic drive shared by the units in one actuator pool,
* ordered recruitment thresholds (size principle),
* firing-rate coding above threshold,
* refractory gamma-renewal discharge rather than Poisson firing,
* motor-unit-specific twitch strength and contraction time,
* recruitment/derecruitment hysteresis.

On the classic torque Ant, the two pools are reduced to one net actuator command.
On ``ant-bio`` they are emitted separately into antagonistic MuJoCo muscle
actuators, so co-contraction is preserved and affects the physical plant.
"""
from __future__ import annotations

import math
import numpy as np

from .fast_kernels import NUMBA_AVAILABLE, njit


@njit(cache=True, nogil=True, fastmath=False)
def _normal_box_muller(rng) -> float:
    u1 = max(rng.random(), 1e-300)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


@njit(cache=True, nogil=True, fastmath=False)
def _gamma3_interval(rng, mean_s: float) -> float:
    """Gamma(shape=3) draw with the requested mean, using uniforms only."""
    if mean_s <= 0.0:
        return 0.0
    product = max(rng.random() * rng.random() * rng.random(), 1e-300)
    return -(mean_s / 3.0) * math.log(product)


@njit(cache=True, nogil=True, fastmath=False)
def _renewal_interval(rng, rate_hz: float, refractory_s: float) -> float:
    """Shifted gamma renewal interval whose total mean is approximately 1/rate."""
    target = 1.0 / max(rate_hz, 1e-9)
    stochastic_mean = max(target - refractory_s, 1e-6)
    return refractory_s + _gamma3_interval(rng, stochastic_mean)


@njit(cache=True, nogil=True, fastmath=False)
def _bio_motor_pool_kernel(
    rng,
    out,
    dt: float,
    thresholds_on,
    thresholds_off,
    unit_strengths,
    unit_kernels,
    min_rate_hz: float,
    max_rate_hz: float,
    drive_rho: float,
    drive_sigma: float,
    cocontraction_drive: float,
    refractory_s: float,
    separate_antagonists: bool,
):
    """Fill ``out`` with motor-pool twitches.

    When ``separate_antagonists`` is false, opposing pools are subtracted into
    one signed torque-like channel. When true, each pool is emitted as a
    separate nonnegative muscle excitation channel ordered [plus, minus].
    """
    out[:] = 0.0
    n, h, output_nu = out.shape
    nu = output_nu // 2 if separate_antagonists else output_nu
    levels = len(unit_strengths)
    kernel_len = unit_kernels.shape[1]

    event_total = 0
    agonist_events = 0
    antagonist_events = 0
    active_units_accum = 0.0
    coactive_steps = 0
    drive_accum = 0.0
    recruit_transitions = 0
    derecruit_transitions = 0
    min_isi_s = 1e30

    active = np.zeros((2, levels), dtype=np.uint8)
    countdown = np.zeros((2, levels), dtype=np.float64)
    last_spike_t = np.full((2, levels), -1e30, dtype=np.float64)
    innovation_scale = drive_sigma * math.sqrt(max(0.0, 1.0 - drive_rho * drive_rho))

    for i in range(n):
        for j in range(nu):
            active[:] = 0
            countdown[:] = 0.0
            last_spike_t[:] = -1e30
            # Stationary zero-mean common command drive. Positive command excites
            # the agonist pool, negative command excites the antagonist pool.
            command = drive_sigma * _normal_box_muller(rng)

            for t in range(h):
                command = drive_rho * command + innovation_scale * _normal_box_muller(rng)
                plus_drive = cocontraction_drive + max(command, 0.0)
                minus_drive = cocontraction_drive + max(-command, 0.0)
                if plus_drive > 1.0:
                    plus_drive = 1.0
                if minus_drive > 1.0:
                    minus_drive = 1.0
                drive_accum += 0.5 * (plus_drive + minus_drive)

                plus_active = 0
                minus_active = 0
                now_s = t * dt
                for pool in range(2):
                    drive = plus_drive if pool == 0 else minus_drive
                    sign = 1.0 if pool == 0 else -1.0
                    pool_active = 0
                    for u in range(levels):
                        was_active = active[pool, u] != 0
                        is_active = was_active
                        if not was_active and drive >= thresholds_on[u]:
                            is_active = True
                            active[pool, u] = 1
                            recruit_transitions += 1
                            # Desynchronise units when a pool is recruited rather
                            # than emitting an artificial simultaneous volley.
                            frac = (drive - thresholds_on[u]) / max(1.0 - thresholds_on[u], 1e-9)
                            frac = min(max(frac, 0.0), 1.0)
                            rate = min_rate_hz + (max_rate_hz - min_rate_hz) * frac
                            countdown[pool, u] = rng.random() * _renewal_interval(
                                rng, rate, refractory_s
                            )
                        elif was_active and drive <= thresholds_off[u]:
                            is_active = False
                            active[pool, u] = 0
                            countdown[pool, u] = 0.0
                            last_spike_t[pool, u] = -1e30
                            derecruit_transitions += 1

                        if not is_active:
                            continue
                        pool_active += 1
                        frac = (drive - thresholds_on[u]) / max(1.0 - thresholds_on[u], 1e-9)
                        frac = min(max(frac, 0.0), 1.0)
                        rate = min_rate_hz + (max_rate_hz - min_rate_hz) * frac
                        countdown[pool, u] -= dt
                        if countdown[pool, u] <= 0.0:
                            # A unit may only contribute a positive twitch to its
                            # muscle pool; antagonism is introduced at pool level.
                            strength = unit_strengths[u]
                            kmax = min(kernel_len, h - t)
                            for lag in range(kmax):
                                value = strength * unit_kernels[u, lag]
                                if separate_antagonists:
                                    out[i, t + lag, 2 * j + pool] += value
                                else:
                                    out[i, t + lag, j] += sign * value
                            event_total += 1
                            if pool == 0:
                                agonist_events += 1
                            else:
                                antagonist_events += 1
                            if last_spike_t[pool, u] > -1e20:
                                isi = now_s - last_spike_t[pool, u]
                                if isi < min_isi_s:
                                    min_isi_s = isi
                            last_spike_t[pool, u] = now_s
                            countdown[pool, u] += _renewal_interval(
                                rng, rate, refractory_s
                            )
                    if pool == 0:
                        plus_active = pool_active
                    else:
                        minus_active = pool_active

                active_units_accum += plus_active + minus_active
                if plus_active > 0 and minus_active > 0:
                    coactive_steps += 1

    denom_steps = max(n * h * nu, 1)
    mean_active_units = active_units_accum / denom_steps
    coactivation_fraction = coactive_steps / denom_steps
    mean_drive = drive_accum / denom_steps
    if min_isi_s > 1e20:
        min_isi_s = math.nan
    return (
        event_total,
        agonist_events,
        antagonist_events,
        mean_active_units,
        coactivation_fraction,
        mean_drive,
        recruit_transitions,
        derecruit_transitions,
        min_isi_s,
    )


def make_motor_unit_pool(levels: int):
    """Return exponentially spaced recruitment thresholds and ordered strengths.

    The threshold spacing gives many low-threshold units and progressively wider
    gaps toward the high-threshold end of the pool, a compact size-principle
    approximation.  Strengths remain a separate, fixed ordered hierarchy.
    """
    levels = max(1, int(levels))
    threshold_min = 0.10
    threshold_max = 0.85
    if levels == 1:
        on = np.array([threshold_min], dtype=np.float64)
    else:
        phase = np.linspace(0.0, 1.0, levels, dtype=np.float64)
        on = threshold_min * np.exp(phase * math.log(threshold_max / threshold_min))
    # Lower derecruitment threshold models simple recruitment hysteresis.
    off = np.maximum(0.0, on - (0.06 + 0.08 * on))
    raw_strength = np.arange(1, levels + 1, dtype=np.float64)
    strengths = raw_strength / float(np.sum(raw_strength))
    return on, off, strengths


def make_heterogeneous_twitch_bank(
    levels: int,
    dt: float,
    base_rise_s: float,
    base_decay_s: float,
    duration_s: float,
):
    """Slow/weak low-threshold units and faster high-threshold units."""
    levels = max(1, int(levels))
    steps = max(1, int(math.ceil(duration_s / dt)))
    t = np.arange(steps, dtype=np.float64) * float(dt)
    kernels = np.zeros((levels, steps), dtype=np.float64)
    rise = np.empty(levels, dtype=np.float64)
    decay = np.empty(levels, dtype=np.float64)
    for u in range(levels):
        phase = 0.0 if levels == 1 else u / float(levels - 1)
        # Low-threshold units are slower; high-threshold units are faster.
        rise[u] = base_rise_s * (1.30 - 0.55 * phase)
        decay[u] = base_decay_s * (1.45 - 0.65 * phase)
        k = np.exp(-t / decay[u]) - np.exp(-t / rise[u])
        k[0] = 0.0
        peak = float(np.max(k))
        if peak > 0.0:
            k /= peak
        kernels[u] = k
    return kernels, rise, decay


class BioMotorPoolSampler:
    """Reusable antagonistic motor-pool proposal sampler."""

    def __init__(
        self,
        *,
        n: int,
        h: int,
        nu: int,
        dt: float,
        levels: int,
        base_rate_hz: float,
        twitch_rise_s: float,
        twitch_decay_s: float,
        twitch_duration_s: float,
        drive_sigma: float = 0.26,
        drive_tau_s: float = 0.10,
        separate_antagonists: bool = False,
        cocontraction_drive: float | None = None,
    ) -> None:
        if not NUMBA_AVAILABLE:
            raise RuntimeError("spike-bio requires Numba")
        self.n = int(n)
        self.h = int(h)
        self.motor_pool_count = int(nu)
        self.separate_antagonists = bool(separate_antagonists)
        self.nu = 2 * self.motor_pool_count if self.separate_antagonists else self.motor_pool_count
        self.dt = float(dt)
        self.levels = max(1, int(levels))
        self.base_rate_hz = float(base_rate_hz)
        self.thresholds_on, self.thresholds_off, self.unit_strengths = make_motor_unit_pool(
            self.levels
        )
        (
            self.unit_kernels,
            self.unit_rise_s,
            self.unit_decay_s,
        ) = make_heterogeneous_twitch_bank(
            self.levels,
            self.dt,
            float(twitch_rise_s),
            float(twitch_decay_s),
            float(twitch_duration_s),
        )
        self.out = np.zeros((self.n, self.h, self.nu), dtype=np.float64)

        # Fixed physiological proposal constants. They are deliberately not CLI
        # knobs: the research comparison is the complete motor-pool mechanism.
        self.max_rate_hz = 3.0 * self.base_rate_hz
        self.drive_tau_s = float(drive_tau_s)
        if self.drive_tau_s <= 0.0:
            raise ValueError("drive_tau_s must be positive")
        self.drive_rho = math.exp(-self.dt / self.drive_tau_s)
        self.drive_sigma = float(drive_sigma)
        if self.drive_sigma <= 0.0:
            raise ValueError("drive_sigma must be positive")
        # Net-torque actuators cannot express stiffness from co-contraction, so
        # their baseline is zero. The muscle plant gets a small shared background
        # drive; hysteresis then allows physically meaningful pool overlap.
        self.cocontraction_drive = (
            0.08 if self.separate_antagonists and cocontraction_drive is None
            else float(0.0 if cocontraction_drive is None else cocontraction_drive)
        )
        if not 0.0 <= self.cocontraction_drive <= 1.0:
            raise ValueError("cocontraction_drive must lie in [0,1]")
        # The real absolute refractory period is shorter than a 20 ms control bin;
        # keep it explicit in the renewal model even though binning limits temporal
        # resolution. Max rate is low enough that same-bin doublets are not needed.
        self.refractory_s = 0.005
        self.last_stats: dict[str, float] = {}

    def sample(self, rng):
        stats = _bio_motor_pool_kernel(
            rng,
            self.out,
            self.dt,
            self.thresholds_on,
            self.thresholds_off,
            self.unit_strengths,
            self.unit_kernels,
            self.base_rate_hz,
            self.max_rate_hz,
            self.drive_rho,
            self.drive_sigma,
            self.cocontraction_drive,
            self.refractory_s,
            self.separate_antagonists,
        )
        (
            event_total,
            agonist_events,
            antagonist_events,
            mean_active_units,
            coactivation_fraction,
            mean_drive,
            recruit_transitions,
            derecruit_transitions,
            min_isi_s,
        ) = stats
        self.last_stats = {
            "spike_events_mean": float(event_total) / max(self.n, 1),
            "bio_agonist_events_mean": float(agonist_events) / max(self.n, 1),
            "bio_antagonist_events_mean": float(antagonist_events) / max(self.n, 1),
            "bio_mean_active_units": float(mean_active_units),
            "bio_coactivation_fraction": float(coactivation_fraction),
            "bio_mean_common_drive": float(mean_drive),
            "bio_recruit_transitions_mean": float(recruit_transitions) / max(self.n, 1),
            "bio_derecruit_transitions_mean": float(derecruit_transitions) / max(self.n, 1),
            "bio_min_observed_isi_s": float(min_isi_s),
            "bio_min_rate_hz": float(self.base_rate_hz),
            "bio_max_rate_hz": float(self.max_rate_hz),
            "bio_refractory_s": float(self.refractory_s),
            "bio_drive_sigma": float(self.drive_sigma),
            "bio_drive_tau_s": float(self.drive_tau_s),
            "bio_cocontraction_drive": float(self.cocontraction_drive),
            "bio_separate_antagonists": float(self.separate_antagonists),
        }
        return self.out, self.last_stats

    def estimate_unit_rms(self, *, seed: int = 918273, n_calibration: int = 256) -> float:
        """Deterministic calibration independent of the controller's RNG stream."""
        temp_nu = 2 if self.separate_antagonists else 1
        temp = np.zeros((int(n_calibration), self.h, temp_nu), dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        _bio_motor_pool_kernel(
            rng,
            temp,
            self.dt,
            self.thresholds_on,
            self.thresholds_off,
            self.unit_strengths,
            self.unit_kernels,
            self.base_rate_hz,
            self.max_rate_hz,
            self.drive_rho,
            self.drive_sigma,
            self.cocontraction_drive,
            self.refractory_s,
            self.separate_antagonists,
        )
        if self.separate_antagonists:
            # Runtime Ant-Bio proposals are centered across the rollout population.
            # Match RMS per physical muscle channel so --joint-noise has the same
            # 16-D actuator-space meaning for standard, spike and spike-bio.
            temp -= np.mean(temp, axis=0, keepdims=True)
            return float(np.sqrt(np.mean(temp * temp)))
        return float(np.sqrt(np.mean(temp * temp)))
