from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class LBPSResult:
    temperature: float
    alpha: float
    ess: float
    score: float
    finite_count: int
    reward_norm: float
    expected_return: float


def _score(costs: np.ndarray, alpha: float, delta: float, rho: float, reward_norm: float):
    finite = np.isfinite(costs)
    if not np.any(finite):
        return -math.inf, 0.0, -math.inf
    c = costs[finite]
    w = np.exp(np.clip(-alpha * (c - rho), -745.0, 0.0))
    sw = float(np.sum(w))
    sw2 = float(np.sum(w * w))
    if sw <= 0.0 or sw2 <= 0.0:
        return -math.inf, 0.0, -math.inf
    ess = sw * sw / sw2
    expected_return = float(np.sum(w * (-c)) / sw)
    penalty = reward_norm * math.sqrt((1.0 - delta) / (delta * max(ess, 1e-300)))
    return expected_return - penalty, ess, expected_return


def optimize_lbps_temperature(costs, *, delta: float, fallback_temperature: float, iterations: int = 32) -> LBPSResult:
    """Watson-Peters LBPS temperature optimization, matching the reference controller."""
    values = np.asarray(costs, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    fallback_alpha = 1.0 / max(float(fallback_temperature), 1e-300)
    if finite.size == 0:
        return LBPSResult(float(fallback_temperature), fallback_alpha, 0.0, -math.inf, 0, 0.0, -math.inf)
    rho = float(np.min(finite))
    spread = float(np.max(finite) - rho)
    reward_norm = float(np.max(np.abs(finite)))
    if finite.size <= 1 or spread <= 1e-12 or reward_norm <= 1e-15:
        score, ess, er = _score(values, fallback_alpha, delta, rho, reward_norm)
        return LBPSResult(float(fallback_temperature), fallback_alpha, ess, score, int(finite.size), reward_norm, er)

    score0, _, _ = _score(values, 0.0, delta, rho, reward_norm)
    alpha1 = 1.0 / spread
    score1, _, _ = _score(values, alpha1, delta, rho, reward_norm)
    left, right = 0.0, alpha1
    if score1 > score0:
        prevprev, prev, prev_score = 0.0, alpha1, score1
        bracketed = False
        for _ in range(40):
            nxt = prev * 2.0
            nxt_score, _, _ = _score(values, nxt, delta, rho, reward_norm)
            if nxt_score <= prev_score:
                left, right = prevprev, nxt
                bracketed = True
                break
            prevprev, prev, prev_score = prev, nxt, nxt_score
        if not bracketed:
            score, ess, er = _score(values, prev, delta, rho, reward_norm)
            return LBPSResult(1.0 / max(prev, 1e-300), prev, ess, score, int(finite.size), reward_norm, er)

    golden = 0.6180339887498949
    a, b = left, right
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc, _, _ = _score(values, c, delta, rho, reward_norm)
    fd, _, _ = _score(values, d, delta, rho, reward_norm)
    for _ in range(max(8, int(iterations))):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - golden * (b - a)
            fc, _, _ = _score(values, c, delta, rho, reward_norm)
        else:
            a, c, fc = c, d, fd
            d = a + golden * (b - a)
            fd, _, _ = _score(values, d, delta, rho, reward_norm)
    alpha = 0.5 * (a + b)
    score, ess, er = _score(values, alpha, delta, rho, reward_norm)
    if score0 >= score:
        alpha = 0.0
        score, ess, er = _score(values, alpha, delta, rho, reward_norm)
    if alpha <= 1e-14 / spread:
        alpha = 1e-14 / spread
        score, ess, er = _score(values, alpha, delta, rho, reward_norm)
    return LBPSResult(1.0 / alpha, alpha, ess, score, int(finite.size), reward_norm, er)


def weighted_control_sequence(costs: np.ndarray, controls: np.ndarray, temperature: float) -> np.ndarray:
    costs = np.asarray(costs, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    finite = np.isfinite(costs)
    if not np.any(finite):
        return np.mean(controls, axis=0)
    rho = float(np.min(costs[finite]))
    w = np.zeros(len(costs), dtype=np.float64)
    w[finite] = np.exp(np.clip(-(costs[finite] - rho) / max(float(temperature), 1e-300), -745.0, 0.0))
    total = float(np.sum(w))
    if total <= 1e-12:
        return np.mean(controls[finite], axis=0)
    return np.einsum("n,nhu->hu", w / total, controls)
