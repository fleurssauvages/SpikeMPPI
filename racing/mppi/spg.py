from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass
class SPGFactors:
    task_factor: np.ndarray      # [H, nu, 2]
    null_projector: np.ndarray   # [H, nu, nu] (kept for diagnostics)
    corrected_covariance: np.ndarray  # [H, 2, 2]
    displacement: np.ndarray     # [H, 2]
    pseudoinverse: np.ndarray | None = None  # [H, nu, 2]
    jacobian: np.ndarray | None = None       # [H, 2, nu]


def damped_pseudoinverse(jacobian: np.ndarray, damping: float) -> np.ndarray:
    j = np.asarray(jacobian, dtype=np.float64)
    return j.T @ np.linalg.inv(j @ j.T + float(damping) * np.eye(j.shape[0]))


def center_correct_prior_covariance(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    nominal_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Second moment of the spatial prior about the nominal proposal center.

    This preserves the reference controller's Sigma + d d^T correction.
    """
    d = np.asarray(prior_mean, dtype=np.float64) - np.asarray(nominal_position, dtype=np.float64)
    cov = np.asarray(prior_covariance, dtype=np.float64)
    corrected = 0.5 * (cov + cov.T) + np.outer(d, d)
    return corrected, d


def build_spg_factors(
    jacobians: np.ndarray,
    prior_means: np.ndarray,
    prior_covariances: np.ndarray,
    nominal_endpoints: np.ndarray,
    *,
    damping: float = 1e-6,
    covariance_jitter: float = 1e-8,
) -> SPGFactors:
    """Vectorized SPG factor construction over the complete horizon."""
    J = np.asarray(jacobians, dtype=np.float64)
    means = np.asarray(prior_means, dtype=np.float64)
    covs = np.asarray(prior_covariances, dtype=np.float64)
    endpoints = np.asarray(nominal_endpoints, dtype=np.float64)
    h, _, nu = J.shape

    displacement = means - endpoints
    corrected = 0.5 * (covs + np.swapaxes(covs, -1, -2))
    corrected = corrected + np.einsum("hi,hj->hij", displacement, displacement)
    eye2 = np.eye(2, dtype=np.float64)
    corrected = corrected + float(covariance_jitter) * eye2[None, :, :]

    # np.linalg.eigh and inv are batched for arrays with leading dimensions,
    # avoiding H Python iterations over tiny 2x2 matrices.
    eigval, eigvec = np.linalg.eigh(corrected)
    eigval = np.maximum(eigval, float(covariance_jitter))
    root = np.matmul(
        eigvec * np.sqrt(eigval)[:, None, :],
        np.swapaxes(eigvec, -1, -2),
    )

    jt = np.swapaxes(J, 1, 2)
    gram = np.matmul(J, jt) + float(damping) * eye2[None, :, :]
    pseudoinverse = np.matmul(jt, np.linalg.inv(gram))
    task_factor = np.matmul(pseudoinverse, root)
    null_projector = (
        np.eye(nu, dtype=np.float64)[None, :, :] - np.matmul(pseudoinverse, J)
    )
    return SPGFactors(
        task_factor, null_projector, corrected, displacement, pseudoinverse, J.copy()
    )


@lru_cache(maxsize=32)
def _temporal_smoothing_matrix(h: int, rho: float) -> np.ndarray:
    r = float(np.clip(rho, 0.0, 0.999999))
    beta = float(np.sqrt(max(0.0, 1.0 - r * r)))
    rows = np.arange(int(h), dtype=np.int64)[:, None]
    cols = np.arange(int(h), dtype=np.int64)[None, :]
    lag = rows - cols
    matrix = np.zeros((int(h), int(h)), dtype=np.float64)
    valid = lag >= 0
    matrix[valid] = beta * np.power(r, lag[valid])
    # The original recurrence starts with y_0=x_0 (not beta*x_0).
    matrix[:, 0] = np.power(r, np.arange(int(h), dtype=np.float64))
    matrix.setflags(write=False)
    return matrix


def _temporal_smooth(noise: np.ndarray, rho: float) -> np.ndarray:
    """Apply the controller's AR(1) smoothing without a Python horizon loop."""
    x = np.asarray(noise, dtype=np.float64)
    h = int(x.shape[1])
    r = float(np.clip(rho, 0.0, 0.999999))
    if h <= 1 or r <= 0.0:
        return x
    return np.einsum(
        "tk,nku->ntu", _temporal_smoothing_matrix(h, r), x, optimize=True
    )

def sample_joint_noise(
    rng: np.random.Generator,
    factors: SPGFactors,
    *,
    n: int,
    default_std: np.ndarray,
    temporal_smoothing: float,
    null_std_scale: float = 0.15,
    spg_mix: float = 0.9,
) -> np.ndarray:
    """Sample SPG joint noise in task and null spaces.

    ``spg_mix`` blends SPG exploration with the robot's default actuator noise,
    analogous to retaining an uninformed proposal component in the reference
    controller.
    """
    h, nu, _ = factors.task_factor.shape
    count = int(n)
    mix = float(np.clip(spg_mix, 0.0, 1.0))
    std = np.asarray(default_std, dtype=np.float64)[None, None, :]

    if mix > 0.0:
        z_task = rng.standard_normal((count, h, 2))
        task = np.einsum("huj,nhj->nhu", factors.task_factor, z_task)
        if float(null_std_scale) > 0.0:
            z_null = rng.standard_normal((count, h, nu))
            if nu >= 12 and factors.pseudoinverse is not None and factors.jacobian is not None:
                # Apply (I - J^dagger J)z in factorized form for larger actuator sets.
                task_coords = np.einsum("hju,nhu->nhj", factors.jacobian, z_null)
                correction = np.einsum("huj,nhj->nhu", factors.pseudoinverse, task_coords)
                null = z_null - correction
            else:
                null = np.einsum("huv,nhv->nhu", factors.null_projector, z_null)
            task = task + null * std * float(null_std_scale)
        noise = np.sqrt(mix) * task
    else:
        noise = np.zeros((count, h, nu), dtype=np.float64)

    if mix < 1.0:
        z_default = rng.standard_normal((count, h, nu))
        noise = noise + np.sqrt(max(0.0, 1.0 - mix)) * (z_default * std)

    return _temporal_smooth(noise, temporal_smoothing)
