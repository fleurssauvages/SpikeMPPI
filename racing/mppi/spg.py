from __future__ import annotations

from dataclasses import dataclass
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
    """Generalize SPG from 2 controls to an arbitrary number of MuJoCo actuators.

    For J_t in R^(2 x nu), a spatial factor L_p is mapped to joint controls as
    J_t^dagger L_p. Null-space exploration is kept separately, avoiding a dense
    nu x nu covariance Cholesky for every horizon step.
    """
    J = np.asarray(jacobians, dtype=np.float64)
    means = np.asarray(prior_means, dtype=np.float64)
    covs = np.asarray(prior_covariances, dtype=np.float64)
    endpoints = np.asarray(nominal_endpoints, dtype=np.float64)
    h, _, nu = J.shape
    task_factor = np.zeros((h, nu, 2), dtype=np.float64)
    null_projector = np.zeros((h, nu, nu), dtype=np.float64)
    corrected = np.zeros((h, 2, 2), dtype=np.float64)
    displacement = np.zeros((h, 2), dtype=np.float64)
    pseudoinverse = np.zeros((h, nu, 2), dtype=np.float64)

    eye_u = np.eye(nu, dtype=np.float64)
    for t in range(h):
        sigma_hat, d = center_correct_prior_covariance(means[t], covs[t], endpoints[t])
        sigma_hat = sigma_hat + float(covariance_jitter) * np.eye(2)
        eigval, eigvec = np.linalg.eigh(0.5 * (sigma_hat + sigma_hat.T))
        eigval = np.maximum(eigval, float(covariance_jitter))
        root = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T
        pinv = damped_pseudoinverse(J[t], damping)
        pseudoinverse[t] = pinv
        task_factor[t] = pinv @ root
        null_projector[t] = eye_u - pinv @ J[t]
        corrected[t] = sigma_hat
        displacement[t] = d
    return SPGFactors(task_factor, null_projector, corrected, displacement, pseudoinverse, J.copy())


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
    z_task = rng.standard_normal((count, h, 2))
    z_null = rng.standard_normal((count, h, nu))
    z_default = rng.standard_normal((count, h, nu))

    task = np.einsum("huj,nhj->nhu", factors.task_factor, z_task)
    if nu >= 12 and factors.pseudoinverse is not None and factors.jacobian is not None:
        # Apply (I - J^dagger J)z in factorized form for larger actuator sets.
        # For small nu (e.g. Ant=8), NumPy's dense einsum is actually faster.
        task_coords = np.einsum("hju,nhu->nhj", factors.jacobian, z_null)
        correction = np.einsum("huj,nhj->nhu", factors.pseudoinverse, task_coords)
        null = z_null - correction
    else:
        null = np.einsum("huv,nhv->nhu", factors.null_projector, z_null)
    null *= np.asarray(default_std, dtype=np.float64)[None, None, :] * float(null_std_scale)
    default = z_default * np.asarray(default_std, dtype=np.float64)[None, None, :]

    mix = float(np.clip(spg_mix, 0.0, 1.0))
    noise = np.sqrt(mix) * (task + null) + np.sqrt(max(0.0, 1.0 - mix)) * default

    rho = float(np.clip(temporal_smoothing, 0.0, 0.999999))
    beta = np.sqrt(max(0.0, 1.0 - rho * rho))
    for t in range(1, h):
        noise[:, t] = rho * noise[:, t - 1] + beta * noise[:, t]
    return noise
