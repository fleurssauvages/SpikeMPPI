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


@dataclass
class SPGTimeDependentFactors:
    """SPG proposal factors built from a future task-space sensitivity window.

    For each control row ``k`` we keep the forward sensitivity

        H_k = d [y_{k+1}, ..., y_{k+W}] / d z_k

    and form its damped local inverse

        G_k ~= d z_k / d [y_{k+1}, ..., y_{k+W}].

    ``task_factor[k]`` maps standard-normal task noise over all retained future
    task points into actuator noise at control row ``k``.  Because every future
    prior covariance is kept separately before the mapping, the induced
    actuator variance can change along the MPC horizon instead of being based on
    a single terminal endpoint.
    """

    task_factor: np.ndarray          # [H, nu, 2W]
    null_projector: np.ndarray       # [H, nu, nu]
    corrected_covariance: np.ndarray # [H, W, 2, 2]
    displacement: np.ndarray         # [H, W, 2]
    inverse_sensitivity: np.ndarray  # [H, nu, 2W] ~= dz / dY
    forward_sensitivity: np.ndarray  # [H, W, 2, nu] = dY / dz
    valid_lengths: np.ndarray        # [H]


def build_spg_time_dependent_factors(
    sensitivities: np.ndarray,
    prior_means: np.ndarray,
    prior_covariances: np.ndarray,
    nominal_future: np.ndarray,
    *,
    damping: float = 1e-6,
    covariance_jitter: float = 1e-8,
) -> SPGTimeDependentFactors:
    """Construct time-dependent SPG factors from future task sensitivities.

    ``sensitivities[k, ell]`` is the finite-difference forward Jacobian

        d y_{k+ell+1} / d z_k,     ell = 0, ..., W-1.

    For the valid future points of each horizon row, the forward Jacobians are
    stacked into ``H_k``.  Its damped pseudoinverse ``G_k`` maps the block
    diagonal spatial second moment into actuator space:

        C_k = blockdiag(Sigma_{k,ell} + d_{k,ell} d_{k,ell}^T)
        delta z_k = G_k C_k^(1/2) xi.

    This preserves the original SPG zero-mean proposal: displacement changes
    the exploration variance through the second moment, but does not add a
    deterministic control correction.
    """
    forward = np.asarray(sensitivities, dtype=np.float64)
    means = np.asarray(prior_means, dtype=np.float64)
    covs = np.asarray(prior_covariances, dtype=np.float64)
    nominal = np.asarray(nominal_future, dtype=np.float64)

    if forward.ndim != 4 or forward.shape[2] != 2:
        raise ValueError("sensitivities must have shape [H,W,2,nu]")
    h, window, _, nu = forward.shape
    if means.shape != (h, window, 2):
        raise ValueError(f"prior_means must have shape {(h, window, 2)}")
    if covs.shape != (h, window, 2, 2):
        raise ValueError(f"prior_covariances must have shape {(h, window, 2, 2)}")
    if nominal.shape != (h, window, 2):
        raise ValueError(f"nominal_future must have shape {(h, window, 2)}")

    damping = max(float(damping), 0.0)
    jitter = max(float(covariance_jitter), 0.0)
    eye2 = np.eye(2, dtype=np.float64)
    eyeu = np.eye(nu, dtype=np.float64)

    displacement = means - nominal
    corrected = 0.5 * (covs + np.swapaxes(covs, -1, -2))
    corrected = corrected + np.einsum(
        "hwi,hwj->hwij", displacement, displacement, optimize=True
    )
    corrected = corrected + jitter * eye2[None, None, :, :]

    # Symmetric roots for every future 2-D task covariance.  We keep the roots
    # separate rather than assembling H large block matrices.
    eigval, eigvec = np.linalg.eigh(corrected)
    eigval = np.maximum(eigval, max(jitter, 1e-15))
    roots = np.matmul(
        eigvec * np.sqrt(eigval)[..., None, :],
        np.swapaxes(eigvec, -1, -2),
    )

    valid_lengths = np.minimum(window, h - np.arange(h, dtype=np.int64))
    valid_mask = np.arange(window, dtype=np.int64)[None, :] < valid_lengths[:, None]
    masked_forward = forward * valid_mask[:, :, None, None]
    Hstack = masked_forward.reshape(h, 2 * window, nu)
    obs_dim = 2 * window

    # Batched Tikhonov pseudoinverse.  Padding invalid tail observations with
    # zero Jacobian rows is exact under damping: the corresponding columns of G
    # stay zero, while all H horizon rows can be solved in one NumPy call.
    if damping <= 0.0:
        inverse_sensitivity = np.linalg.pinv(Hstack)
    elif obs_dim <= nu:
        gram = np.matmul(Hstack, np.swapaxes(Hstack, -1, -2))
        gram = gram + damping * np.eye(obs_dim, dtype=np.float64)[None, :, :]
        solved = np.linalg.solve(gram, Hstack)
        inverse_sensitivity = np.swapaxes(solved, -1, -2)
    else:
        ht = np.swapaxes(Hstack, -1, -2)
        gram = np.matmul(ht, Hstack) + damping * eyeu[None, :, :]
        inverse_sensitivity = np.linalg.solve(gram, ht)

    null_projector = (
        eyeu[None, :, :] - np.matmul(inverse_sensitivity, Hstack)
    )

    # Apply each 2x2 spatial root to its own two columns of G, equivalent to
    # G @ blockdiag(root_0, ..., root_W-1) without constructing block matrices.
    Gblocks = inverse_sensitivity.reshape(h, nu, window, 2)
    factor_blocks = np.einsum(
        "huwi,hwij->huwj", Gblocks, roots, optimize=True
    )
    task_factor = factor_blocks.reshape(h, nu, 2 * window)

    return SPGTimeDependentFactors(
        task_factor=task_factor,
        null_projector=null_projector,
        corrected_covariance=corrected,
        displacement=displacement,
        inverse_sensitivity=inverse_sensitivity,
        forward_sensitivity=forward.copy(),
        valid_lengths=valid_lengths,
    )


def sample_time_dependent_joint_noise(
    rng: np.random.Generator,
    factors: SPGTimeDependentFactors,
    *,
    n: int,
    default_std: np.ndarray,
    temporal_smoothing: float,
    null_std_scale: float = 0.15,
    spg_mix: float = 0.9,
) -> np.ndarray:
    """Sample the time-dependent zero-mean SPG proposal.

    Temporal smoothing is deliberately kept identical to the existing SPG
    variant: construct the task/null-space actuator perturbation first, then
    apply the same AR(1) filter.  The only algorithmic change in this variant is
    therefore the time-dependent G[k] covariance projection.
    """
    h, nu, task_dim = factors.task_factor.shape
    count = int(n)
    mix = float(np.clip(spg_mix, 0.0, 1.0))
    std = np.asarray(default_std, dtype=np.float64)[None, None, :]

    if mix > 0.0:
        z_task = rng.standard_normal((count, h, task_dim))
        task = np.einsum(
            "hud,nhd->nhu", factors.task_factor, z_task, optimize=True
        )

        if float(null_std_scale) > 0.0:
            z_null = rng.standard_normal((count, h, nu))
            null = np.einsum(
                "huv,nhv->nhu", factors.null_projector, z_null, optimize=True
            )
            task = task + null * std * float(null_std_scale)
        noise = np.sqrt(mix) * task
    else:
        noise = np.zeros((count, h, nu), dtype=np.float64)

    if mix < 1.0:
        z_default = rng.standard_normal((count, h, nu))
        noise = noise + np.sqrt(max(0.0, 1.0 - mix)) * (z_default * std)

    return _temporal_smooth(noise, temporal_smoothing)
