"""Within-service correlation modeling using Ledoit-Wolf shrinkage."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.covariance import LedoitWolf

from otel_synth.config import ServiceCorrelation

logger = logging.getLogger(__name__)


def compute_service_correlation(
    service_name: str,
    series_keys: list[str],
    series_data: dict[str, np.ndarray],
) -> ServiceCorrelation:
    """Compute shrunk covariance matrix for all series within a service.

    Args:
        service_name: name of the service
        series_keys: ordered list of series key strings
        series_data: dict mapping series key -> value array (same length per series)

    Returns:
        ServiceCorrelation with the shrunk covariance matrix
    """
    if len(series_keys) < 2:
        return ServiceCorrelation(
            service_name=service_name,
            series_keys=series_keys,
            covariance_matrix=[],
        )

    # Build the data matrix: rows = timestamps, columns = series
    n_points = min(len(series_data[k]) for k in series_keys)
    data_matrix = np.column_stack([series_data[k][:n_points] for k in series_keys])

    # Handle constant columns (zero variance) — replace with small noise
    col_stds = np.std(data_matrix, axis=0)
    for i in range(data_matrix.shape[1]):
        if col_stds[i] < 1e-10:
            data_matrix[:, i] += np.random.default_rng(42).normal(0, 1e-8, n_points)

    try:
        lw = LedoitWolf()
        lw.fit(data_matrix)
        cov = lw.covariance_
    except Exception as e:
        logger.warning(f"Ledoit-Wolf failed for {service_name}: {e}, using diagonal")
        cov = np.diag(np.var(data_matrix, axis=0))

    return ServiceCorrelation(
        service_name=service_name,
        series_keys=series_keys,
        covariance_matrix=cov.tolist(),
    )


def generate_correlated_innovations(
    correlation: ServiceCorrelation,
    n_points: int,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate correlated standard-normal innovations for a service.

    Normalizes the stored covariance to a correlation matrix, then uses
    Cholesky decomposition to produce correlated standard normal samples.
    Each series gets innovations with mean=0, std=1, but cross-series
    correlation structure is preserved. Caller applies marginal
    transformation (AR(1) filtering, scaling by series stats).

    Returns dict mapping series key -> standard normal innovation array.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_series = len(correlation.series_keys)
    if n_series == 0:
        return {}

    if not correlation.covariance_matrix or n_series < 2:
        return {k: rng.standard_normal(n_points) for k in correlation.series_keys}

    cov = np.array(correlation.covariance_matrix)

    # Normalize covariance → correlation matrix
    diag = np.sqrt(np.diag(cov))
    diag = np.where(diag < 1e-10, 1.0, diag)
    corr = cov / np.outer(diag, diag)
    # Ensure diagonal is exactly 1
    np.fill_diagonal(corr, 1.0)

    # Ensure positive semi-definite via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-10)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize diagonal after reconstruction
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    try:
        L = np.linalg.cholesky(corr)
        z = rng.standard_normal((n_points, n_series))
        samples = z @ L.T
    except np.linalg.LinAlgError:
        logger.warning("Cholesky decomposition failed, using independent samples")
        samples = rng.standard_normal((n_points, n_series))

    return {
        key: samples[:, i] for i, key in enumerate(correlation.series_keys)
    }
