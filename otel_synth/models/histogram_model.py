"""Histogram-aware profiling and generation.

Histogram metrics consist of linked _bucket, _count, and _sum series.
We fit a continuous distribution from the observed bucket counts and
generate consistent (_bucket, _count, _sum) tuples from it.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar

from otel_synth.config import HistogramFamilyProfile, SeriesStats
from otel_synth.models.series_profile import _compute_stats


def _estimate_observations_per_step(
    count_values: np.ndarray,
) -> np.ndarray:
    """Compute per-step observation counts from cumulative _count values."""
    diffs = np.diff(count_values)
    # Handle resets
    diffs = np.where(diffs < 0, count_values[1:], diffs)
    return np.maximum(diffs, 0.0)


def _fit_distribution_from_buckets(
    le_boundaries: list[float],
    cumulative_counts: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Fit a log-normal distribution from histogram bucket boundaries and counts.

    Args:
        le_boundaries: sorted list of finite le values (no +Inf)
        cumulative_counts: cumulative counts at each boundary (last entry is total count / +Inf)

    Returns:
        (dist_name, params_dict) for the fitted distribution
    """
    if len(le_boundaries) == 0 or cumulative_counts[-1] == 0:
        return "lognorm", {"s": 1.0, "loc": 0.0, "scale": 1.0}

    total = cumulative_counts[-1]
    # Convert cumulative counts to CDF values
    cdf_values = cumulative_counts[:-1] / total  # exclude +Inf (always 1.0)
    boundaries = np.array(le_boundaries, dtype=float)

    # Filter out zero or negative boundaries for lognorm
    valid = boundaries > 0
    if valid.sum() < 2:
        # Fall back to exponential if too few valid boundaries
        median_est = boundaries[len(boundaries) // 2] if len(boundaries) > 0 else 1.0
        return "lognorm", {"s": 1.0, "loc": 0.0, "scale": max(median_est, 0.01)}

    boundaries = boundaries[valid]
    cdf_values = cdf_values[valid]

    # Clamp CDF values to valid range
    cdf_values = np.clip(cdf_values, 0.001, 0.999)

    # Fit log-normal: minimize sum of squared CDF differences
    def objective(params):
        mu, sigma = params
        if sigma <= 0:
            return 1e10
        # Clamp mu to prevent overflow in exp()
        mu_clamped = np.clip(mu, -20, 20)
        scale = np.exp(mu_clamped)
        sigma_safe = min(abs(sigma), 5.0)
        predicted_cdf = sp_stats.lognorm.cdf(boundaries, s=sigma_safe, scale=scale)
        return np.sum((predicted_cdf - cdf_values) ** 2)

    from scipy.optimize import minimize

    # Initial guess from quantiles
    median_idx = np.searchsorted(cdf_values, 0.5)
    if median_idx < len(boundaries):
        mu0 = np.log(max(boundaries[median_idx], 0.01))
    else:
        mu0 = np.log(max(boundaries[-1], 0.01))
    sigma0 = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = minimize(objective, [mu0, sigma0], method="Nelder-Mead")
    mu_fit, sigma_fit = result.x
    sigma_fit = max(abs(sigma_fit), 0.01)
    # Clamp scale to prevent overflow during generation
    scale = np.exp(np.clip(mu_fit, -20, 20))
    # Clamp sigma to prevent extreme tail values
    sigma_fit = min(sigma_fit, 5.0)

    return "lognorm", {"s": sigma_fit, "loc": 0.0, "scale": float(scale)}


def profile_histogram_family(
    family_name: str,
    labels_without_le: dict[str, str],
    le_boundaries: list[float],
    bucket_data: dict[str, np.ndarray],
    count_values: np.ndarray,
    sum_values: np.ndarray,
    timestamps: np.ndarray,
    le_boundary_strings: list[str] | None = None,
) -> HistogramFamilyProfile:
    """Profile a histogram metric family.

    Args:
        family_name: base metric name (without _bucket/_count/_sum)
        labels_without_le: labels dict without the 'le' key
        le_boundaries: sorted list of finite le values
        bucket_data: dict mapping le string -> cumulative count array
        count_values: cumulative _count values over time
        sum_values: cumulative _sum values over time
        timestamps: timestamp array
        le_boundary_strings: original le label strings (preserves formatting)
    """
    # Compute per-step observation counts
    obs_per_step = _estimate_observations_per_step(count_values)
    obs_stats = _compute_stats(obs_per_step) if len(obs_per_step) > 0 else SeriesStats()

    # Get a representative cumulative count snapshot for fitting
    # Use the diff across all timestamps to get total observations per bucket
    # Use original le strings for lookup since bucket_data keys are original strings
    _le_strings = (le_boundary_strings or [str(le) for le in le_boundaries]) + ["+Inf"]
    all_le = sorted(le_boundaries) + [float("inf")]
    cumulative_counts = np.zeros(len(all_le))
    for i, le_str in enumerate(_le_strings):
        if le_str in bucket_data:
            vals = bucket_data[le_str]
            diffs = np.diff(vals)
            diffs = np.where(diffs < 0, vals[1:], diffs)
            cumulative_counts[i] = np.sum(np.maximum(diffs, 0))

    # Fit distribution
    dist_name, dist_params = _fit_distribution_from_buckets(
        le_boundaries, cumulative_counts
    )

    # Preserve original le string formatting
    if not le_boundary_strings:
        le_boundary_strings = [str(le) for le in le_boundaries]

    return HistogramFamilyProfile(
        family_name=family_name,
        labels_without_le=labels_without_le,
        le_boundaries=le_boundaries,
        le_boundary_strings=le_boundary_strings,
        dist_name=dist_name,
        dist_params=dist_params,
        observations_per_step=obs_stats,
        n_timestamps=len(timestamps),
    )


def compute_histogram_delta(
    baseline: HistogramFamilyProfile,
    anomaly: HistogramFamilyProfile,
) -> HistogramFamilyProfile:
    """Compute distribution parameter deltas for anomaly vs baseline."""
    delta = {}
    for key in baseline.dist_params:
        delta[key] = anomaly.dist_params.get(key, 0) - baseline.dist_params.get(key, 0)

    anomaly.delta_dist_params = delta
    if baseline.observations_per_step.mean > 0:
        anomaly.delta_observations_mean = (
            anomaly.observations_per_step.mean - baseline.observations_per_step.mean
        )
    return anomaly


def generate_histogram_family(
    profile: HistogramFamilyProfile,
    n_points: int,
    initial_count: float = 0.0,
    initial_sum: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate synthetic histogram data from a profile.

    Returns:
        Dict with keys: "_count", "_sum", and one entry per le boundary
        (including "+Inf"). Each value is a cumulative array of length n_points.
    """
    if rng is None:
        rng = np.random.default_rng()

    obs_mean = max(profile.observations_per_step.mean, 0.1)
    obs_std = max(profile.observations_per_step.std, 0.1)

    # Build the scipy distribution — clamp params to valid ranges
    dist = sp_stats.lognorm(
        s=max(profile.dist_params.get("s", 1.0), 0.01),
        loc=profile.dist_params.get("loc", 0.0),
        scale=max(profile.dist_params.get("scale", 1.0), 1e-10),
    )

    le_bounds = profile.le_boundaries  # finite only, as floats
    le_strings = profile.le_boundary_strings or [str(le) for le in le_bounds]
    all_le = le_bounds + [float("inf")]
    all_le_strings = le_strings + ["+Inf"]

    # Initialize cumulative arrays
    cum_count = np.zeros(n_points)
    cum_sum = np.zeros(n_points)
    cum_buckets = {le_str: np.zeros(n_points) for le_str in all_le_strings}

    running_count = initial_count
    running_sum = initial_sum
    running_buckets = {k: initial_count for k in cum_buckets}

    # For proper initialization, set finite buckets proportionally
    for le_str, le_val in zip(all_le_strings, all_le):
        if le_val == float("inf"):
            running_buckets[le_str] = running_count
        else:
            cdf_val = dist.cdf(le_val)
            running_buckets[le_str] = running_count * cdf_val

    for t in range(n_points):
        # Draw observation count for this step
        n_obs = max(int(round(rng.normal(obs_mean, obs_std))), 0)

        if n_obs > 0:
            # Sample observations from fitted distribution
            samples = dist.rvs(size=n_obs, random_state=rng)
            samples = np.maximum(samples, 0.0)
            # Clamp to prevent inf/overflow — use 10x the max finite le as upper bound
            max_le = le_bounds[-1] if le_bounds else 1.0
            samples = np.minimum(samples, max_le * 10)
            samples = np.where(np.isfinite(samples), samples, max_le)

            running_count += n_obs
            running_sum += np.sum(samples)

            # Assign to buckets
            for le_str, le_val in zip(all_le_strings, all_le):
                if le_val == float("inf"):
                    running_buckets[le_str] += n_obs
                else:
                    running_buckets[le_str] += int(np.sum(samples <= le_val))

        cum_count[t] = running_count
        cum_sum[t] = running_sum
        for le_str in cum_buckets:
            cum_buckets[le_str][t] = running_buckets[le_str]

    # Enforce monotonicity across le boundaries at each timestamp
    le_keys = list(cum_buckets.keys())
    for t in range(n_points):
        prev = 0.0
        for le_str in le_keys:
            cum_buckets[le_str][t] = max(cum_buckets[le_str][t], prev)
            prev = cum_buckets[le_str][t]
        # +Inf must equal count
        cum_buckets[le_keys[-1]][t] = cum_count[t]

    result = {"_count": cum_count, "_sum": cum_sum}
    result.update(cum_buckets)
    return result
