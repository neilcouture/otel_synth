"""Per-series statistical profiling and generation for gauges and counters."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from otel_synth.config import SeriesProfile, SeriesStats, MetricType


def _compute_stats(values: np.ndarray) -> SeriesStats:
    """Compute statistical summary of a 1-D array."""
    if len(values) < 2:
        return SeriesStats(
            mean=float(values[0]) if len(values) == 1 else 0.0,
            n_points=len(values),
        )

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # Autocorrelation at lag 1
    if std > 0 and len(values) > 2:
        centered = values - mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            autocorr = float(np.corrcoef(centered[:-1], centered[1:])[0, 1])
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0

    # Linear trend slope
    if len(values) > 2:
        x = np.arange(len(values))
        slope = float(np.polyfit(x, values, 1)[0])
    else:
        slope = 0.0

    # Skewness and kurtosis — require meaningful variance to avoid precision loss
    coeff_of_var = std / max(abs(mean), 1e-12)
    if len(values) > 3 and std > 1e-12 and coeff_of_var > 1e-9:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skew = float(sp_stats.skew(values))
            kurt = float(sp_stats.kurtosis(values))
        if not np.isfinite(skew):
            skew = 0.0
        if not np.isfinite(kurt):
            kurt = 0.0
    else:
        skew = 0.0
        kurt = 0.0

    return SeriesStats(
        mean=mean,
        std=std,
        min=float(np.min(values)),
        max=float(np.max(values)),
        skewness=skew,
        kurtosis=kurt,
        autocorrelation_lag1=autocorr,
        trend_slope=slope,
        n_points=len(values),
    )


def profile_series(
    metric_name: str,
    labels: dict[str, str],
    timestamps: np.ndarray,
    values: np.ndarray,
) -> SeriesProfile:
    """Profile a single gauge or counter series."""
    metric_type = MetricType.detect(metric_name)
    prof = SeriesProfile(
        metric_name=metric_name,
        labels=labels,
        metric_type=metric_type.value,
    )

    if metric_type == MetricType.COUNTER:
        prof.stats = _compute_stats(values)
        # Compute rates (diffs) — handle resets
        diffs = np.diff(values)
        # Detect resets: large negative diffs
        resets = diffs < 0
        n_resets = int(np.sum(resets))
        # For rate computation, replace reset diffs with the post-reset value
        # (assuming counter restarted from 0)
        rates = diffs.copy()
        if n_resets > 0:
            reset_indices = np.where(resets)[0]
            for idx in reset_indices:
                rates[idx] = values[idx + 1]  # post-reset value is the rate since restart
        # Rates should be >= 0
        rates = np.maximum(rates, 0.0)
        prof.rate_stats = _compute_stats(rates)
        # Resets per hour
        if len(timestamps) > 1:
            ts = pd.to_datetime(timestamps)
            duration_hours = (ts.max() - ts.min()).total_seconds() / 3600
            prof.resets_per_hour = n_resets / max(duration_hours, 1 / 3600)
        else:
            prof.resets_per_hour = 0.0
    else:
        # Gauge
        prof.stats = _compute_stats(values)

    return prof


def compute_series_delta(
    baseline: SeriesProfile,
    anomaly: SeriesProfile,
) -> SeriesProfile:
    """Compute anomaly delta relative to baseline, updating the anomaly profile in place."""
    anomaly.delta_mean = anomaly.stats.mean - baseline.stats.mean
    anomaly.delta_std = anomaly.stats.std - baseline.stats.std

    if anomaly.metric_type == "counter" and anomaly.rate_stats and baseline.rate_stats:
        anomaly.delta_rate_mean = anomaly.rate_stats.mean - baseline.rate_stats.mean
        anomaly.delta_rate_std = anomaly.rate_stats.std - baseline.rate_stats.std

    return anomaly


def generate_series(
    profile: SeriesProfile,
    n_points: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic values from a series profile."""
    if rng is None:
        rng = np.random.default_rng()

    if profile.metric_type == "counter":
        return _generate_counter(profile, n_points, rng)
    else:
        return _generate_gauge(profile, n_points, rng)


def generate_series_with_innovations(
    profile: SeriesProfile,
    innovations: np.ndarray,
) -> np.ndarray:
    """Generate synthetic values using pre-drawn correlated innovations.

    The innovations should be standard normal (mean=0, std=1).
    This function applies AR(1) filtering and marginal transformation.
    """
    n_points = len(innovations)
    if profile.metric_type == "counter":
        return _generate_counter_with_innovations(profile, innovations)
    else:
        return _generate_gauge_with_innovations(profile, innovations)


def _generate_gauge_with_innovations(
    profile: SeriesProfile,
    innovations: np.ndarray,
) -> np.ndarray:
    """Generate gauge values using correlated innovations in AR(1)."""
    s = profile.stats
    n_points = len(innovations)
    if n_points == 0:
        return np.array([])

    phi = np.clip(s.autocorrelation_lag1, -0.99, 0.99)
    innovation_std = s.std * np.sqrt(max(1.0 - phi ** 2, 0.01))

    values = np.empty(n_points)
    values[0] = s.mean + innovations[0] * max(s.std, 1e-12)
    for i in range(1, n_points):
        values[i] = s.mean + phi * (values[i - 1] - s.mean) + innovations[i] * max(innovation_std, 1e-12)

    if abs(s.trend_slope) > 0:
        values += s.trend_slope * np.arange(n_points)

    values = np.clip(values, s.min, s.max)
    return values


def _generate_counter_with_innovations(
    profile: SeriesProfile,
    innovations: np.ndarray,
) -> np.ndarray:
    """Generate counter values using correlated innovations for rates."""
    n_points = len(innovations)
    if profile.rate_stats is None:
        return np.full(n_points, profile.stats.mean)

    rs = profile.rate_stats
    phi = np.clip(rs.autocorrelation_lag1, -0.99, 0.99)
    innovation_std = rs.std * np.sqrt(max(1.0 - phi ** 2, 0.01))

    rates = np.empty(n_points)
    rates[0] = max(rs.mean + innovations[0] * max(rs.std, 1e-12), 0.0)
    for i in range(1, n_points):
        rates[i] = rs.mean + phi * (rates[i - 1] - rs.mean) + innovations[i] * max(innovation_std, 1e-12)
    rates = np.maximum(rates, 0.0)

    values = np.cumsum(rates)
    values += max(profile.stats.min, 0.0)
    return values


def _generate_gauge(
    profile: SeriesProfile,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate gauge values with AR(1) autocorrelation."""
    s = profile.stats
    if n_points == 0:
        return np.array([])

    phi = np.clip(s.autocorrelation_lag1, -0.99, 0.99)
    innovation_std = s.std * np.sqrt(max(1.0 - phi ** 2, 0.01))

    # AR(1) process
    values = np.empty(n_points)
    values[0] = s.mean + rng.normal(0, max(s.std, 1e-12))
    for i in range(1, n_points):
        values[i] = s.mean + phi * (values[i - 1] - s.mean) + rng.normal(0, max(innovation_std, 1e-12))

    # Add trend
    if abs(s.trend_slope) > 0:
        values += s.trend_slope * np.arange(n_points)

    # Clip to observed range
    values = np.clip(values, s.min, s.max)

    return values


def _generate_counter(
    profile: SeriesProfile,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate counter values (monotonically increasing via cumulative rates)."""
    if profile.rate_stats is None:
        return np.full(n_points, profile.stats.mean)

    rs = profile.rate_stats
    phi = np.clip(rs.autocorrelation_lag1, -0.99, 0.99)
    innovation_std = rs.std * np.sqrt(max(1.0 - phi ** 2, 0.01))

    # Generate rates with AR(1)
    rates = np.empty(n_points)
    rates[0] = max(rs.mean + rng.normal(0, max(rs.std, 1e-12)), 0.0)
    for i in range(1, n_points):
        rates[i] = rs.mean + phi * (rates[i - 1] - rs.mean) + rng.normal(0, max(innovation_std, 1e-12))
    rates = np.maximum(rates, 0.0)

    # Cumulative sum for counter values
    values = np.cumsum(rates)
    # Start from a plausible initial value
    values += max(profile.stats.min, 0.0)

    return values
