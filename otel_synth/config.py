"""Configuration and profile dataclasses for otel_synth."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class MetricType(str, Enum):
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"

    @staticmethod
    def detect(metric_name: str) -> MetricType:
        """Detect metric type from Prometheus/OTel naming conventions."""
        if metric_name.endswith("_total"):
            return MetricType.COUNTER
        if metric_name.endswith(("_bucket", "_count", "_sum")):
            return MetricType.HISTOGRAM
        return MetricType.GAUGE


def _histogram_family(metric_name: str) -> str:
    """Extract the histogram family base name from a _bucket/_count/_sum metric."""
    for suffix in ("_bucket", "_count", "_sum"):
        if metric_name.endswith(suffix):
            return metric_name[: -len(suffix)]
    return metric_name


# ---------------------------------------------------------------------------
# Series profile (gauges and counters)
# ---------------------------------------------------------------------------


@dataclass
class SeriesStats:
    """Statistical summary of a single time series."""

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    autocorrelation_lag1: float = 0.0
    trend_slope: float = 0.0
    n_points: int = 0


@dataclass
class SeriesProfile:
    """Profile for a single gauge or counter series."""

    metric_name: str
    labels: dict[str, str]
    metric_type: str  # "gauge" or "counter"
    stats: SeriesStats = field(default_factory=SeriesStats)
    # Counter-specific
    rate_stats: SeriesStats | None = None  # stats computed on diff (rate)
    resets_per_hour: float = 0.0
    # Anomaly delta (only for anomaly regime profiles)
    delta_mean: float | None = None
    delta_std: float | None = None
    delta_rate_mean: float | None = None
    delta_rate_std: float | None = None
    existence: str = "both"  # "both", "emergent", "disappeared"


# ---------------------------------------------------------------------------
# Histogram profile
# ---------------------------------------------------------------------------


@dataclass
class HistogramFamilyProfile:
    """Profile for a histogram metric family (all le buckets + count + sum)."""

    family_name: str  # base name without _bucket/_count/_sum
    labels_without_le: dict[str, str]
    le_boundaries: list[float]  # sorted, without +Inf
    le_boundary_strings: list[str] = field(default_factory=list)  # original le label strings
    # Fitted distribution parameters (log-normal by default)
    dist_name: str = "lognorm"
    dist_params: dict[str, float] = field(default_factory=dict)  # shape, loc, scale
    # Per-interval observation count stats
    observations_per_step: SeriesStats = field(default_factory=SeriesStats)
    n_timestamps: int = 0
    # Anomaly delta
    delta_dist_params: dict[str, float] | None = None
    delta_observations_mean: float | None = None
    existence: str = "both"


# ---------------------------------------------------------------------------
# Service correlation
# ---------------------------------------------------------------------------


@dataclass
class ServiceCorrelation:
    """Covariance structure for series within one service."""

    service_name: str
    series_keys: list[str]  # ordered list of series identifiers
    covariance_matrix: list[list[float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regime profile (the saved artifact)
# ---------------------------------------------------------------------------


@dataclass
class ProfileMetadata:
    """Metadata about a profiled regime."""

    source_csv: str = ""
    regime_name: str = ""
    is_baseline: bool = True
    duration_seconds: float = 0.0
    step_seconds: float = 60.0
    timestamp_min: str = ""
    timestamp_max: str = ""
    n_series: int = 0
    n_histogram_families: int = 0


@dataclass
class RegimeProfile:
    """Complete profile for a single regime (baseline or anomaly)."""

    metadata: ProfileMetadata = field(default_factory=ProfileMetadata)
    series_profiles: dict[str, SeriesProfile] = field(default_factory=dict)
    histogram_profiles: dict[str, HistogramFamilyProfile] = field(default_factory=dict)
    service_correlations: dict[str, ServiceCorrelation] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> RegimeProfile:
        with open(path) as f:
            data = json.load(f)
        return _dict_to_regime_profile(data)


def _dict_to_regime_profile(data: dict[str, Any]) -> RegimeProfile:
    """Reconstruct a RegimeProfile from a JSON-loaded dict."""
    metadata = ProfileMetadata(**data.get("metadata", {}))

    series_profiles = {}
    for key, sp_data in data.get("series_profiles", {}).items():
        stats = SeriesStats(**sp_data.pop("stats", {}))
        rate_raw = sp_data.pop("rate_stats", None)
        rate_stats = SeriesStats(**rate_raw) if rate_raw else None
        series_profiles[key] = SeriesProfile(stats=stats, rate_stats=rate_stats, **sp_data)

    histogram_profiles = {}
    for key, hp_data in data.get("histogram_profiles", {}).items():
        obs_raw = hp_data.pop("observations_per_step", {})
        obs_stats = SeriesStats(**obs_raw)
        histogram_profiles[key] = HistogramFamilyProfile(
            observations_per_step=obs_stats, **hp_data
        )

    service_correlations = {}
    for key, sc_data in data.get("service_correlations", {}).items():
        service_correlations[key] = ServiceCorrelation(**sc_data)

    return RegimeProfile(
        metadata=metadata,
        series_profiles=series_profiles,
        histogram_profiles=histogram_profiles,
        service_correlations=service_correlations,
    )


# ---------------------------------------------------------------------------
# Series key helpers
# ---------------------------------------------------------------------------


def series_key(metric_name: str, labels: dict[str, str]) -> str:
    """Deterministic string key for a (metric, labels) pair."""
    sorted_labels = sorted(labels.items())
    return f"{metric_name}|{json.dumps(sorted_labels, sort_keys=True)}"


def histogram_family_key(family_name: str, labels_without_le: dict[str, str]) -> str:
    """Deterministic string key for a histogram family."""
    sorted_labels = sorted(labels_without_le.items())
    return f"{family_name}|{json.dumps(sorted_labels, sort_keys=True)}"
