"""Phase 2: Generate synthetic data from profiles."""

from __future__ import annotations

import ast
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from otel_synth.config import RegimeProfile
from otel_synth.models.correlation import generate_correlated_innovations
from otel_synth.models.histogram_model import generate_histogram_family
from otel_synth.models.series_profile import (
    generate_series,
    generate_series_with_innovations,
)

logger = logging.getLogger(__name__)


def generate(
    profile_path: str | Path,
    start_time: datetime | str,
    duration_minutes: int = 60,
    step_seconds: int = 60,
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic metric data from a profile.

    Args:
        profile_path: path to a .profile.json file
        start_time: start timestamp (datetime or ISO string)
        duration_minutes: how many minutes of data to generate
        step_seconds: interval between data points
        output_path: optional CSV output path
        seed: random seed for reproducibility

    Returns:
        DataFrame in (timestamp, metric, labels, value) long format
    """
    profile = RegimeProfile.load(profile_path)

    if isinstance(start_time, str):
        if start_time.lower() == "now":
            start_time = datetime.utcnow()
        else:
            start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

    n_points = int(duration_minutes * 60 / step_seconds)
    rng = np.random.default_rng(seed)

    df = _generate_from_profile_impl(profile, start_time, n_points, step_seconds, rng)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved synthetic data to {output_path}")

    return df


def generate_from_profile(
    profile: RegimeProfile,
    start_time: datetime,
    n_points: int,
    step_seconds: int = 60,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic data directly from a RegimeProfile object.

    Used by the composer for segment generation.
    """
    if rng is None:
        rng = np.random.default_rng()

    return _generate_from_profile_impl(profile, start_time, n_points, step_seconds, rng)


def _generate_from_profile_impl(
    profile: RegimeProfile,
    start_time: datetime,
    n_points: int,
    step_seconds: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Core generation logic shared by generate() and generate_from_profile()."""
    timestamps = [
        start_time + timedelta(seconds=i * step_seconds) for i in range(n_points)
    ]
    ts_strings = [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps]

    rows: list[dict] = []

    # --- Correlated gauge/counter generation ---
    # Collect which series keys are covered by correlations
    correlated_keys: set[str] = set()
    for svc, corr in profile.service_correlations.items():
        if corr.covariance_matrix and len(corr.series_keys) >= 2:
            # Draw correlated innovations for this service
            innovations = generate_correlated_innovations(corr, n_points, rng)
            for skey in corr.series_keys:
                if skey not in profile.series_profiles:
                    continue
                sp = profile.series_profiles[skey]
                if sp.existence == "disappeared":
                    continue
                if skey not in innovations:
                    continue
                values = generate_series_with_innovations(sp, innovations[skey])
                labels_str = str(sp.labels)
                for i in range(n_points):
                    rows.append({
                        "timestamp": ts_strings[i],
                        "metric": sp.metric_name,
                        "labels": labels_str,
                        "value": values[i],
                    })
                correlated_keys.add(skey)

    # --- Independent gauge/counter generation (uncorrelated fallback) ---
    n_generated = 0
    for skey, sp in profile.series_profiles.items():
        if skey in correlated_keys:
            n_generated += 1
            continue
        if sp.existence == "disappeared":
            continue
        values = generate_series(sp, n_points, rng)
        labels_str = str(sp.labels)
        for i in range(n_points):
            rows.append({
                "timestamp": ts_strings[i],
                "metric": sp.metric_name,
                "labels": labels_str,
                "value": values[i],
            })
        n_generated += 1

    n_correlated = len(correlated_keys)

    # --- Histogram generation (always independent) ---
    n_hist_generated = 0
    for fkey, hp in profile.histogram_profiles.items():
        if hp.existence == "disappeared":
            continue

        hist_data = generate_histogram_family(hp, n_points, rng=rng)

        labels_no_le = hp.labels_without_le
        count_labels_str = str(labels_no_le)
        count_metric = f"{hp.family_name}_count"
        sum_metric = f"{hp.family_name}_sum"

        for i in range(n_points):
            rows.append({
                "timestamp": ts_strings[i],
                "metric": count_metric,
                "labels": count_labels_str,
                "value": hist_data["_count"][i],
            })
            rows.append({
                "timestamp": ts_strings[i],
                "metric": sum_metric,
                "labels": count_labels_str,
                "value": hist_data["_sum"][i],
            })

        bucket_metric = f"{hp.family_name}_bucket"
        all_le = (hp.le_boundary_strings or [str(le) for le in hp.le_boundaries]) + ["+Inf"]
        for le_str in all_le:
            bucket_labels = dict(labels_no_le)
            bucket_labels["le"] = le_str
            bucket_labels_str = str(bucket_labels)
            bucket_values = hist_data[le_str]
            for i in range(n_points):
                rows.append({
                    "timestamp": ts_strings[i],
                    "metric": bucket_metric,
                    "labels": bucket_labels_str,
                    "value": bucket_values[i],
                })
        n_hist_generated += 1

    logger.info(
        f"Generated {n_generated} series ({n_correlated} correlated) + "
        f"{n_hist_generated} histogram families "
        f"({n_points} points each, {len(rows)} total rows)"
    )

    df = pd.DataFrame(rows, columns=["timestamp", "metric", "labels", "value"])
    df = df.sort_values(["timestamp", "metric", "labels"]).reset_index(drop=True)
    return df
