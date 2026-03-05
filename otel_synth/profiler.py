"""Phase 1: Build statistical profiles from regime CSVs."""

from __future__ import annotations

import ast
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from otel_synth.config import (
    MetricType,
    ProfileMetadata,
    RegimeProfile,
    SeriesProfile,
    _histogram_family,
    histogram_family_key,
    series_key,
)
from otel_synth.models.histogram_model import (
    compute_histogram_delta,
    profile_histogram_family,
)
from otel_synth.models.correlation import compute_service_correlation
from otel_synth.models.series_profile import (
    compute_series_delta,
    profile_series,
)

logger = logging.getLogger(__name__)


def _load_csv(path: str | Path) -> pd.DataFrame:
    """Load a regime CSV and parse timestamps and labels."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df


def _parse_labels(label_str: str) -> dict[str, str]:
    """Parse Python dict repr label string."""
    return ast.literal_eval(label_str)


def _profile_regime(
    df: pd.DataFrame,
    regime_name: str,
    source_csv: str,
    is_baseline: bool = True,
) -> RegimeProfile:
    """Profile all series in a single regime DataFrame."""
    logger.info(f"Profiling regime '{regime_name}' ({len(df)} rows)")

    # Metadata
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    duration = (ts_max - ts_min).total_seconds()
    # Infer step from mode of timestamp diffs
    ts_unique = df["timestamp"].sort_values().unique()
    if len(ts_unique) > 1:
        diffs = np.diff(ts_unique.astype(np.int64) // 10**9)
        step = float(np.median(diffs))
    else:
        step = 60.0

    profile = RegimeProfile()
    profile.metadata = ProfileMetadata(
        source_csv=str(source_csv),
        regime_name=regime_name,
        is_baseline=is_baseline,
        duration_seconds=duration,
        step_seconds=step,
        timestamp_min=str(ts_min),
        timestamp_max=str(ts_max),
    )

    # Parse all labels once
    df["_parsed_labels"] = df["labels"].apply(_parse_labels)

    # Classify metrics
    histogram_families: dict[str, dict] = {}  # family_key -> collected data
    gauge_counter_groups: dict[str, dict] = {}  # series_key -> collected data

    # Determine which histogram families are real (have a _bucket metric)
    all_metrics = set(df["metric"].unique())
    real_histogram_families = set()
    for m in all_metrics:
        if m.endswith("_bucket"):
            real_histogram_families.add(_histogram_family(m))

    for metric_name in all_metrics:
        mtype = MetricType.detect(metric_name)
        metric_df = df[df["metric"] == metric_name]

        # Only treat _count/_sum as histogram if a _bucket sibling exists
        if mtype == MetricType.HISTOGRAM:
            family = _histogram_family(metric_name)
            if family not in real_histogram_families:
                mtype = MetricType.GAUGE

        if mtype == MetricType.HISTOGRAM:
            _collect_histogram_data(metric_name, metric_df, histogram_families)
        else:
            _collect_series_data(metric_name, metric_df, gauge_counter_groups)

    # Profile gauge/counter series
    n_series = 0
    for skey, data in gauge_counter_groups.items():
        sp = profile_series(
            metric_name=data["metric_name"],
            labels=data["labels"],
            timestamps=data["timestamps"],
            values=data["values"],
        )
        profile.series_profiles[skey] = sp
        n_series += 1

    # Profile histogram families
    n_hist = 0
    for fkey, data in histogram_families.items():
        hp = profile_histogram_family(
            family_name=data["family_name"],
            labels_without_le=data["labels_without_le"],
            le_boundaries=data["le_boundaries"],
            le_boundary_strings=data.get("le_boundary_strings", []),
            bucket_data=data["bucket_data"],
            count_values=data.get("count_values", np.array([])),
            sum_values=data.get("sum_values", np.array([])),
            timestamps=data["timestamps"],
        )
        profile.histogram_profiles[fkey] = hp
        n_hist += 1

    profile.metadata.n_series = n_series
    profile.metadata.n_histogram_families = n_hist

    # Compute within-service correlations for gauge/counter series
    _compute_service_correlations(profile, gauge_counter_groups)

    logger.info(
        f"  {n_series} series, {n_hist} histogram families, "
        f"{len(profile.service_correlations)} service correlations profiled"
    )
    return profile


def _compute_service_correlations(
    profile: RegimeProfile,
    gauge_counter_groups: dict[str, dict],
) -> None:
    """Group series by service_name and compute Ledoit-Wolf covariance per service."""
    # Group series keys by service_name
    service_series: dict[str, list[str]] = {}
    service_data: dict[str, dict[str, np.ndarray]] = {}

    for skey, data in gauge_counter_groups.items():
        labels = data["labels"]
        service = labels.get("service_name", "unknown")
        if service not in service_series:
            service_series[service] = []
            service_data[service] = {}
        service_series[service].append(skey)

        # Use rates for counters, raw values for gauges
        sp = profile.series_profiles[skey]
        values = data["values"]
        if sp.metric_type == "counter":
            diffs = np.diff(values)
            diffs = np.maximum(diffs, 0.0)
            service_data[service][skey] = diffs
        else:
            service_data[service][skey] = values

    for service, skeys in service_series.items():
        if len(skeys) < 2:
            continue
        # Ensure all arrays are the same length (trim to shortest)
        min_len = min(len(service_data[service][k]) for k in skeys)
        if min_len < 3:
            continue
        trimmed = {k: service_data[service][k][:min_len] for k in skeys}

        corr = compute_service_correlation(service, skeys, trimmed)
        profile.service_correlations[service] = corr


def _collect_series_data(
    metric_name: str,
    metric_df: pd.DataFrame,
    out: dict[str, dict],
) -> None:
    """Group gauge/counter rows by unique label set."""
    for labels_str, group in metric_df.groupby("labels"):
        labels = group["_parsed_labels"].iloc[0]
        skey = series_key(metric_name, labels)
        sorted_group = group.sort_values("timestamp")
        out[skey] = {
            "metric_name": metric_name,
            "labels": labels,
            "timestamps": sorted_group["timestamp"].values,
            "values": sorted_group["value"].values.astype(float),
        }


def _collect_histogram_data(
    metric_name: str,
    metric_df: pd.DataFrame,
    out: dict[str, dict],
) -> None:
    """Collect histogram bucket/count/sum data grouped by family + labels (minus le)."""
    family_name = _histogram_family(metric_name)
    suffix = metric_name[len(family_name):]  # _bucket, _count, or _sum

    for labels_str, group in metric_df.groupby("labels"):
        labels = group["_parsed_labels"].iloc[0]
        labels_no_le = {k: v for k, v in labels.items() if k != "le"}
        fkey = histogram_family_key(family_name, labels_no_le)

        if fkey not in out:
            out[fkey] = {
                "family_name": family_name,
                "labels_without_le": labels_no_le,
                "le_boundaries": [],
                "le_string_map": {},  # float -> original string
                "bucket_data": {},
                "count_values": np.array([]),
                "sum_values": np.array([]),
                "timestamps": np.array([]),
            }

        sorted_group = group.sort_values("timestamp")
        values = sorted_group["value"].values.astype(float)
        timestamps = sorted_group["timestamp"].values

        if suffix == "_bucket":
            le_val = labels.get("le", "+Inf")
            out[fkey]["bucket_data"][le_val] = values
            if le_val != "+Inf":
                try:
                    le_float = float(le_val)
                    out[fkey]["le_boundaries"].append(le_float)
                    out[fkey]["le_string_map"][le_float] = le_val
                except ValueError:
                    pass
            # Keep timestamps from any bucket series
            if len(timestamps) > len(out[fkey]["timestamps"]):
                out[fkey]["timestamps"] = timestamps
        elif suffix == "_count":
            out[fkey]["count_values"] = values
            if len(timestamps) > len(out[fkey]["timestamps"]):
                out[fkey]["timestamps"] = timestamps
        elif suffix == "_sum":
            out[fkey]["sum_values"] = values

    # Deduplicate and sort le boundaries, build ordered string list
    for fkey in out:
        sorted_floats = sorted(set(out[fkey]["le_boundaries"]))
        out[fkey]["le_boundaries"] = sorted_floats
        string_map = out[fkey]["le_string_map"]
        out[fkey]["le_boundary_strings"] = [
            string_map.get(f, str(f)) for f in sorted_floats
        ]


def _profile_regime_from_csv(
    csv_path: str,
    regime_name: str,
    is_baseline: bool,
) -> RegimeProfile:
    """Load a CSV and profile it. Top-level function for multiprocessing."""
    # Re-configure logging in worker processes
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _logger = logging.getLogger(__name__)
    _logger.info(f"Loading CSV: {csv_path}")
    df = _load_csv(csv_path)
    return _profile_regime(df, regime_name, csv_path, is_baseline=is_baseline)


def profile_all(
    regimes_path: str = "./regimes.json",
    output_dir: str = "./profiles/",
    workers: int = 0,
) -> dict[str, RegimeProfile]:
    """Profile all regimes from a regimes.json config.

    Args:
        regimes_path: path to regimes.json
        output_dir: output directory for profile files
        workers: number of parallel worker processes (0 = sequential)

    Returns dict mapping regime name to its profile.
    """
    regimes_path = Path(regimes_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(regimes_path) as f:
        regimes_config = json.load(f)

    # Resolve CSV paths relative to regimes.json location
    base_dir = regimes_path.parent

    # Profile baseline first
    if "baseline" not in regimes_config:
        raise ValueError("regimes.json must contain a 'baseline' entry")

    profiles: dict[str, RegimeProfile] = {}

    baseline_csv = base_dir / regimes_config["baseline"]
    logger.info(f"Loading baseline CSV: {baseline_csv}")
    baseline_df = _load_csv(baseline_csv)
    baseline_profile = _profile_regime(baseline_df, "baseline", str(baseline_csv), is_baseline=True)

    out_path = output_dir / "baseline.profile.json"
    baseline_profile.save(out_path)
    logger.info(f"Saved baseline profile to {out_path}")
    profiles["baseline"] = baseline_profile

    # Collect anomaly regimes
    anomaly_regimes = {
        name: str(base_dir / csv_path)
        for name, csv_path in regimes_config.items()
        if name != "baseline"
    }

    if not anomaly_regimes:
        return profiles

    if workers > 0:
        profiles.update(_profile_anomalies_parallel(
            anomaly_regimes, baseline_profile, output_dir, workers,
        ))
    else:
        profiles.update(_profile_anomalies_sequential(
            anomaly_regimes, baseline_profile, output_dir,
        ))

    return profiles


def _profile_anomalies_sequential(
    anomaly_regimes: dict[str, str],
    baseline_profile: RegimeProfile,
    output_dir: Path,
) -> dict[str, RegimeProfile]:
    """Profile anomaly regimes sequentially."""
    profiles: dict[str, RegimeProfile] = {}
    for regime_name, csv_full in anomaly_regimes.items():
        logger.info(f"Loading anomaly CSV: {csv_full}")
        anomaly_df = _load_csv(csv_full)
        anomaly_profile = _profile_regime(
            anomaly_df, regime_name, csv_full, is_baseline=False
        )
        _compute_regime_deltas(baseline_profile, anomaly_profile)

        out_path = output_dir / f"{regime_name}.profile.json"
        anomaly_profile.save(out_path)
        logger.info(f"Saved anomaly profile to {out_path}")
        profiles[regime_name] = anomaly_profile
    return profiles


def _profile_anomalies_parallel(
    anomaly_regimes: dict[str, str],
    baseline_profile: RegimeProfile,
    output_dir: Path,
    workers: int,
) -> dict[str, RegimeProfile]:
    """Profile anomaly regimes in parallel using multiprocessing."""
    profiles: dict[str, RegimeProfile] = {}
    n_workers = min(workers, len(anomaly_regimes))
    logger.info(f"Profiling {len(anomaly_regimes)} anomaly regimes with {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_name = {
            executor.submit(
                _profile_regime_from_csv, csv_full, regime_name, False
            ): regime_name
            for regime_name, csv_full in anomaly_regimes.items()
        }
        for future in as_completed(future_to_name):
            regime_name = future_to_name[future]
            try:
                anomaly_profile = future.result()
            except Exception:
                logger.exception(f"Failed to profile regime '{regime_name}'")
                continue

            # Deltas are cheap — compute in main process
            _compute_regime_deltas(baseline_profile, anomaly_profile)

            out_path = output_dir / f"{regime_name}.profile.json"
            anomaly_profile.save(out_path)
            logger.info(f"Saved anomaly profile to {out_path}")
            profiles[regime_name] = anomaly_profile

    return profiles


def _compute_regime_deltas(
    baseline: RegimeProfile,
    anomaly: RegimeProfile,
) -> None:
    """Compute per-series and per-histogram deltas for anomaly vs baseline."""
    # Series deltas
    for skey, anomaly_sp in anomaly.series_profiles.items():
        if skey in baseline.series_profiles:
            compute_series_delta(baseline.series_profiles[skey], anomaly_sp)
            anomaly_sp.existence = "both"
        else:
            anomaly_sp.existence = "emergent"

    # Mark disappeared series
    for skey in baseline.series_profiles:
        if skey not in anomaly.series_profiles:
            # Create a placeholder for disappeared series
            bp = baseline.series_profiles[skey]
            disappeared = SeriesProfile(
                metric_name=bp.metric_name,
                labels=bp.labels,
                metric_type=bp.metric_type,
                existence="disappeared",
            )
            anomaly.series_profiles[skey] = disappeared

    # Histogram deltas
    for fkey, anomaly_hp in anomaly.histogram_profiles.items():
        if fkey in baseline.histogram_profiles:
            compute_histogram_delta(baseline.histogram_profiles[fkey], anomaly_hp)
            anomaly_hp.existence = "both"
        else:
            anomaly_hp.existence = "emergent"

    for fkey in baseline.histogram_profiles:
        if fkey not in anomaly.histogram_profiles:
            from otel_synth.config import HistogramFamilyProfile
            bp = baseline.histogram_profiles[fkey]
            disappeared = HistogramFamilyProfile(
                family_name=bp.family_name,
                labels_without_le=bp.labels_without_le,
                le_boundaries=bp.le_boundaries,
                existence="disappeared",
            )
            anomaly.histogram_profiles[fkey] = disappeared
