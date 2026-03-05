"""Phase 3: Compose multi-segment scenarios with regime mixing."""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from otel_synth.config import RegimeProfile, SeriesProfile, HistogramFamilyProfile
from otel_synth.generator import generate_from_profile

logger = logging.getLogger(__name__)


def compose(
    scenario_path: str | Path,
    output_path: str | Path = "./output/scenario.csv",
    seed: int | None = None,
) -> pd.DataFrame:
    """Compose a multi-segment scenario from a YAML config.

    Returns the combined DataFrame in (timestamp, metric, labels, value) format.
    """
    scenario_path = Path(scenario_path)
    output_path = Path(output_path)

    with open(scenario_path) as f:
        config = yaml.safe_load(f)

    profiles_dir = Path(config.get("profiles_dir", "./profiles/"))
    # Resolve relative to scenario file location
    if not profiles_dir.is_absolute():
        profiles_dir = scenario_path.parent / profiles_dir

    scenario = config["scenario"]
    start_time_str = scenario.get("start_time", "now")
    if start_time_str.lower() == "now":
        start_time = datetime.utcnow()
    else:
        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))

    step_seconds = scenario.get("step_seconds", 60)

    # Load all needed profiles
    profile_cache: dict[str, RegimeProfile] = {}

    def get_profile(name: str) -> RegimeProfile:
        if name not in profile_cache:
            path = profiles_dir / f"{name}.profile.json"
            profile_cache[name] = RegimeProfile.load(path)
        return profile_cache[name]

    rng = np.random.default_rng(seed)

    segments_dfs: list[pd.DataFrame] = []
    ground_truth_rows: list[dict] = []
    current_time = start_time

    for segment in scenario["segments"]:
        duration_minutes = segment["duration_minutes"]
        n_points = int(duration_minutes * 60 / step_seconds)

        # Determine which regime(s) this segment uses
        if "regime" in segment:
            regime = segment["regime"]
            regime_names = regime if isinstance(regime, list) else [regime]
        elif "regimes" in segment:
            regime_names = segment["regimes"]
        else:
            raise ValueError(f"Segment must have 'regime' or 'regimes': {segment}")

        is_anomaly = not (regime_names == ["baseline"])

        # Build the effective profile for this segment
        if regime_names == ["baseline"]:
            effective_profile = get_profile("baseline")
        elif len(regime_names) == 1:
            effective_profile = get_profile(regime_names[0])
        else:
            # Mixed anomaly regimes — compose additively on baseline
            effective_profile = _compose_anomaly_profiles(
                get_profile("baseline"),
                [get_profile(name) for name in regime_names],
            )

        segment_df = generate_from_profile(
            effective_profile, current_time, n_points, step_seconds, rng
        )
        segments_dfs.append(segment_df)

        # Ground truth
        if is_anomaly:
            segment_end = current_time + timedelta(minutes=duration_minutes)
            ground_truth_rows.append({
                "start_time": current_time.isoformat() + "Z",
                "end_time": segment_end.isoformat() + "Z",
                "regimes": ",".join(regime_names),
            })

        current_time += timedelta(minutes=duration_minutes)

    # Combine all segments
    combined = pd.concat(segments_dfs, ignore_index=True)
    combined = combined.sort_values(["timestamp", "metric", "labels"]).reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved scenario data to {output_path} ({len(combined)} rows)")

    # Save ground truth
    gt_config = config.get("ground_truth", {})
    gt_path = gt_config.get("output", str(output_path.parent / "ground_truth.csv"))
    if not Path(gt_path).is_absolute():
        gt_path = scenario_path.parent / gt_path
    gt_df = pd.DataFrame(ground_truth_rows, columns=["start_time", "end_time", "regimes"])
    Path(gt_path).parent.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(gt_path, index=False)
    logger.info(f"Saved ground truth to {gt_path} ({len(gt_df)} anomaly segments)")

    return combined


def _compose_anomaly_profiles(
    baseline: RegimeProfile,
    anomaly_profiles: list[RegimeProfile],
) -> RegimeProfile:
    """Compose multiple anomaly profiles additively on top of baseline.

    Mean shifts are summed, variance scales are multiplied.
    """
    composed = copy.deepcopy(baseline)
    composed.metadata.is_baseline = False
    composed.metadata.regime_name = "composed"

    # Compose series profile deltas
    for ap in anomaly_profiles:
        for skey, asp in ap.series_profiles.items():
            if asp.existence == "disappeared":
                continue
            if asp.existence == "emergent":
                # Add emergent series from this anomaly
                composed.series_profiles[skey] = copy.deepcopy(asp)
                continue

            if skey in composed.series_profiles:
                csp = composed.series_profiles[skey]
                # Additive mean shift
                if asp.delta_mean is not None:
                    csp.stats.mean += asp.delta_mean
                # Multiplicative variance scaling (delta_std is additive to std)
                if asp.delta_std is not None:
                    csp.stats.std = max(csp.stats.std + asp.delta_std, 0.01)
                # Counter rate deltas
                if asp.delta_rate_mean is not None and csp.rate_stats:
                    csp.rate_stats.mean += asp.delta_rate_mean
                if asp.delta_rate_std is not None and csp.rate_stats:
                    csp.rate_stats.std = max(csp.rate_stats.std + asp.delta_rate_std, 0.01)

    # Handle disappeared: remove only if ALL anomaly profiles mark as disappeared
    for skey in list(composed.series_profiles.keys()):
        if all(
            skey in ap.series_profiles and ap.series_profiles[skey].existence == "disappeared"
            for ap in anomaly_profiles
        ):
            composed.series_profiles[skey].existence = "disappeared"

    # Compose histogram profile deltas
    for ap in anomaly_profiles:
        for fkey, ahp in ap.histogram_profiles.items():
            if ahp.existence == "disappeared":
                continue
            if ahp.existence == "emergent":
                composed.histogram_profiles[fkey] = copy.deepcopy(ahp)
                continue

            if fkey in composed.histogram_profiles:
                chp = composed.histogram_profiles[fkey]
                if ahp.delta_dist_params:
                    for param_key, delta_val in ahp.delta_dist_params.items():
                        if param_key in chp.dist_params:
                            chp.dist_params[param_key] += delta_val
                if ahp.delta_observations_mean is not None:
                    chp.observations_per_step.mean += ahp.delta_observations_mean

    return composed


def analyze_scenario(scenario_path: str | Path) -> None:
    """Analyze a scenario YAML and print a summary."""
    scenario_path = Path(scenario_path)

    with open(scenario_path) as f:
        config = yaml.safe_load(f)

    profiles_dir = Path(config.get("profiles_dir", "./profiles/"))
    if not profiles_dir.is_absolute():
        profiles_dir = scenario_path.parent / profiles_dir

    scenario = config["scenario"]
    step_seconds = scenario.get("step_seconds", 60)

    # Parse segments
    regime_minutes: dict[str, float] = {}
    total_minutes = 0.0
    n_segments = 0
    n_anomaly_segments = 0
    multi_regime_segments: list[tuple[list[str], float]] = []
    all_regime_names: set[str] = set()

    for segment in scenario["segments"]:
        duration = segment["duration_minutes"]
        n_segments += 1
        total_minutes += duration

        if "regime" in segment:
            regime = segment["regime"]
            regime_names = regime if isinstance(regime, list) else [regime]
        elif "regimes" in segment:
            regime_names = segment["regimes"]
        else:
            continue

        is_anomaly = regime_names != ["baseline"]
        if is_anomaly:
            n_anomaly_segments += 1

        if len(regime_names) > 1:
            multi_regime_segments.append((regime_names, duration))

        for name in regime_names:
            all_regime_names.add(name)
            regime_minutes[name] = regime_minutes.get(name, 0.0) + duration

    # Format duration
    def fmt_duration(minutes: float) -> str:
        total_secs = int(minutes * 60)
        days, rem = divmod(total_secs, 86400)
        hours, rem = divmod(rem, 3600)
        mins, _ = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if mins or not parts:
            parts.append(f"{mins}m")
        return " ".join(parts)

    # Print summary
    print(f"Scenario: {scenario_path.name}")
    print(f"Total duration: {fmt_duration(total_minutes)}")
    print(f"Step: {step_seconds}s")
    print(f"Segments: {n_segments} total, {n_segments - n_anomaly_segments} baseline, {n_anomaly_segments} anomaly")
    print()

    # Per-regime breakdown
    print("Regime breakdown:")
    # Sort: baseline first, then by time descending
    sorted_regimes = sorted(
        regime_minutes.items(),
        key=lambda x: (x[0] != "baseline", -x[1]),
    )
    for name, mins in sorted_regimes:
        pct = mins / total_minutes * 100
        label = "baseline" if name == "baseline" else "anomaly"
        print(f"  {name:40s} {fmt_duration(mins):>10s}  {pct:5.1f}%  ({label})")
    print()

    # Multi-regime segments
    if multi_regime_segments:
        print("Multi-regime segments:")
        for names, dur in multi_regime_segments:
            print(f"  [{', '.join(names)}] — {fmt_duration(dur)}")
        print()

    # Check profiles exist
    missing = []
    found_profiles: dict[str, Path] = {}
    for name in sorted(all_regime_names):
        path = profiles_dir / f"{name}.profile.json"
        if path.exists():
            found_profiles[name] = path
        else:
            missing.append(name)

    if missing:
        print(f"MISSING profiles ({len(missing)}):")
        for name in missing:
            print(f"  {profiles_dir / name}.profile.json")
        print()

    # Estimated output size from baseline profile
    if "baseline" in found_profiles:
        bp = RegimeProfile.load(found_profiles["baseline"])
        n_series = bp.metadata.n_series
        n_hist = bp.metadata.n_histogram_families
        total_points = int(total_minutes * 60 / step_seconds)
        # Each series = 1 row per point; each histogram family = (n_buckets + 2) rows per point
        # Approximate histogram bucket count from first family
        n_bucket_rows = 0
        for hp in bp.histogram_profiles.values():
            n_bucket_rows += len(hp.le_boundaries) + 1 + 2  # buckets + +Inf + _count + _sum
        est_rows = (n_series * total_points) + (n_bucket_rows * total_points)
        print(f"Estimated output: ~{est_rows:,} rows ({n_series} series + {n_hist} histogram families, {total_points} points)")
    elif not missing:
        print("(Could not estimate output size — no baseline profile found)")
    print(f"Profiles dir: {profiles_dir}")
    if not missing:
        print("All profiles found.")
