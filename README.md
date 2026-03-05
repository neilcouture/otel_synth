# otel_synth

A tool that learns statistical profiles from real Prometheus/OTel metric captures and generates new synthetic data that is statistically indistinguishable from the original.

The primary use case is producing large volumes of realistic OTel metrics data with known anomaly patterns for validating anomaly detection systems.

## How It Works

```
Raw CSVs ──► profile ──► .profile.json ──► generate/compose ──► Synthetic CSVs
  (real)      (learn)      (reusable)        (produce)           (+ ground truth)
```

1. **Capture** raw metric CSVs from a Prometheus server (one CSV per regime: baseline, anomaly variants, etc.)
2. **Profile** each regime to extract statistical fingerprints — distributions, autocorrelation, trends, within-service correlations, and anomaly deltas relative to baseline
3. **Generate** unlimited new synthetic data from those profiles, or **compose** multi-segment scenarios that mix baseline and anomaly regimes with ground truth labels

## Input Format

otel_synth consumes raw long-format CSVs with four columns:

```
timestamp,metric,labels,value
2026-01-22 03:59:00,http_server_request_duration_seconds_bucket,"{'service_name': 'frontend', 'le': '0.1', ...}",1234.0
2026-01-22 04:00:00,http_server_request_duration_seconds_bucket,"{'service_name': 'frontend', 'le': '0.1', ...}",1237.0
```

- **timestamp** — `YYYY-MM-DD HH:MM:SS`
- **metric** — Prometheus metric name
- **labels** — Python dict repr string
- **value** — float

This is the format returned by `PrometheusClient.fetch_metrics_range()`. Metric types are auto-detected from naming conventions (`_total` = counter, `_bucket`/`_count`/`_sum` = histogram, everything else = gauge).

## Quick Start

### 1. Set up regimes

Create a `regimes.json` mapping regime names to raw CSV files:

```json
{
  "baseline": "./db/baseline.csv",
  "adFailure": "./db/adFailure.csv",
  "adHighCpu": "./db/adHighCpu.csv"
}
```

The `baseline` regime is required — all anomaly regimes are profiled relative to it.

### 2. Profile

```bash
python -m otel_synth.cli profile --regimes ./regimes.json --output-dir ./profiles/
```

This produces one `.profile.json` per regime:

```
profiles/
├── baseline.profile.json
├── adFailure.profile.json
└── adHighCpu.profile.json
```

### 3. Generate (single regime)

Generate synthetic data from a single profile:

```bash
python -m otel_synth.cli generate \
  --profile ./profiles/baseline.profile.json \
  --start-time "2026-01-22T00:00:00Z" \
  --duration 120 \
  --output ./output/synthetic.csv \
  --seed 42
```

### 4. Analyze (scenario dry-run)

Preview a scenario without running it — shows total duration, per-regime time breakdown, multi-regime segments, estimated output size, and checks that all required profiles exist:

```bash
python -m otel_synth.cli analyze --scenario scenario.yaml
```

```
Scenario: scenario.yaml
Total duration: 16h 15m
Step: 60s
Segments: 32 total, 16 baseline, 16 anomaly

Regime breakdown:
  baseline                                    12h 15m   75.4%  (baseline)
  adHighCpu                                       30m    3.1%  (anomaly)
  emailMemoryLeak                                 30m    3.1%  (anomaly)
  adFailure                                       15m    1.5%  (anomaly)
  ...

Multi-regime segments:
  [productCatalogFailure, adHighCpu, emailMemoryLeak] — 15m

Estimated output: ~7,146,750 rows (3355 series + 221 histogram families, 975 points)
Profiles dir: profiles
All profiles found.
```

### 5. Compose (multi-segment scenarios)

Create a `scenario.yaml`:

```yaml
profiles_dir: ./profiles/

scenario:
  start_time: "2026-01-22T00:00:00Z"
  step_seconds: 60

  segments:
    - regime: baseline
      duration_minutes: 60

    - regime: adFailure
      duration_minutes: 15

    - regime: baseline
      duration_minutes: 45

    - regimes: [adFailure, adHighCpu]    # simultaneous anomalies
      duration_minutes: 15

    - regime: baseline
      duration_minutes: 90

ground_truth:
  output: ./output/ground_truth.csv
```

```bash
python -m otel_synth.cli compose \
  --scenario scenario.yaml \
  --output ./output/scenario.csv \
  --seed 42
```

This produces the synthetic data CSV and a ground truth CSV:

```csv
start_time,end_time,regimes
2026-01-22T01:00:00+00:00Z,2026-01-22T01:15:00+00:00Z,adFailure
2026-01-22T01:45:00+00:00Z,2026-01-22T02:00:00+00:00Z,"adFailure,adHighCpu"
```

Only non-baseline segments appear in ground truth — this is the validation key for anomaly detection.

## CLI Reference

```
python -m otel_synth.cli [-v] {profile,generate,compose,analyze}
```

| Command | Flag | Default | Description |
|---------|------|---------|-------------|
| `profile` | `--regimes` | `./regimes.json` | Path to regimes config |
| | `--output-dir` | `./profiles/` | Output directory for profiles |
| | `--workers` | `0` | Parallel worker processes (0 = sequential) |
| `generate` | `--profile` | *(required)* | Path to `.profile.json` |
| | `--start-time` | `now` | ISO timestamp or `now` |
| | `--duration` | `60` | Duration in minutes |
| | `--step` | `60` | Step interval in seconds |
| | `--output` | `./output/synthetic.csv` | Output CSV path |
| | `--seed` | | Random seed |
| `analyze` | `--scenario` | *(required)* | Path to scenario YAML |
| `compose` | `--scenario` | *(required)* | Path to scenario YAML |
| | `--output` | `./output/scenario.csv` | Output CSV path |
| | `--seed` | | Random seed |

Use `-v` for verbose logging.

## What Gets Profiled

For each regime, otel_synth captures:

- **Gauges** — mean, std, min/max, skewness, kurtosis, lag-1 autocorrelation, linear trend
- **Counters** — raw value stats plus rate (diff) stats, reset detection and frequency
- **Histograms** — log-normal distribution fit from bucket boundaries and counts, per-step observation rate
- **Within-service correlation** — Ledoit-Wolf shrunk covariance matrix across all gauge/counter series per `service_name`
- **Anomaly deltas** — per-series shifts in mean and variance relative to baseline, emergent/disappeared series tracking

## How Generation Works

1. For each service with a correlation matrix, draw correlated standard-normal innovations via Cholesky decomposition
2. Feed innovations into per-series AR(1) processes that match each series' autocorrelation, mean, and variance
3. Counters: cumulative sum of generated rates (guaranteed monotonically increasing)
4. Histograms: sample from fitted distribution, derive consistent `_bucket`/`_count`/`_sum` with monotonicity constraints
5. Segment transitions in compose mode are hard cuts (realistic for sudden failures)
6. Mixed anomaly regimes compose additively: mean shifts are summed, variance scales are multiplied

## Dependencies

- `pandas` — CSV I/O
- `numpy` — numerical generation
- `scipy` — distribution fitting, statistics
- `scikit-learn` — Ledoit-Wolf covariance estimation
- `pyyaml` — scenario config

No dependency on Prometheus, any running infrastructure, or any ETL pipeline. otel_synth is a pure offline tool: CSVs in, profiles out, synthetic CSVs out.

## Project Structure

```
otel_synth/
├── cli.py                  # CLI entry point
├── config.py               # Dataclasses and serialization
├── profiler.py             # Phase 1: statistical profiling
├── generator.py            # Phase 2: synthetic generation
├── composer.py             # Phase 3: multi-segment scenarios
└── models/
    ├── series_profile.py   # Gauge/counter profiling and AR(1) generation
    ├── histogram_model.py  # Histogram distribution fitting and generation
    └── correlation.py      # Ledoit-Wolf within-service correlation
```
