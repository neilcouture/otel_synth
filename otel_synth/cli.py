"""CLI entry point for otel_synth."""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="otel_synth",
        description="Synthetic OTel Metrics Generator",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # profile command
    profile_parser = subparsers.add_parser(
        "profile", help="Profile all regimes from a regimes.json config"
    )
    profile_parser.add_argument(
        "--regimes",
        default="./regimes.json",
        help="Path to regimes.json (default: ./regimes.json)",
    )
    profile_parser.add_argument(
        "--output-dir",
        default="./profiles/",
        help="Output directory for profile files (default: ./profiles/)",
    )
    profile_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes (0 = sequential, default: 0)",
    )

    # generate command
    gen_parser = subparsers.add_parser(
        "generate", help="Generate synthetic data from a profile"
    )
    gen_parser.add_argument(
        "--profile", required=True, help="Path to a .profile.json file"
    )
    gen_parser.add_argument(
        "--start-time",
        default="now",
        help="Start timestamp (ISO format or 'now', default: now)",
    )
    gen_parser.add_argument(
        "--duration", type=int, default=60, help="Duration in minutes (default: 60)"
    )
    gen_parser.add_argument(
        "--step", type=int, default=60, help="Step interval in seconds (default: 60)"
    )
    gen_parser.add_argument(
        "--output",
        default="./output/synthetic.csv",
        help="Output CSV path (default: ./output/synthetic.csv)",
    )
    gen_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # compose command
    compose_parser = subparsers.add_parser(
        "compose", help="Compose a multi-segment scenario"
    )
    compose_parser.add_argument(
        "--scenario", required=True, help="Path to scenario YAML file"
    )
    compose_parser.add_argument(
        "--output",
        default="./output/scenario.csv",
        help="Output CSV path (default: ./output/scenario.csv)",
    )
    compose_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a scenario YAML and print a summary"
    )
    analyze_parser.add_argument(
        "--scenario", required=True, help="Path to scenario YAML file"
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "profile":
        from otel_synth.profiler import profile_all

        profiles = profile_all(
            regimes_path=args.regimes,
            output_dir=args.output_dir,
            workers=args.workers,
        )
        print(f"Profiled {len(profiles)} regimes")

    elif args.command == "generate":
        from otel_synth.generator import generate

        df = generate(
            profile_path=args.profile,
            start_time=args.start_time,
            duration_minutes=args.duration,
            step_seconds=args.step,
            output_path=args.output,
            seed=args.seed,
        )
        print(f"Generated {len(df)} rows to {args.output}")

    elif args.command == "analyze":
        from otel_synth.composer import analyze_scenario

        analyze_scenario(scenario_path=args.scenario)

    elif args.command == "compose":
        from otel_synth.composer import compose

        df = compose(
            scenario_path=args.scenario,
            output_path=args.output,
            seed=args.seed,
        )
        print(f"Composed {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
