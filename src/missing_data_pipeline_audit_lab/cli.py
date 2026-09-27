"""Command-line entry point."""

from __future__ import annotations

import argparse
import json

from .evaluation import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the missing-data pipeline audit.")
    parser.add_argument("--output-dir", default="artifacts", help="Artifact directory")
    parser.add_argument("--replicates", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    summary = run_experiment(
        output_dir=arguments.output_dir,
        n_replicates=arguments.replicates,
        seed=arguments.seed,
    )
    print(json.dumps(summary, indent=2))
