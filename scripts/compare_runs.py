#!/usr/bin/env python3
"""Cross-experiment comparison for policy repository runs.

Scans all experiment folders under ``outputs/experiments/``, reads
``config.json`` and evaluation CSV files from each, and generates
comparison plots.

Usage:
    # Compare all experiments
    python scripts/compare_runs.py

    # Compare specific run directories
    python scripts/compare_runs.py --runs outputs/experiments/dino_matching_2026-06-07_22-56-21 ...

    # Filter to specific repo types
    python scripts/compare_runs.py --filter dino,morphology

    # Print summary table to stdout only (no plots)
    python scripts/compare_runs.py --table-only

Output (saved to ``outputs/run_comparisons/`` by default):
    comparison_solve_rate.png          — grouped bar chart: mean solve rate
    comparison_avg_return.png          — grouped bar chart: mean avg return
    frozen_vs_finetune_scatter.png     — scatter: frozen vs finetuned
    per_env_distributions.png          — violin plot: per-env solve rates
    comparison_summary.csv             — all metrics in one table
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from apa.utils.analysis import (
    collect_experiment_results,
    generate_comparison_plots,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-experiment comparison for policy repo runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", default="outputs/experiments",
        help="Parent directory containing experiment subdirectories "
             "(default: outputs/experiments/).  Ignored when --runs is used.",
    )
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Explicit list of run directories to compare "
             "(overrides --input).",
    )
    parser.add_argument(
        "--output", default="outputs/run_comparisons",
        help="Directory to save comparison plots and summary CSV "
             "(default: outputs/run_comparisons/).",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Comma-separated repo types to include "
             "(e.g., 'dino_matching,morphology_matching').",
    )
    parser.add_argument(
        "--table-only", action="store_true",
        help="Print summary table to stdout and exit "
             "(no plot generation).",
    )
    args = parser.parse_args()

    # ── Collect results ──────────────────────────────────────────────────
    if args.runs:
        run_dirs = [Path(d) for d in args.runs]
        outputs_root = None
    else:
        run_dirs = None
        outputs_root = Path(args.input)

    df = collect_experiment_results(
        outputs_root=outputs_root,
        run_dirs=run_dirs,
    )

    if df.empty:
        logger.error("No valid experiment directories found.  "
                      "Each directory needs config.json and eval_frozen.csv.")
        sys.exit(1)

    logger.info(
        f"Found {len(df)} runs across {df['repo_type'].nunique()} repo types: "
        f"{sorted(df['repo_type'].unique())}"
    )

    # ── Filter ────────────────────────────────────────────────────────────
    if args.filter:
        allowed = set(args.filter.split(","))
        df = df[df["repo_type"].isin(allowed)]
        if df.empty:
            logger.error(f"No runs match filter: {args.filter}")
            sys.exit(1)
        logger.info(f"After filtering: {len(df)} runs.")

    # ── Output ────────────────────────────────────────────────────────────
    if args.table_only:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 240)
        pd.set_option("display.max_colwidth", 70)
        # Reorder: metrics first, then config
        front = [c for c in [
            "repo_display", "repo_type",
            "frozen_solve_rate_mean", "frozen_solve_rate_std",
            "finetune_solve_rate_mean", "finetune_solve_rate_std",
            "frozen_avg_return_mean", "frozen_avg_return_std",
            "finetune_avg_return_mean", "finetune_avg_return_std",
        ] if c in df.columns]
        rest = [c for c in df.columns if c not in front and c != "run_dir"]
        print(df[front + rest].to_string(index=False))
    else:
        out_dir = Path(args.output)
        root = Path(args.input) if outputs_root is not None else (
            Path(args.runs[0]).parent if args.runs else None
        )
        generate_comparison_plots(df, root, out_dir)


if __name__ == "__main__":
    main()
