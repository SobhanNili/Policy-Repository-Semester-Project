#!/usr/bin/env python3
"""Regenerate diagnostic plots from a completed run's saved data.

Usage:
    python scripts/plot_results.py outputs/dino_matching_2026-06-06_09-17-04/

Loads ``step_logs.pkl`` from the given run directory, generates all
diagnostic plots, and saves them to ``<run_dir>/plots/``.  Any existing
``plots/`` directory is deleted first, so re-running with updated plotting
code cleanly replaces old plots.

Examples:
    # Re-plot a specific run
    python scripts/plot_results.py outputs/random_matching_2026-06-05_15-57-42/

    # Re-plot the most recent run
    python scripts/plot_results.py $(ls -dt outputs/*/ | head -1)
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from loguru import logger

from apa.utils.analysis import generate_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate diagnostic plots from a completed run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dir", type=str,
        help="Path to a run output directory containing step_logs.pkl.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    pkl_path = run_dir / "step_logs.pkl"
    if not pkl_path.exists():
        logger.error(f"Error: {pkl_path} not found — not a valid run directory?")
        sys.exit(1)

    # ── Load saved data ────────────────────────────────────────────────────
    logger.info(f"Loading step_logs from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        step_logs = pickle.load(f)
    logger.info(f"Loaded {len(step_logs)} step logs.")

    # ── Remove old plots for a clean slate ──────────────────────────────────
    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        logger.info(f"Removing existing plots/ directory ...")
        shutil.rmtree(plots_dir)

    # ── Generate fresh plots ───────────────────────────────────────────────
    generate_plots(step_logs, run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
