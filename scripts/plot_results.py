#!/usr/bin/env python3
"""(Re)generate diagnostic plots from a completed experiment run's saved data.

Usage Examples
--------------
Single experiment: python scripts/plot_results.py outputs/dino_matching_2026-06-06_09-17-04/

Batch experiment: python scripts/plot_results.py --experiments-root outputs/experiments/

Loads ``step_logs.pkl`` from the given run directory (or each experiment
subdirectory under *experiments-root*), generates all diagnostic plots, and
saves them to ``<run_dir>/plots/``.  Any existing ``plots/`` directory is
deleted first, so re-running with updated plotting code cleanly replaces old
plots.
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from loguru import logger

from apa.utils.analysis import generate_plots

def _process_single_run(run_dir: Path) -> None:
    """Load step_logs.pkl from *run_dir* and (re)generate its plots."""
    pkl_path = run_dir / "step_logs.pkl"
    if not pkl_path.exists():
        logger.warning(f"Skipping {run_dir} - no step_logs.pkl found.")
        return

    logger.info(f"Loading step_logs from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        step_logs = pickle.load(f)
    logger.info(f"  Loaded {len(step_logs)} step logs.")

    # Remove old plots (if existing) for a clean slate
    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        logger.info(f"  Removing existing plots/ ...")
        shutil.rmtree(plots_dir)

    # Generate plots
    generate_plots(step_logs, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate diagnostic plots from completed runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "run_dir", type=str, nargs="?", default=None,
        help="Path to a single experiment output directory containing step_logs.pkl.",
    )
    mode_group.add_argument(
        "--experiments-root", type=str, default='outputs/experiments/',
        help=(
            "Root directory containing experiment subdirectories. "
            "Plots are regenerated for every subdirectory that contains "
            "a step_logs.pkl file."
        ),
    )
    args = parser.parse_args()

    # Batch mode: iterate over experiment subdirectories
    if args.experiments_root:
        root = Path(args.experiments_root)
        if not root.is_dir():
            logger.error(f"Experiments root not found: {root}")
            sys.exit(1)

        run_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and (d / "step_logs.pkl").exists()
        )
        if not run_dirs:
            logger.error(f"No experiment directories with step_logs.pkl found under {root}")
            sys.exit(1)

        logger.info(
            f"Batch plotting {len(run_dirs)} experiment(s) from {root} ..."
        )
        for i, d in enumerate(run_dirs, 1):
            logger.info(f"\n[{i}/{len(run_dirs)}] {d.name}")
            _process_single_run(d)

        logger.info(f"\nAll {len(run_dirs)} experiments plotted.")
        return

    # Single-run mode
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        logger.error(f"Directory not found: {run_dir}")
        sys.exit(1)

    _process_single_run(run_dir)
    logger.info("Done.")

if __name__ == "__main__":
    main()
