#!/usr/bin/env python3
"""Submit each experiment as a separate Run:AI job."""

import time
import subprocess
import sys
from datetime import datetime

# Base runai arguments
BASE = [
    "runai", "submit",
    "--image", "registry.rcp.epfl.ch/nili-base/rcp-uv-base:v0.1",
    "--gpu", "1",
    "--node-pools", "v100",
    "--existing-pvc", "claimname=home,path=/home/niliahma",
]

# ── Define every experiment as (job_name, python_command) ──────────────────
EXPERIMENTS = [
    # Dino sweeps
    ("dino-sigma-0-3",  "python scripts/run.py --repo dino --sigma 0.3"),
    ("dino-sigma-0-5",  "python scripts/run.py --repo dino --sigma 0.5"),
    ("dino-sigma-0-7",  "python scripts/run.py --repo dino --sigma 0.7"),

    # Morphology sweeps
    ("morph-sigma-0-15", "python scripts/run.py --repo morphology --sigma 0.15"),
    ("morph-sigma-0-3",  "python scripts/run.py --repo morphology --sigma 0.3"),
    ("morph-sigma-0-5",  "python scripts/run.py --repo morphology --sigma 0.5"),

    # Policy latent sweeps
    ("policy-sigma-0-3", "python scripts/run.py --repo policy_latent --sigma 0.3"),
    ("policy-sigma-0-5", "python scripts/run.py --repo policy_latent --sigma 0.5"),
    ("policy-sigma-0-7", "python scripts/run.py --repo policy_latent --sigma 0.7"),

    # Performance sweeps
    ("perf-rank",      "python scripts/run.py --repo performance --mode return_rank"),
    ("perf-solve-rate", "python scripts/run.py --repo performance --mode solve_rate"),
    ("perf-hybrid",    "python scripts/run.py --repo performance --mode hybrid"),

    # Random
    ("random", "python scripts/run.py --repo random"),

    # Generalist
    ("generalist", "python scripts/run.py --repo generalist"),
]


def main():
    # Optional: only submit experiments whose names match a filter from CLI
    # e.g. `python submit_sweep.py dino`  → submits only dino jobs
    filter_str = sys.argv[1] if len(sys.argv) > 1 else ""

    for job_name, cmd in EXPERIMENTS:
        if filter_str and filter_str not in job_name:
            continue

        full_cmd = (
            f"cd $HOME/SP2026_OpenEndedLearning/ && "
            f"source .venv/bin/activate && "
            f"{cmd}"
        )

        submit_cmd = BASE + [
            "--name", job_name + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            "--", "bash", "-c", full_cmd,
        ]

        print(f"🚀 Submitting {job_name} ...")
        subprocess.run(submit_cmd)
        time.sleep(2)


if __name__ == "__main__":
    main()