#!/usr/bin/env python3
"""Experiment runner for PolicyRepo-based skill repository training.

Usage:
    python scripts/run.py --repo random
    python scripts/run.py --repo dino --sigma 0.3 --temperature 1.0
    python scripts/run.py --repo performance --mode rank --policy_eval_ep_count 5
    python scripts/run.py --repo morphology --history_decay 0.1 --sigma 0.15

Each run creates a timestamped output directory containing:
    - step_logs.pkl      : list of RepoStepLog namedtuples
    - config.json         : resolved configuration dict
    - repo_kwargs.json    : repo-specific kwargs used for this run
    - plots/              : 5 diagnostic PNG plots
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
from tqdm.auto import tqdm

matplotlib.use("Agg")  # headless — must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import wandb  # noqa: E402

from apa.utils.config import load_config  # noqa: E402
from apa.utils.env import (  # noqa: E402
    get_random_envs,
    load_levels,
    make_env_from_config,
)
from apa.repo import (  # noqa: E402
    DinoMatchingRepo,
    MorphologyMatchingRepo,
    PerformanceMatchingRepo,
    PixelMatchingRepo,
    RandomMatchingRepo,
)

# ──────────────────────────────────────────────────────────────────────────────
# Repo registry: name → (class, dict of default kwargs)
# Each entry maps CLI-friendly names to their class and the constructor kwargs
# that can be overridden from the command line.
# ──────────────────────────────────────────────────────────────────────────────

REPO_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "random": (RandomMatchingRepo, {}),
    "pixel": (
        PixelMatchingRepo,
        {"history_window": 5, "policy_success_threshold": 0.5},
    ),
    "morphology": (
        MorphologyMatchingRepo,
        {
            "history_decay": 0.1,
            "sigma": 0.15,
            "temperature": 1.0,
            "feature_weights": "(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)",
        },
    ),
    "performance": (
        PerformanceMatchingRepo,
        {
            "mode": "rank",
            "policy_eval_ep_count": 3,
            "temperature": 1.0,
            "hybrid_return_weight": 0.1,
        },
    ),
    "dino": (
        DinoMatchingRepo,
        {
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "history_decay": 0.2,
            "sigma": 0.3,
            "temperature": 1.0,
        },
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────────


def _coerce_value(raw: str) -> Any:
    """Attempt to parse a CLI string value into a Python literal.

    Tries JSON first (handles numbers, bools, lists), falls back to string.
    """
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _post_process_kwargs(repo_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply type-specific post-processing to parsed kwargs."""
    if repo_name == "morphology" and "feature_weights" in kwargs:
        fw = kwargs["feature_weights"]
        if isinstance(fw, str):
            # Strip parentheses and parse
            fw = fw.strip("()")
            kwargs["feature_weights"] = tuple(float(x.strip()) for x in fw.split(","))
        elif isinstance(fw, list):
            kwargs["feature_weights"] = tuple(float(x) for x in fw)
    return kwargs


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, Any]]:
    """Parse CLI arguments, returning (shared_args, repo_kwargs).

    Shared args: --repo, --seed, --config_name, --output_dir
    Repo kwargs: dynamically parsed from registry for the selected repo type.
    """
    # First pass — just get the repo name
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--repo", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args(argv)

    if pre_args.repo is None:
        pre_parser.error("--repo is required. Choices: " + ", ".join(REPO_REGISTRY))
    if pre_args.repo not in REPO_REGISTRY:
        pre_parser.error(
            f"Unknown repo '{pre_args.repo}'. Choices: " + ", ".join(REPO_REGISTRY)
        )

    repo_name = pre_args.repo
    _, known_keys = REPO_REGISTRY[repo_name]

    # Build the main parser — note: ―repo is not required here since pre-parser
    # already consumed it; we set it explicitly after parsing.
    parser = argparse.ArgumentParser(
        description="Run a PolicyRepo training experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_make_epilog(repo_name, known_keys),
    )
    parser.add_argument(
        "--repo",
        default=repo_name,
        choices=list(REPO_REGISTRY),
        help="Repository type to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--config_name",
        default="naive_repo",
        help="Hydra config name (default: naive_repo).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Parent directory for run outputs (default: outputs).",
    )

    # Dynamically add repo-specific kwargs
    for key, default_val in known_keys.items():
        parser.add_argument(
            f"--{key}",
            type=str,
            default=None,
            help=f"Default: {default_val}",
        )

    args = parser.parse_args(remaining)
    # Ensure repo is set (in case it wasn't in remaining)
    args.repo = repo_name

    # Build repo kwargs: defaults overridden by CLI
    repo_kwargs: dict[str, Any] = dict(known_keys)  # start with defaults
    for key in known_keys:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            repo_kwargs[key] = _coerce_value(cli_val)

    repo_kwargs = _post_process_kwargs(repo_name, repo_kwargs)

    return args, repo_kwargs


def _make_epilog(repo_name: str, known_keys: dict[str, Any]) -> str:
    """Build help epilogue showing repo-specific options."""
    lines = [f"\nRepo-specific options for '{repo_name}':"]
    if not known_keys:
        lines.append("  (none)")
    else:
        for k, v in known_keys.items():
            lines.append(f"  --{k}  (default: {v})")
    lines.append("\nExample:")
    parts = [f"python scripts/run.py --repo {repo_name}"]
    for k, v in list(known_keys.items())[:3]:
        parts.append(f"--{k} {v}")
    lines.append("  " + " ".join(parts))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Environment setup (mirrors notebook)
# ──────────────────────────────────────────────────────────────────────────────


def setup_environments(config: dict, rng: jax.Array):
    """Create training (and optionally eval) environments from config.

    Returns (env, env_params, static_env_params, train_env_reset_states, rng).
    Mutates config["num_train_envs"] when loading from level lists.
    """
    env, env_params, static_env_params = make_env_from_config(config)

    if config["train_level_mode"] == "random":
        rng, sub_rng = jr.split(rng)
        train_env_reset_states = get_random_envs(
            config["num_train_envs"], env, env_params, sub_rng
        )
    else:
        train_env_reset_states = load_levels(
            config["train_levels_list"],
            static_env_params_override=static_env_params,
        )
        config["num_train_envs"] = len(config["train_levels_list"])

    return env, env_params, static_env_params, train_env_reset_states, rng


# ──────────────────────────────────────────────────────────────────────────────
# WandB setup (mirrors notebook)
# ──────────────────────────────────────────────────────────────────────────────


def setup_wandb(config: dict) -> None:
    """Initialize wandb if config['use_wandb'] is True."""
    if not config.get("use_wandb", False):
        return
    wandb.init(
        project="Naive Skill Repo",
        config=dict(config),
        name=config.get("wandb_runname", None),
    )
    wandb.define_metric("repo_step")
    wandb.define_metric("videos/*", step_metric="repo_step")
    wandb.define_metric("solve_rate/*", step_metric="repo_step")
    wandb.define_metric("ep_length/*", step_metric="repo_step")
    wandb.define_metric("ep_return/*", step_metric="repo_step")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (extracted from notebook analysis cells)
# ──────────────────────────────────────────────────────────────────────────────


def generate_plots(step_logs: list, output_dir: Path) -> None:
    """Generate all 5 diagnostic plots from step_logs and save to output_dir/plots/.

    Args:
        step_logs: list of RepoStepLog namedtuples from a completed run.
        output_dir: path to the run's output directory.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    T = len(step_logs)
    P = step_logs[0].affinity_scores.shape[0]

    # ── Unpack step_logs into numpy arrays ──────────────────────────────────
    affinities = np.stack([np.array(log.affinity_scores) for log in step_logs])  # (T, P, E)
    active_masks = np.stack([np.array(log.active_mask) for log in step_logs])  # (T, E, P)
    solve_rates = np.stack([np.array(log.solve_rate) for log in step_logs])  # (T, P, E)
    avg_returns = np.stack([np.array(log.avg_return) for log in step_logs])  # (T, P, E)

    # ── Derived metrics ─────────────────────────────────────────────────────
    active_for_metric = active_masks.transpose(0, 2, 1)  # (T, P, E)

    mean_solve_rate = np.full((T, P), np.nan)
    mean_avg_return = np.full((T, P), np.nan)
    n_active_envs = np.zeros((T, P), dtype=int)

    for t in range(T):
        for p in range(P):
            active = active_for_metric[t, p]
            n_active_envs[t, p] = active.sum()
            if active.any():
                mean_solve_rate[t, p] = np.nanmean(
                    np.where(active, solve_rates[t, p], np.nan)
                )
                mean_avg_return[t, p] = np.nanmean(
                    np.where(active, avg_returns[t, p], np.nan)
                )

    EPS = 1e-12
    env_entropies = -np.sum(affinities * np.log(affinities + EPS), axis=1)  # (T, E)
    mean_env_entropy = env_entropies.mean(axis=1)
    std_env_entropy = env_entropies.std(axis=1)

    # Spearman correlation
    from scipy.stats import spearmanr

    spearman_rhos = np.full(T, np.nan)
    for t in range(T):
        aff_flat = affinities[t].ravel()
        sr_flat = solve_rates[t].ravel()
        valid = ~np.isnan(sr_flat)
        if (
            valid.sum() > 2
            and np.ptp(aff_flat[valid]) > 0
            and np.ptp(sr_flat[valid]) > 0
        ):
            rho, _ = spearmanr(aff_flat[valid], sr_flat[valid])
            spearman_rhos[t] = rho

    steps = np.arange(T)
    colors = sns.color_palette("muted", P)

    # ── Plot 1: Policy Performance Over Steps ───────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for p in range(P):
        ax1.plot(
            steps,
            mean_solve_rate[:, p],
            color=colors[p],
            lw=2,
            label=f"Policy {p}",
            marker=".",
            markersize=4,
        )
    ax1.set_ylabel("Solve Rate")
    ax1.set_title("Policy Performance on Active Envs Over Training")
    ax1.legend(ncol=P, loc="lower right")
    ax1.set_ylim(-0.05, 1.05)

    for p in range(P):
        ax2.plot(
            steps,
            mean_avg_return[:, p],
            color=colors[p],
            lw=2,
            label=f"Policy {p}",
            marker=".",
            markersize=4,
        )
    ax2.set_ylabel("Avg Return")
    ax2.set_xlabel("Repo Step")
    ax2.legend(ncol=P, loc="lower right")

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_1_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: Active Envs Per Policy ──────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.stackplot(
        steps,
        n_active_envs.T,
        labels=[f"Policy {p}" for p in range(P)],
        colors=colors,
        alpha=0.85,
    )
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Number of Active Envs")
    ax.set_title("Environments Assigned Per Policy Over Training")
    ax.legend(loc="upper right")
    ax.set_xlim(0, T - 1)

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_2_active_envs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Mean Per-Environment Entropy ────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    max_entropy = np.log(P)
    ax.plot(steps, mean_env_entropy, color="tab:blue", lw=2, label="Mean entropy")
    ax.fill_between(
        steps,
        mean_env_entropy - std_env_entropy,
        mean_env_entropy + std_env_entropy,
        alpha=0.2,
        color="tab:blue",
        label="±1 std",
    )
    ax.axhline(
        max_entropy,
        color="gray",
        ls="--",
        lw=1,
        label=f"Max (log P = {max_entropy:.2f})",
    )
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Mean Per-Env Entropy")
    ax.set_title("Assignment Ambiguity Over Training")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_3_entropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: Solve Rate by Affinity Quantile ─────────────────────────────
    n_bins = 5
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binned_solve_rate = np.full((T, n_bins), np.nan)
    bin_labels = [
        f"[{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f})" for i in range(n_bins)
    ]

    for t in range(T):
        aff_flat = affinities[t].ravel()
        sr_flat = solve_rates[t].ravel()
        valid = ~np.isnan(sr_flat)
        if valid.any():
            aff_v = aff_flat[valid]
            sr_v = sr_flat[valid]
            for b in range(n_bins):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                if b == n_bins - 1:
                    in_bin = (aff_v >= lo) & (aff_v <= hi)
                else:
                    in_bin = (aff_v >= lo) & (aff_v < hi)
                if in_bin.sum() > 0:
                    binned_solve_rate[t, b] = sr_v[in_bin].mean()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    palette = sns.color_palette("viridis", n_bins)

    for b in range(n_bins):
        ax.plot(
            steps,
            binned_solve_rate[:, b],
            color=palette[b],
            lw=2,
            label=bin_labels[b],
            marker=".",
            markersize=4,
        )
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Mean Solve Rate")
    ax.set_title("Solve Rate by Affinity Quintile Over Training")
    ax.legend(title="Affinity bin", loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(
        plots_dir / "plot_4_affinity_quantiles.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ── Plot 5: Affinity–Performance Spearman Correlation ───────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    valid = ~np.isnan(spearman_rhos)
    ax.plot(
        steps[valid],
        spearman_rhos[valid],
        color="tab:green",
        lw=2,
        marker=".",
        markersize=4,
    )
    ax.axhline(0, color="gray", ls="--", lw=1, label="ρ = 0 (no correlation)")
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Affinity–Solve Rate Correlation Over Training")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_5_spearman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 5 plots to {plots_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sanitize_config(config: dict) -> dict:
    """Convert config values to JSON-serializable types."""
    out: dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if not isinstance(x, (str, int, float, bool, type(None))) else x for x in v]
        else:
            out[k] = str(v)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args, repo_kwargs = parse_args(argv)

    # ── Resolve config path relative to this script ─────────────────────────
    script_dir = Path(__file__).resolve().parent
    config_dir = script_dir.parent / "third_party" / "kinetix" / "configs"
    if not config_dir.exists():
        print(f"ERROR: Config directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load and normalise config ───────────────────────────────────────────
    config = load_config(str(config_dir), args.config_name)
    config["seed"] = args.seed
    print(f"Loaded config '{args.config_name}' from {config_dir}")

    # ── Output directory ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_dir) / f"{args.repo}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {run_dir}")

    # ── WandB setup ─────────────────────────────────────────────────────────
    setup_wandb(config)

    # ── RNG ─────────────────────────────────────────────────────────────────
    rng = jr.PRNGKey(config["seed"])
    print(f"JAX devices: {jax.devices()}")

    # ── Environment creation ────────────────────────────────────────────────
    _, _, _, train_env_reset_states, rng = setup_environments(config, rng)
    print(
        f"Created {config['num_train_envs']} training environments "
        f"(mode: {config['train_level_mode']})"
    )

    # ── Repo initialization ─────────────────────────────────────────────────
    repo_cls, _ = REPO_REGISTRY[args.repo]
    print(f"Initializing {args.repo} repo with kwargs: {repo_kwargs}")
    policy_repo = repo_cls(config, train_env_reset_states, rng, **repo_kwargs)

    # ── Training loop ───────────────────────────────────────────────────────
    num_repo_updates = int(config["num_repo_updates"])
    print(f"Starting training: {num_repo_updates} repo updates...")
    for _ in tqdm(range(num_repo_updates), desc="Repo updates"):
        policy_repo.advance()
    print(f"Training complete. {len(policy_repo.step_logs)} steps logged.")

    # ── Save outputs ────────────────────────────────────────────────────────
    with open(run_dir / "step_logs.pkl", "wb") as f:
        pickle.dump(policy_repo.step_logs, f)

    with open(run_dir / "config.json", "w") as f:
        json.dump(_sanitize_config(config), f, indent=2, default=str)

    with open(run_dir / "repo_kwargs.json", "w") as f:
        json.dump(repo_kwargs, f, indent=2, default=str)

    print(f"Saved step_logs.pkl, config.json, repo_kwargs.json to {run_dir}")

    # ── Generate plots ──────────────────────────────────────────────────────
    if len(policy_repo.step_logs) > 0:
        generate_plots(policy_repo.step_logs, run_dir)
    else:
        print("Warning: No step_logs collected — skipping plots.")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    if config.get("use_wandb", False):
        wandb.finish()

    print(f"Done. All outputs in {run_dir}")


if __name__ == "__main__":
    main()
