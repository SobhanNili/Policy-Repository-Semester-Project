#!/usr/bin/env python3
"""Experiment runner for PolicyRepo-based skill repository training.

Reads experiment configuration from ``configs/config.yaml`` (Hydra), which
composes kinetix config groups via ``hydra.searchpath``.  Repo-specific
hyperparameters are defined in ``configs/repo/<type>.yaml``.

Usage:
    python scripts/run.py --repo random
    python scripts/run.py --repo dino --sigma 0.3 --temperature 1.0
    python scripts/run.py --repo performance --mode rank

Repos available: random, pixel, morphology, performance, dino.
Use ``python scripts/run.py --repo <type> --help`` to see repo-specific options.

Each run creates a timestamped output directory containing:
    - step_logs.pkl      : list of RepoStepLog namedtuples
    - config.json         : resolved configuration dict (from Hydra)
    - repo_kwargs.json    : repo-specific constructor kwargs used for this run
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
    PolicyLatentMatchingRepo,
    PerformanceMatchingRepo,
    RandomMatchingRepo,
)

# ──────────────────────────────────────────────────────────────────────────────
# Repo registry: repo_type string → (class, list of constructor kwargs keys)
#
# The actual default values for each kwarg live in ``configs/repo/<type>.yaml``
# and are resolved by Hydra at config-load time.  This registry only needs to
# know which keys to extract from the flattened config dict and pass to the
# repo constructor.
# ──────────────────────────────────────────────────────────────────────────────

REPO_REGISTRY: dict[str, tuple[type, list[str]]] = {
    "random_matching":      (RandomMatchingRepo,      []),
    "morphology_matching":  (MorphologyMatchingRepo,   ["history_decay", "sigma", "feature_weights"]),
    "performance_matching": (PerformanceMatchingRepo,  ["mode", "policy_eval_ep_count", "hybrid_return_weight"]),
    "dino_matching":        (DinoMatchingRepo,         ["model_name", "history_decay", "sigma"]),
    "network_embedding_matching": (PolicyLatentMatchingRepo, ["checkpoint_path", "history_decay", "sigma"]),
}

# Short aliases for the CLI (--repo dino → repo_type = dino_matching)
REPO_ALIASES: dict[str, str] = {
    "random":      "random_matching",
    "morphology":  "morphology_matching",
    "performance": "performance_matching",
    "dino":        "dino_matching",
    "network":     "network_embedding_matching",
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments.

    Returns (args, hydra_overrides) where:
        args             — parsed Namespace (repo, seed, output_dir)
        hydra_overrides  — extra CLI args converted to Hydra override strings
                           (e.g. ["sigma=0.3", "temperature=1.0"])
    """
    parser = argparse.ArgumentParser(
        description="Run a PolicyRepo training experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_make_epilog(),
    )
    parser.add_argument(
        "--repo", required=True,
        choices=list(REPO_ALIASES),
        help="Repository type (short alias).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config).",
    )
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Parent directory for run outputs (default: outputs).",
    )
    # --config_name is hidden; most users should use the default config.yaml
    parser.add_argument(
        "--config_name", default="config",
        help=argparse.SUPPRESS,
    )

    args, remaining = parser.parse_known_args(argv)

    # Convert remaining --key value pairs to Hydra override format "key=value"
    hydra_overrides = _remaining_to_hydra_overrides(remaining)

    return args, hydra_overrides


def _remaining_to_hydra_overrides(remaining: list[str]) -> list[str]:
    """Convert ``--key value`` pairs to Hydra ``key=value`` overrides.

    Supports:
      --key value   → key=value
      --key         → key=true        (boolean flag, no following value)

    Keys that match a known repo kwarg are automatically prefixed with
    ``repo.`` so Hydra applies them to the correct config group
    (e.g. ``--sigma 0.3`` → ``repo.sigma=0.3``).
    """
    # Build a set of all known repo kwargs keys
    _known_repo_keys: set[str] = set()
    for _cls, keys in REPO_REGISTRY.values():
        _known_repo_keys.update(keys)

    overrides: list[str] = []
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith("--"):
            key = arg[2:]  # strip leading --
            if i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                val = remaining[i + 1]
                i += 2
            else:
                val = "true"
                i += 1
            # Prefix repo-specific keys so Hydra finds them
            qualified_key = f"repo.{key}" if key in _known_repo_keys else key
            overrides.append(f"{qualified_key}={val}")
        else:
            i += 1
    return overrides


def _make_epilog() -> str:
    """Build help epilogue with available repos and examples."""
    lines = [
        "\nAvailable repos:",
    ]
    for alias, full in REPO_ALIASES.items():
        _, kwargs_keys = REPO_REGISTRY[full]
        keys_str = ", ".join(f"--{k}" for k in kwargs_keys) if kwargs_keys else "(no extra options)"
        lines.append(f"  {alias:<14} → {full:<24}  {keys_str}")

    lines += [
        "",
        "Examples:",
        "  python scripts/run.py --repo random",
        "  python scripts/run.py --repo dino --sigma 0.3 --temperature 1.0",
        "  python scripts/run.py --repo performance --mode rank",
        "",
        "Any config key can be overridden via the command line:",
        "  python scripts/run.py --repo dino --seed 42 --num_repo_updates 20",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Environment setup (mirrors notebook)
# ──────────────────────────────────────────────────────────────────────────────


def _extract_repo_kwargs(config: dict, kwargs_keys: list[str],
                         repo_type: str) -> dict[str, Any]:
    """Pull repo constructor kwargs from the flattened config dict.

    Applies any necessary type conversions (e.g. feature_weights list → tuple).
    """
    kwargs = {k: config[k] for k in kwargs_keys if k in config}
    # MorphologyMatchingRepo expects feature_weights as a tuple of floats
    if repo_type == "morphology_matching" and "feature_weights" in kwargs:
        fw = kwargs["feature_weights"]
        if isinstance(fw, list):
            kwargs["feature_weights"] = tuple(float(x) for x in fw)
        elif isinstance(fw, str):
            fw = fw.strip("()")
            kwargs["feature_weights"] = tuple(float(x.strip()) for x in fw.split(","))
    return kwargs


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

    # ── Utility margin: per-env gap between best and second-best raw score ──
    # Sorted along policy axis: best (index -1), second-best (index -2)
    sorted_aff = np.sort(affinities, axis=1)  # (T, P, E) — ascending
    per_env_margins = sorted_aff[:, -1, :] - sorted_aff[:, -2, :]  # (T, E)
    mean_margin = per_env_margins.mean(axis=1)
    std_margin = per_env_margins.std(axis=1)

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

    # ── Plot 3: Assignment Decisiveness (Utility Margin) ────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    ax.plot(steps, mean_margin, color="tab:blue", lw=2, label="Mean margin")
    ax.fill_between(
        steps,
        mean_margin - std_margin,
        mean_margin + std_margin,
        alpha=0.2,
        color="tab:blue",
        label="±1 std",
    )
    ax.axhline(0, color="gray", ls="--", lw=1, label="Zero margin (ties)")
    ax.set_xlabel("Repo Step")
    ax.set_ylabel("Mean Utility Margin")
    ax.set_title("Assignment Decisiveness Over Training")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(plots_dir / "plot_3_margin.png", dpi=150, bbox_inches="tight")
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
    args, hydra_overrides = parse_args(argv)

    # Resolve full repo_type from CLI alias
    repo_type = REPO_ALIASES[args.repo]

    # ── Resolve config path relative to this script ─────────────────────────
    script_dir = Path(__file__).resolve().parent
    config_dir = script_dir.parent / "configs"
    if not config_dir.exists():
        print(f"ERROR: Config directory not found: {config_dir}", file=sys.stderr)
        sys.exit(1)

    # Build Hydra overrides: repo selection + any CLI key=value pairs
    overrides = [f"repo={args.repo}"] + hydra_overrides

    # ── Load and normalise config (Hydra resolves defaults + overrides) ─────
    config = load_config(str(config_dir), args.config_name, overrides=overrides)
    if args.seed is not None:
        config["seed"] = args.seed
    print(f"Loaded config '{args.config_name}' from {config_dir}")
    print(f"  Repo: {repo_type}")

    # ── Extract repo-specific kwargs from flattened config ──────────────────
    repo_cls, kwargs_keys = REPO_REGISTRY[repo_type]
    repo_kwargs = _extract_repo_kwargs(config, kwargs_keys, repo_type)
    print(f"  Repo kwargs: {repo_kwargs}")

    # ── Output directory ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_dir) / f"{repo_type}_{timestamp}"
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
    print(f"Initializing {repo_type} repo with kwargs: {repo_kwargs}")
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
