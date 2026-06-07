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
import numpy as np
from tqdm.auto import tqdm
from loguru import logger

import wandb  # noqa: E402

from apa.utils.analysis import generate_plots, save_env_observations  # noqa: E402

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
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip generating diagnostic plots after training.",
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
    """Create training and evaluation environments from config.

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

    eval_env_reset_states = load_levels(config["eval_levels"], static_env_params_override=static_env_params)

    return env, env_params, static_env_params, train_env_reset_states, eval_env_reset_states, rng


# ──────────────────────────────────────────────────────────────────────────────
# WandB setup (mirrors notebook)
# ──────────────────────────────────────────────────────────────────────────────


def setup_wandb(config: dict) -> None:
    """Initialize wandb if config['use_wandb'] is True."""
    if not config.get("use_wandb", False):
        return
    wandb.init(
        project="Policy Repo",
        config=dict(config),
        name=config.get("wandb_runname", None),
    )
    wandb.define_metric("repo_step")
    wandb.define_metric("videos/*", step_metric="repo_step")
    wandb.define_metric("solve_rate/*", step_metric="repo_step")
    wandb.define_metric("ep_length/*", step_metric="repo_step")
    wandb.define_metric("ep_return/*", step_metric="repo_step")


# (generate_plots lives in apa/utils/analysis.py)


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
        logger.error(f"ERROR: Config directory not found: {config_dir}")
        sys.exit(1)

    # Build Hydra overrides: repo selection + any CLI key=value pairs
    overrides = [f"repo={args.repo}"] + hydra_overrides

    # ── Load and normalise config (Hydra resolves defaults + overrides) ─────
    config = load_config(str(config_dir), args.config_name, overrides=overrides)
    if args.seed is not None:
        config["seed"] = args.seed
    logger.info(f"Loaded config '{args.config_name}' from {config_dir}")
    logger.info(f"  Repo: {repo_type}")

    # ── Extract repo-specific kwargs from flattened config ──────────────────
    repo_cls, kwargs_keys = REPO_REGISTRY[repo_type]
    repo_kwargs = _extract_repo_kwargs(config, kwargs_keys, repo_type)
    logger.info(f"  Repo kwargs: {repo_kwargs}")

    # ── Output directory ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_dir) / f"{repo_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info(f"Output directory: {run_dir}")

    # ── WandB setup ─────────────────────────────────────────────────────────
    setup_wandb(config)

    # ── RNG ─────────────────────────────────────────────────────────────────
    rng = jr.PRNGKey(config["seed"])
    logger.info(f"JAX devices: {jax.devices()}")

    # ── Environment creation ────────────────────────────────────────────────
    env, env_params, static_env_params, train_env_reset_states, eval_env_reset_states, rng = setup_environments(config, rng)
    logger.info(
        f"Created {config['num_train_envs']} training environments "
        f"(mode: {config['train_level_mode']})"
    )

    # ── Save env observation images ─────────────────────────────────────────
    save_env_observations(env_params, static_env_params, train_env_reset_states,
                          run_dir, folder="env_train_observations")
    save_env_observations(env_params, static_env_params, eval_env_reset_states,
                          run_dir, folder="eval_env_observations")

    # ── Repo initialization ─────────────────────────────────────────────────
    from apa.repo import PolicyRepo
    logger.info(f"Initializing {repo_type} repo with kwargs: {repo_kwargs}")
    policy_repo: PolicyRepo = repo_cls(config, train_env_reset_states, rng, **repo_kwargs)

    # ── Training loop ───────────────────────────────────────────────────────
    num_repo_updates = int(config["num_repo_updates"])
    logger.info(f"Starting training: {num_repo_updates} repo updates...")
    for _ in tqdm(range(num_repo_updates), desc="Repo updates"):
        policy_repo.advance()
    logger.info(f"Training complete. {len(policy_repo.step_logs)} steps logged.")

    # ── Evaluation step ───────────────────────────────────────────────────────
    logger.info("Starting evaluation...")   
    eval_results = policy_repo.evaluate_finetune(eval_env_reset_states)
    logger.info("Evaluation complete.")

    # ── Save outputs ────────────────────────────────────────────────────────
    with open(run_dir / "step_logs.pkl", "wb") as f:
        pickle.dump(policy_repo.step_logs, f)

    with open(run_dir / "eval_results.pkl", "wb") as f:
        pickle.dump(eval_results, f)

    with open(run_dir / "config.json", "w") as f:
        json.dump(_sanitize_config(config), f, indent=2, default=str)

    with open(run_dir / "repo_kwargs.json", "w") as f:
        json.dump(repo_kwargs, f, indent=2, default=str)

    logger.info(f"Saved step_logs.pkl, config.json, repo_kwargs.json to {run_dir}")

    # ── Generate plots ──────────────────────────────────────────────────────
    if not args.skip_plots:
        if len(policy_repo.step_logs) > 0:
            generate_plots(policy_repo.step_logs, run_dir)
        else:
            logger.warning("Warning: No step_logs collected — skipping plots.")
    else:
        logger.info("Plots skipped (--skip-plots).")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    if config.get("use_wandb", False):
        wandb.finish()

    logger.info(f"Done. All outputs in {run_dir}")


if __name__ == "__main__":
    main()
