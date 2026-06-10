#!/usr/bin/env python3
"""Train the kinetix PPO experiment using a config from this repository.

Composes config groups from ``third_party/kinetix/configs`` via
``hydra.searchpath``, with overrides defined in
``configs/ppo_training/config.yaml``.

Usage
-----
    python scripts/train_ppo.py
    python scripts/train_ppo.py save_path=checkpoints/my_run
    python scripts/train_ppo.py learning.num_train_envs=512 total_timesteps=100000000

Model checkpoints are saved to
``{save_path}/{run_name}-{hash}-{timesteps}B/full_model.pbz2``,
where ``run_name``, ``hash`` and ``timesteps`` are set by the kinetix
``normalise_config`` helper based on the resolved config values.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
from flax.serialization import to_state_dict
from hydra import compose, initialize_config_dir
from kinetix.util import generate_params_from_config, normalise_config, save_model
from omegaconf import OmegaConf
from loguru import logger

# ── Load make_train from kinetix's experiments/ppo.py ────────────────────────
# experiments/ is *not* a subpackage of kinetix, so we use importlib instead.
_here = Path(__file__).resolve().parent
_experiments_ppo = _here.parent / "third_party" / "kinetix" / "experiments" / "ppo.py"
_spec = importlib.util.spec_from_file_location("kinetix_ppo", str(_experiments_ppo))
_ppo_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ppo_mod)  # type: ignore[union-attr]
make_train = _ppo_mod.make_train


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> list[str]:
    """Extract Hydra ``key=value`` overrides from CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train the kinetix PPO experiment.",
        add_help=False,
    )
    parser.add_argument("overrides", nargs="*")
    args, _ = parser.parse_known_args(argv)
    return args.overrides


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    overrides = parse_args(sys.argv[1:])

    # Load and resolve the Hydra config from configs/ppo_training/config.yaml
    config_dir = Path(__file__).resolve().parent.parent / "configs" / "ppo_training"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)

    # Convert to dict and normalise — flattens config groups (env, learning,
    # model, misc, …), sets run_name, random_hash, num_updates, etc.
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = normalise_config(cfg, "PPO")

    # Generate environment parameters from the resolved config
    env_params, static_env_params = generate_params_from_config(cfg)
    cfg["env_params"] = to_state_dict(env_params)
    cfg["static_env_params"] = to_state_dict(static_env_params)

    # Optional wandb initialisation
    if cfg.get("use_wandb", False):
        from kinetix.util.config import init_wandb

        init_wandb(cfg, "PPO")

    # Compile the training function and run it
    rng = jax.random.PRNGKey(cfg["seed"])
    rng, _rng = jax.random.split(rng)
    train_fn = jax.jit(make_train(cfg, env_params, static_env_params))
    out = train_fn(_rng)

    # Save the final policy checkpoint
    if cfg.get("save_policy", True):
        train_state = jax.tree.map(lambda x: x, out["runner_state"][0])
        save_model(
            train_state,
            cfg["total_timesteps"],
            cfg,
            save_to_wandb=cfg.get("use_wandb", False),
        )
        logger.info(f"Checkpoint saved under: {cfg['save_path']}")


if __name__ == "__main__":
    main()
