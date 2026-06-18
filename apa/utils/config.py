from kinetix.environment import ActionType, ObservationType
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

def normalise_config(config):
    keys = ["env", "learning", "model", "misc", "eval", "ued",
            "env_size", "train_levels", "repo"]
    for k in keys:
        if k not in config or config[k] is None:
            config[k] = {}
        small_d = config[k]
        del config[k]
        for kk, vv in small_d.items():
            assert kk not in config, kk
            config[kk] = vv

    config["observation_type_str"] = config["observation_type"]
    config["observation_type"] = ObservationType.from_string(config["observation_type"])
    config["action_type_str"] = config["action_type"]
    config["action_type"] = ActionType.from_string(config["action_type"])

    if config["num_train_envs"] % config["num_minibatches"] != 0:
        raise ValueError("num_train_envs must be divisible by num_minibatches")
    if config["num_minibatches"] > config["num_train_envs"]:
        raise ValueError("num_minibatches must be <= num_train_envs")

    return config

def load_config(config_path: str, config_name: str,
                overrides: list[str] | None = None):
    """Load and normalise a Hydra config.

    Args:
        config_path: path to the Hydra config directory.
        config_name: name of the primary config file (without .yaml).
        overrides: optional list of Hydra override strings (e.g. ["repo=dino", "sigma=0.5"]).

    Returns:
        Flattened dict with all config group keys promoted to top level.
    """
    config_dir = Path(config_path).resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return normalise_config(OmegaConf.to_container(cfg, resolve=True))