from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

from kinetix.environment import ActionType, LogWrapper, ObservationType
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.util import generate_params_from_config

import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from kinetix.models import ScannedRNN, make_network_from_config
from flax.serialization import to_state_dict
from typing import NamedTuple, Any
from tqdm.auto import tqdm
import numpy as np
import jax
from functools import partial
from kinetix.util import load_evaluation_levels
from kinetix.environment.ued.ued import make_reset_fn_list_of_levels

def normalise_config(config):
    keys = ["env", "learning", "model", "misc", "eval", "ued", "env_size", "train_levels"]
    for k in keys:
        if k not in config:
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

def load_config(config_path: str, config_name: str):
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
    return normalise_config(OmegaConf.to_container(cfg, resolve=True))

def make_env_from_config(config):
    """Create an env template from config. (ONLY ENV DEFINITION)"""
    env_params, static_env_params = generate_params_from_config(config)
    reset_fn = make_reset_fn_from_config(config, env_params, static_env_params)

    env = LogWrapper(
        make_kinetix_env(
            action_type=config["action_type"],
            observation_type=config["observation_type"],
            reset_fn=reset_fn,
            env_params=env_params,
            static_env_params=static_env_params,
        )
    )

    return env, env_params, static_env_params


def make_network(config,env,env_params,static_env_params):
    config = dict(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    network = make_network_from_config(env, env_params, config)
    return network

def init_random_policy(config, network, env, env_params, rng):
    config = dict(config)
    n_concurrent_envs = int(config.get("num_env_copies",1))

    obs, _ = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_concurrent_envs),
        env_params,
    )
    done = jnp.zeros((n_concurrent_envs,), dtype=jnp.bool_)
    hstate = ScannedRNN.initialize_carry(n_concurrent_envs)

    init_x = jax.tree.map(lambda x: x[None, ...], (obs, done))
    params = network.init(rng, hstate, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(float(config.get("max_grad_norm", 1.0))),
        optax.adam(float(config.get("lr", 3e-4)), eps=1e-5),
    )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    return train_state, hstate

def init_network_from_params(config, network, params):
    config = dict(config)
    n_concurrent_envs = int(config.get("num_env_copies",1))
    hstate = ScannedRNN.initialize_carry(n_concurrent_envs)

    tx = optax.chain(
        optax.clip_by_global_norm(float(config.get("max_grad_norm", 1.0))),
        optax.adam(float(config.get("lr", 3e-4)), eps=1e-5),
    )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    return train_state, hstate


# ---- PPO-HELPERS ----

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any
    info: Any


def ppo_update(config, train_state, traj, advantages, targets, init_hstate, rng):
    def loss_fn(params, init_hstate, traj_batch, gae, targets):
        _, pi, value = train_state.apply_fn(params, init_hstate[0], (traj_batch.obs, traj_batch.done))
        log_prob = pi.log_prob(traj_batch.action)

        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config["clip_eps"], config["clip_eps"]
        )
        value_loss = 0.5 * jnp.maximum(
            jnp.square(value - targets),
            jnp.square(value_pred_clipped - targets),
        ).mean()

        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        ratio = jnp.exp(log_prob - traj_batch.log_prob)

        policy_loss = -jnp.minimum(
            ratio * gae,
            jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * gae,
        ).mean()

        entropy = pi.entropy().mean()
        total_loss = policy_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
        return total_loss, (value_loss, policy_loss, entropy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def minibatch_update(train_state, batch):
        init_hstate, traj_batch, advantages, targets = batch
        (loss, _), grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    batch = (init_hstate, traj, advantages, targets)

    permutation = jax.random.permutation(rng, traj.action.shape[1])
    batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

    batch = jax.tree.map(
        lambda x: jnp.swapaxes(
            jnp.reshape(x, [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:])),
            1,
            0,
        ),
        batch,
    )

    def epoch_update(train_state, _):
        train_state, losses = jax.lax.scan(minibatch_update, train_state, batch)
        return train_state, losses

    train_state, losses = jax.lax.scan(epoch_update, train_state, None, length=config["update_epochs"])
    return train_state, losses


def ppo_rollout(config, env, env_params, train_state, hstate, obs, env_state, rng, reset_state_override=None):
    batch_size = jax.tree_util.tree_leaves(obs)[0].shape[0]
    init_hstate = hstate

    def env_step(carry, _):
        train_state, env_state, obs, done, hstate, rng = carry
        rng, action_rng, step_rng = jax.random.split(rng, 3)

        ac_in = (jax.tree.map(lambda x: x[None, ...], obs), done[None, :])
        hstate, pi, value = train_state.apply_fn(train_state.params, hstate, ac_in)

        action = pi.sample(seed=action_rng).squeeze(0)
        log_prob = pi.log_prob(action).squeeze(0)
        value = value.squeeze(0)

        step_keys = jax.random.split(step_rng, batch_size)
        if reset_state_override is None:
            next_obs, next_env_state, reward, next_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_keys, env_state, action, env_params)
        else:
            next_obs, next_env_state, reward, next_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None, 0)
            )(step_keys, env_state, action, env_params, reset_state_override)

        transition = Transition(done, action, value, reward, log_prob, obs, info)
        carry = (train_state, next_env_state, next_obs, next_done, hstate, rng)
        return carry, transition

    done0 = jnp.zeros((batch_size,), dtype=jnp.bool_)
    carry = (train_state, env_state, obs, done0, hstate, rng)
    carry, traj = jax.lax.scan(env_step, carry, None, length=config["num_steps"])

    train_state, env_state, obs, done, hstate, rng = carry

    ac_in = (jax.tree.map(lambda x: x[None, ...], obs), done[None, :])
    _, _, last_value = train_state.apply_fn(train_state.params, hstate, ac_in)
    last_value = last_value.squeeze(0)

    def gae_step(carry, transition):
        gae, next_value, next_done = carry
        delta = transition.reward + config["gamma"] * next_value * (1 - next_done) - transition.value
        gae = delta + config["gamma"] * config["gae_lambda"] * (1 - next_done) * gae
        return (gae, transition.value, transition.done), gae

    _, advantages = jax.lax.scan(
        gae_step,
        (jnp.zeros_like(last_value), last_value, done),
        traj,
        reverse=True,
    )
    targets = advantages + traj.value

    return init_hstate, obs, env_state, traj, advantages, targets

def make_ppo_step(config, env, env_params):
    def ppo_step(train_state, hstate, obs, env_state, rollout_rngs, update_rngs, reset_state_override=None):

        def make_ppo_rollout(train_state, hstate, obs, env_state, rng, reset_state_override):
            return ppo_rollout(
                config, env, env_params, train_state, hstate, obs, env_state, rng,
                reset_state_override=reset_state_override
            )

        def make_ppo_update(train_state, traj, advantages, targets, hstate, rng):
            return ppo_update(config, train_state, traj, advantages, targets, hstate, rng)

        hstate, obs, env_state, traj, advantages, targets = jax.vmap(make_ppo_rollout)(
            train_state, hstate, obs, env_state, rollout_rngs, reset_state_override
        )

        train_state, losses = jax.vmap(make_ppo_update)(
            train_state, traj, advantages, targets, hstate, update_rngs
        )
        return train_state, hstate, obs, env_state, losses, traj.info

    return jax.jit(ppo_step)

def load_and_replicate_levels(level_names: list[str], envs_in_batch: int):
    """
    Load predefined levels and replicate each one envs_in_batch times.
    
    Args:
        level_names: List of level names (e.g., ["s/h0_weak_thrust", "s/h7_unicycle_left"])
        envs_in_batch: Number of copies per level (ENVS_IN_BATCH)
    
    Returns:
        env_state: EnvState pytree with shape (NUM_LOADED_LEVELS, ENVS_IN_BATCH, ...)
    """
    # Load all levels (shape: (NUM_LEVELS, ...))
    levels, static_env_params = load_evaluation_levels(level_names)

    # reset_fn = make_reset_fn_list_of_levels(level_names,static_env_params)
    
    # Replicate each level envs_in_batch times
    # From (NUM_LEVELS, ...) to (NUM_LEVELS, ENVS_IN_BATCH, ...)
    env_state = jax.tree.map(
        lambda x: jnp.repeat(x[:, None, ...], envs_in_batch, axis=1),
        levels
    )
    
    return env_state

