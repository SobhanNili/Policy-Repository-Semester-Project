from __future__ import annotations

import optax
from flax.training.train_state import TrainState
from kinetix.models import ScannedRNN, make_network_from_config
import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kinetix.environment.env_state import EnvParams, StaticEnvParams  
    from kinetix.environment import LogWrapper

def make_network(config,env: LogWrapper,env_params: EnvParams,static_env_params: StaticEnvParams):
    """Create the policy network template (logic) from config and environment parameters.

    Network architecture paramteres are determined by config, and the env params determine the input and output shapes."""

    config = dict(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    network = make_network_from_config(env, env_params, config)
    return network

def init_random_policy(config, network, env: LogWrapper, env_params: EnvParams, n_concurrent_envs: int, rng):
    """Initialize the policy network parameters with compatible shapes to the environment.
    
    Args:
        config: The configuration dictionary containing optmizer hyperparameters.
        network: The policy network template
        env: The environment template
        env_params: The environment logic parameters
        n_concurrent_envs: Number of parallel environments to initialize the network for. The env states passed to the network will have leading shape (n_concurrent_envs, ...)
        rng: JAX PRNG key used for initilization

    Returns:
        train_state: The initialized TrainState object containing the network parameters and optimizer state
        hstate: The initialized RNN hidden state with leading shape (n_concurrent_envs, ...)
    """
    config = dict(config)

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