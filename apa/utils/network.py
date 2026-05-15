import optax
from flax.training.train_state import TrainState
from kinetix.models import ScannedRNN, make_network_from_config
import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict

def make_network(config,env,env_params,static_env_params):
    config = dict(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    network = make_network_from_config(env, env_params, config)
    return network

def init_random_policy(config, network, env, env_params,n_concurrent_envs, rng):
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

def init_network_from_params(config, network, params,n_concurrent_envs):
    config = dict(config)
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