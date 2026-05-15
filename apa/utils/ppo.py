import jax
import jax.numpy as jnp
from typing import NamedTuple, Any

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