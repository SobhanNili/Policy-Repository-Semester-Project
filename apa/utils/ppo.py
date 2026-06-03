import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Any

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any
    info: Any


def ppo_update(config, train_state, traj, advantages, targets, init_hstate, rng, env_mask=None):
    """PPO update with optional per-env loss masking.

    Args:
        env_mask: optional (B,) boolean or float mask. 1.0 = active, 0.0 = inactive.
                  Inactive envs contribute zero gradient.  If None, all envs are active.
    """
    # Normalise mask: None → all ones; ensure float32
    if env_mask is None:
        env_mask = jnp.ones(traj.action.shape[1], dtype=jnp.float32)
    else:
        env_mask = env_mask.astype(jnp.float32)

    def loss_fn(params, init_hstate, traj_batch, gae, targets, mask_batch):
        _, pi, value = train_state.apply_fn(params, init_hstate[0], (traj_batch.obs, traj_batch.done))
        log_prob = pi.log_prob(traj_batch.action)

        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config["clip_eps"], config["clip_eps"]
        )
        # Masked value loss — mask is (B_mb,) broadcast over action dims if any
        per_env_value_loss = 0.5 * jnp.maximum(
            jnp.square(value - targets),
            jnp.square(value_pred_clipped - targets),
        )
        value_loss = jnp.sum(mask_batch * per_env_value_loss) / jnp.maximum(jnp.sum(mask_batch), 1.0)

        # GAE normalisation over active envs only
        gae = (gae - jnp.sum(mask_batch * gae) / jnp.maximum(jnp.sum(mask_batch), 1.0)) / (
            jnp.sqrt(jnp.sum(mask_batch * jnp.square(gae)) / jnp.maximum(jnp.sum(mask_batch), 1.0)) + 1e-8
        )
        ratio = jnp.exp(log_prob - traj_batch.log_prob)

        per_env_policy_loss = -jnp.minimum(
            ratio * gae,
            jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * gae,
        )
        policy_loss = jnp.sum(mask_batch * per_env_policy_loss) / jnp.maximum(jnp.sum(mask_batch), 1.0)

        per_env_entropy = pi.entropy()
        entropy = jnp.sum(mask_batch * per_env_entropy) / jnp.maximum(jnp.sum(mask_batch), 1.0)
        total_loss = policy_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
        return total_loss, (value_loss, policy_loss, entropy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Permute the main batch elements and the mask separately because
    # the mask has batch on axis 0 (no leading time dimension).
    permutation = jax.random.permutation(rng, traj.action.shape[1])
    batch_no_mask = (init_hstate, traj, advantages, targets)
    batch_no_mask = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch_no_mask)
    mask_permuted = env_mask[permutation]  # (B,) permuted along axis 0

    # Reshape main elements for minibatches
    batch_no_mask = jax.tree.map(
        lambda x: jnp.swapaxes(
            jnp.reshape(x, [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:])),
            1, 0,
        ),
        tuple(batch_no_mask),
    )
    # Mask: (B,) → (num_minibatches, B_per_mb)
    mask_arr = jnp.reshape(mask_permuted, [config["num_minibatches"], -1])

    def minibatch_update(train_state, batch):
        init_hstate, traj_batch, advantages, targets, mask_batch = batch
        (loss, _), grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets, mask_batch)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    batch = batch_no_mask + (mask_arr,)

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
    def ppo_step(train_state, hstate, obs, env_state, rollout_rngs, update_rngs, reset_state_override=None, env_mask=None):
        """Executes PPO rollout + update for all policies.

        Args:
            env_mask: optional (P, B) float/bool mask. 1.0 = active env for that policy.
                      Passed through to ppo_update for loss masking.
        """

        def make_ppo_rollout(train_state, hstate, obs, env_state, rng, reset_state_override):
            return ppo_rollout(
                config, env, env_params, train_state, hstate, obs, env_state, rng,
                reset_state_override=reset_state_override
            )

        def make_ppo_update(train_state, traj, advantages, targets, hstate, rng, mask):
            return ppo_update(config, train_state, traj, advantages, targets, hstate, rng, env_mask=mask)

        hstate, obs, env_state, traj, advantages, targets = jax.vmap(make_ppo_rollout)(
            train_state, hstate, obs, env_state, rollout_rngs, reset_state_override
        )

        if env_mask is None:
            P, B = jax.tree_util.tree_leaves(obs)[0].shape[:2]
            env_mask = jnp.ones((P, B), dtype=jnp.float32)

        train_state, losses = jax.vmap(make_ppo_update)(
            train_state, traj, advantages, targets, hstate, update_rngs, env_mask
        )
        return train_state, hstate, obs, env_state, losses, traj.info

    return jax.jit(ppo_step)

@partial(jax.jit, static_argnames=['network_apply', 'env_step', 'rollout_length'])
def perform_generic_jitted_rollout(network_apply, env_step, params, hstate, env_state, obs, resets, rng, rollout_length):
    """
    A generic jitted rollout utility function.
    Given matching Pytrees constraints sizes (usually P x B matrices setup), rolls out execution for given steps.
    """
    def eval_step(carry, _):
        env_state, obs, hstate, step_rng = carry
        
        # Split rng for action selection
        step_rng, pi_rng = jax.random.split(step_rng)
        
        # Usually obs needs an extra time dimension prepended -> (1, ...) for the network apply
        # We assume the caller handles whatever dimension constraints inside network_apply closure
        hstate, pi = network_apply(params, hstate, obs)
        
        action = jnp.squeeze(pi.sample(seed=pi_rng)) # Removes the temporary Time dim
        
        step_rng, env_rng = jax.random.split(step_rng)
        
        next_obs, next_env_state, _, _, info = env_step(
            env_rng, env_state, action, resets
        )
        
        return (next_env_state, next_obs, hstate, step_rng), info

    _, info = jax.lax.scan(
        eval_step, 
        (env_state, obs, hstate, rng), 
        None, 
        length=rollout_length
    )
    return info