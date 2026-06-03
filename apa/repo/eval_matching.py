from apa.repo import PolicyRepo, PolicyState
from apa.utils.ppo import perform_generic_jitted_rollout
import jax.random as jr
import jax.numpy as jnp
import jax
import numpy as np


class EvalRepo(PolicyRepo):
    """Matches policies to environments based on learnability: P_success * (1 - P_success).

    Runs short evaluation rollouts on all candidate envs to estimate solve rates,
    then computes column-wise softmax of learnability as the affinity.
    """

    def __init__(self, config, unique_env_reset_states, rng,
                 policy_eval_ep_count: int) -> None:
        """Args:
            policy_eval_ep_count: number of episodes (pessimistically 256 steps each)
                                  for the pre-matching evaluation rollout.
        """
        super().__init__(config, unique_env_reset_states, rng)
        KINETIX_TRUNCATION_STEPS = 256
        self.num_eval_steps = KINETIX_TRUNCATION_STEPS * policy_eval_ep_count

    # ── Affinity computation ───────────────────────────────────────────────

    def compute_affinity(self, candidate_env_indices: jax.Array) -> jax.Array:
        """Run eval rollouts on all (policy × candidate env) pairs, compute
        learnability, then column-wise softmax."""
        self.rng, sub_rng = jr.split(self.rng)
        P = self.num_policies
        E_subset = candidate_env_indices.shape[0]

        # Extract environment states for candidate envs (single copy each)
        pool_envs = jax.tree.map(
            lambda x: x[candidate_env_indices], self.unique_env_reset_states
        )  # (E_subset, ...)
        # Expand to all policies: (P, E_subset, ...)
        pool_envs_all = jax.tree.map(
            lambda x: jnp.repeat(x[None, ...], P, axis=0), pool_envs
        )
        pool_obs = jax.vmap(jax.vmap(self.env.get_obs))(pool_envs_all.env_state)

        params = self.policy_repo.train_state.params
        hstate = jax.tree.map(lambda x: x[:, 0], self.policy_repo.hstate)
        hstate_all = jax.tree.map(
            lambda x: jnp.repeat(x[:, None, ...], E_subset, axis=1), hstate
        )  # (P, E_subset, ...)

        def network_apply(p, h, o):
            def single_policy_apply(p_, h_, o_):
                ac_in = (
                    jax.tree.map(lambda x: x[None, ...], o_),
                    jnp.zeros((1, E_subset), dtype=bool)
                )
                new_h, pi, _ = self.network.apply(p_, h_, ac_in)
                return new_h, pi
            return jax.vmap(single_policy_apply)(p, h, o)

        def env_step(rng, states, actions, resets):
            step_rngs = jr.split(rng, P * E_subset).reshape((P, E_subset, -1))
            return jax.vmap(jax.vmap(self.env.step, in_axes=(0, 0, 0, None, 0)),
                            in_axes=(0, 0, 0, None, 0))(
                step_rngs, states, actions, self.env_params, resets
            )

        self.rng, rollout_rng = jr.split(self.rng)
        info = perform_generic_jitted_rollout(
            network_apply, env_step, params, hstate_all, pool_envs_all, pool_obs,
            pool_envs_all, rollout_rng, self.num_eval_steps
        )

        returned_episodes = info.get("returned_episode", jnp.zeros(()))
        solved = info.get("returned_episode_solved", jnp.zeros(()))

        episodes_completed = jnp.sum(returned_episodes, axis=0)  # (P, E_subset)
        sum_solved = jnp.sum(jnp.where(returned_episodes, solved, 0), axis=0)

        solve_rate = jnp.where(episodes_completed > 0,
                               sum_solved / episodes_completed, 0.0)

        # Learnability = P_success * (1 - P_success) — peaks at 0.5
        learnability = solve_rate * (1.0 - solve_rate)

        # Column-wise softmax over policies
        affinity = jax.nn.softmax(learnability, axis=0)
        return affinity
