from apa.repo import PolicyRepo, PolicyState
from apa.utils.ppo import perform_generic_jitted_rollout
import jax.random as jr
import jax.numpy as jnp
import jax
import numpy as np


class EvalRepo(PolicyRepo):
    def __init__(self, config, unique_env_reset_states, rng, env_pool_size: int, policy_eval_ep_count: int) -> None:
        # env_pool_size: number of envs that the policy is tested on at each repo step before choosing one to get triained on 
        # policy_eval_ep_count: number of min episodes (in pessimistic way of episodes taking all the 256 steps) for the eval duration
        super().__init__(config, unique_env_reset_states, rng)
        KINETIX_TRUNCATION_STEPS = 256
        self.env_pool_size = env_pool_size
        self.num_eval_steps = KINETIX_TRUNCATION_STEPS * policy_eval_ep_count

    def match(self) -> jax.Array:
        self.rng, sub_rng = jr.split(self.rng)
        
        # 1. For each policy, sample a pool of unique candidate environment indices
        def sample_pool(rng):
            return jr.choice(rng, self.num_distinct_envs, shape=(self.env_pool_size,), replace=False)
        
        env_pool_indices = jax.vmap(sample_pool)(jr.split(sub_rng, self.num_policies)) # (P, E_pool)
        
        # Extract environment states / obs based on indices
        pool_envs = jax.tree.map(lambda x: x[env_pool_indices], self.unique_env_reset_states) # (P, E_pool, ...)
        pool_obs = jax.vmap(jax.vmap(self.env.get_obs))(pool_envs.env_state) # (P, E_pool, ...)
        
        # Expand policy weights & hidden states across the chosen E_pool
        params = self.policy_repo.train_state.params # (P, ...)
        hstate = jax.tree.map(lambda x: x[:, 0], self.policy_repo.hstate) # Use first copy context: (P, ...)
        hstate_pool = jax.tree.map(lambda x: jnp.repeat(x[:, None, ...], self.env_pool_size, axis=1), hstate) # (P, E_pool, ...)
        
        def network_apply(p, h, o):
            def single_policy_apply(p_, h_, o_):
                ac_in = (
                    jax.tree.map(lambda x: x[None, ...], o_), 
                    jnp.zeros((1, self.env_pool_size), dtype=bool) 
                )
                new_h, pi, _ = self.network.apply(p_, h_, ac_in)
                return new_h, pi
            return jax.vmap(single_policy_apply)(p, h, o)
            
        def env_step(rng, states, actions, resets):
            step_rngs = jr.split(rng, self.num_policies * self.env_pool_size).reshape(
                (self.num_policies, self.env_pool_size, -1)
            )
            # Add an inner vmap layer to loop over env_pool_size
            return jax.vmap(jax.vmap(self.env.step, in_axes=(0, 0, 0, None, 0)), in_axes=(0, 0, 0, None, 0))(
                step_rngs, states, actions, self.env_params, resets
            )
            
        self.rng, rollout_rng = jr.split(self.rng)
        info = perform_generic_jitted_rollout(
            network_apply, env_step, params, hstate_pool, pool_envs, pool_obs, 
            pool_envs, rollout_rng, self.num_eval_steps
        )
        
        returned_episodes = info.get("returned_episode", jnp.zeros(()))
        solved = info.get("returned_episode_solved", jnp.zeros(()))
        
        # Calculate metric values over evaluated horizon
        episodes_completed = jnp.sum(returned_episodes, axis=0) # (P, E_pool)
        sum_solved = jnp.sum(jnp.where(returned_episodes, solved, 0), axis=0) # (P, E_pool)
        
        solve_rate = jnp.where(episodes_completed > 0, sum_solved / episodes_completed, 0.0) # (P, E_pool)
        
        # Calculate Learnability = P_success * (1 - P_success)
        learnability = solve_rate * (1.0 - solve_rate) # Maps bounds [0, 1] toward maximum values ~0.5
        
        # Select best env out of the candidate pool per policy
        self.rng, noise_rng = jr.split(self.rng)
        noise = jr.uniform(noise_rng, learnability.shape, minval=0.0, maxval=1e-5)
        best_pool_indices = jnp.argmax(learnability + noise, axis=1) # (P,)
        
        self._last_chosen_learnability = jnp.max(learnability, axis=1) # (P,)
        
        policy2env_pairing = jax.vmap(lambda p_idx, b_idx: env_pool_indices[p_idx, b_idx])(
            jnp.arange(self.num_policies), best_pool_indices
        )
        
        return policy2env_pairing
    
    def update(self, policy_state: PolicyState, per_env_perf_history: dict, policy2env_pairing: jax.Array) -> None:
        self.policy_repo = policy_state
        
        # Calculate & log historic training solve rate over each matched session
        policy2env_pairing_col = policy2env_pairing.reshape((self.num_policies, 1))
        policy_step_solve_rate = jnp.zeros((self.num_policies, 1))

        for policy_idx in range(self.num_policies):
            policy_solved = np.array(per_env_perf_history['solved'][policy_idx])
            solve_rate = np.mean(policy_solved) if len(policy_solved) > 0 else 0.0
            policy_step_solve_rate = policy_step_solve_rate.at[policy_idx].set(float(solve_rate))
            
        learnability_col = self._last_chosen_learnability.reshape((self.num_policies, 1))
            
        if not hasattr(self, "policy_perf_hist") or self.policy_perf_hist is None:
            self.policy_perf_hist = {
                "matched_envs": policy2env_pairing_col, 
                "step_perf": policy_step_solve_rate,
                "chosen_learnability": learnability_col
            }
        else:
            self.policy_perf_hist["matched_envs"] = jnp.concatenate(
                [self.policy_perf_hist["matched_envs"], policy2env_pairing_col], axis=1
            )
            self.policy_perf_hist["step_perf"] = jnp.concatenate(
                [self.policy_perf_hist["step_perf"], policy_step_solve_rate], axis=1
            )
            if "chosen_learnability" in self.policy_perf_hist:
                self.policy_perf_hist["chosen_learnability"] = jnp.concatenate(
                    [self.policy_perf_hist["chosen_learnability"], learnability_col], axis=1
                )
            else:
                self.policy_perf_hist["chosen_learnability"] = learnability_col
    
    def evaluate(self) -> jax.Array:
        # Retrieve param grid map layout
        params = self.policy_repo.train_state.params # (P, ...)
        hstate = jax.tree.map(lambda x: x[:, 0], self.policy_repo.hstate) # (P, ...)
        
        # Broadcast completely to pairs for evaluations: shape becomes (P, num_distinct_envs, ...)
        hstate_all = jax.tree.map(lambda x: jnp.repeat(x[:, None, ...], self.num_distinct_envs, axis=1), hstate)
        all_envs = jax.tree.map(lambda x: jnp.repeat(x[None, ...], self.num_policies, axis=0), self.unique_env_reset_states)
        all_obs = jax.vmap(jax.vmap(self.env.get_obs))(all_envs.env_state)
        
        def network_apply(p, h, o):
            def single_policy_apply(p_, h_, o_):
                ac_in = (
                    jax.tree.map(lambda x: x[None, ...], o_), 
                    jnp.zeros((1, self.num_distinct_envs), dtype=bool) 
                )
                new_h, pi, _ = self.network.apply(p_, h_, ac_in)
                return new_h, pi
            return jax.vmap(single_policy_apply)(p, h, o)
            
        def env_step(rng, states, actions, resets):
            step_rngs = jr.split(rng, self.num_policies * self.num_distinct_envs).reshape(
                (self.num_policies, self.num_distinct_envs, -1)
            )
            # Two nested vmaps to handle (num_policies, num_distinct_envs)
            return jax.vmap(jax.vmap(self.env.step, in_axes=(0, 0, 0, None, 0)), in_axes=(0, 0, 0, None, 0))(
                step_rngs, states, actions, self.env_params, resets
            )
            
        self.rng, rollout_rng = jr.split(self.rng)
        info = perform_generic_jitted_rollout(
            network_apply, env_step, params, hstate_all, all_envs, all_obs, 
            all_envs, rollout_rng, self.env_params.max_timesteps
        )
        
        returned_episodes = info.get("returned_episode", jnp.zeros(()))
        solved = info.get("returned_episode_solved", jnp.zeros(()))
        
        episodes_completed = jnp.sum(returned_episodes, axis=0) # (P, E_distinct)
        sum_solved = jnp.sum(jnp.where(returned_episodes, solved, 0), axis=0) # (P, E_distinct)
        solve_rate = jnp.where(episodes_completed > 0, sum_solved / episodes_completed, 0.0) # (P, E_distinct)
        
        # Select best policy per distinct env (axis 0 of P x E matrix represents P, meaning argmax yields E slice indices)
        self.rng, sub_rng = jr.split(self.rng)
        noise = jr.uniform(sub_rng, solve_rate.shape, minval=0.0, maxval=1e-5)
        best_policy_per_env = jnp.argmax(solve_rate + noise, axis=0) # (num_distinct_envs,)
        
        # Execute the primary evaluate routine across the selected combinations to standardize the output format
        return super().evaluate_pairing(best_policy_per_env)

    