from apa.repo import PolicyRepo, PolicyState
import jax.random as jr
import jax.numpy as jnp
import jax
from typing import NamedTuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class PolicyPerfHistory: # supposed to be vmapped. Skill index is same as the batch index
    matched_envs: jnp.ndarray
    step_perf: jnp.ndarray

    # (P, T)
    def is_empty(self):
        return self.matched_envs is None or self.step_perf is None

class PixelMatchingRepo(PolicyRepo):
    def __init__(self, config, batched_env_reset_states, rng, env_pool_size: int, history_window: int, policy_success_threshold: float = 0.5) -> None:
        """Initializes a repository that matches policies to visually similar environments.
        
        Args:
            config: Configuration dictionary.
            batched_env_reset_states: Batched initial level states.
            rng: JAX random key.
            env_pool_size (int): Number of candidate environments sampled to form a matching pool.
            history_window (int): Number of past repository steps to consider when tracking a policy's successes.
            policy_success_threshold (float): The minimum average solve rate needed for an environment to be considered a 'success' in history.
        """
        # history_window: window of how much many past repo steps to consider
        # env_pool_size: number of envs the policies can get matched to
        super().__init__(config, batched_env_reset_states, rng)
        self.initial_pixel_obs = jax.vmap(lambda s: self.pixel_renderer(s.env_state))(self.unique_env_reset_states).image # (num_distinct_envs, 125, 125, 3)
        self.env_pool_size = env_pool_size
        self.history_window = history_window
        self.policy_success_threshold = policy_success_threshold
        self.policy_perf_hist = PolicyPerfHistory(None,None)

    def match(self) -> jax.Array:
        """Assigns policies to environments based on visual similarity to past successes.
        
        For each policy, it samples a candidate pool of `env_pool_size` environments and scores them.
        The candidate visually closest to the policy's previously solved environments is chosen.
        For the first step (no history), environments are assigned completely uniformly at random.
        
        Returns:
            jax.Array: Assignment array of shape (num_policies,).
        """
        self.rng, sub_rng = jr.split(self.rng)
        if self.policy_perf_hist.matched_envs is None:
            # random assignment for the first step
            policy2env_pairing = jr.choice(sub_rng,self.num_distinct_envs,shape=(self.num_policies,),replace=True)
        else:
            env_pool_indices = jr.choice(sub_rng,self.num_distinct_envs,shape=(self.env_pool_size,),replace=False)
            policy2env_scores = self.score_policy2env_pairing(env_pool_indices) # Shape: (P, E_pool)
            
            # Get the index of the max score in the pool for each policy
            self.rng, sub_rng = jr.split(self.rng)
            noise = jr.uniform(sub_rng, policy2env_scores.shape, minval=0.0, maxval=1e-5) # for tie-breaking in the first step, so not all policies fall into the same env. 
            best_pool_indices = jnp.argmax(policy2env_scores + noise, axis=1) # Shape: (P,)
            
            # Map back to the actual environment indices
            policy2env_pairing = env_pool_indices[best_pool_indices] # Shape: (P,)
            
        return policy2env_pairing
    
    def score_policy2env_pairing(self, env_pool_indices: jnp.ndarray) -> jnp.ndarray:
        """Scores candidate environments based on pixel-wise similarity to a policy's past successes.
        
        The score is inversely proportional to the mean-squared-error (MSE) between the initial pixel 
        observation of the candidate environment and the most visually similar environment the policy 
        has successfully solved in the recent `history_window`. 
        
        Args:
            env_pool_indices (jnp.ndarray): Indices of the candidate environment pool.
            
        Returns:
            jnp.ndarray: Scores array of shape (num_policies, len(env_pool_indices)).
        """
        pool_pixel_obs = self.initial_pixel_obs[env_pool_indices] # (E_pool, 125, 125, 3)

        # Fix: Check the time dimension (axis 1), and slice the last `history_window` steps
        if self.policy_perf_hist.matched_envs.shape[1] >= self.history_window:
            filtered_matched_env = self.policy_perf_hist.matched_envs[:, -self.history_window:]
            filtered_step_perf = self.policy_perf_hist.step_perf[:, -self.history_window:]
        else:
            filtered_matched_env = self.policy_perf_hist.matched_envs
            filtered_step_perf = self.policy_perf_hist.step_perf

        # Define the scoring logic for a single policy
        def score_single_policy(past_envs, past_perfs):
            # past_envs: (T,), past_perfs: (T,)
            success_mask = past_perfs > self.policy_success_threshold # (T,)
            past_obs = self.initial_pixel_obs[past_envs] # (T, 125, 125, 3)

            # Define how to score a single candidate environment from the pool
            def score_candidate_env(candidate_obs):
                # Calculate pixel distance (e.g. Mean Squared Error) to all past environments
                # candidate_obs: (125, 125, 3) broadcasted against past_obs: (T, 125, 125, 3)
                mse_dist = jnp.mean(jnp.square(past_obs.astype(jnp.float32) - candidate_obs.astype(jnp.float32)), axis=(-3, -2, -1)) # (T,)

                # Mask out unsuccessful environments by forcing their distance to infinity
                valid_dists = jnp.where(success_mask, mse_dist, jnp.inf)
                
                # Get the shortest distance 
                min_dist = jnp.min(valid_dists)

                # Score is 1 / dist. Add a small epsilon (1e-8) to prevent division by zero
                score = 1.0 / (min_dist + 1e-8)

                # Return 0 if the policy has NO successful environments in its history
                return jnp.where(jnp.any(success_mask), score, 0.0)

            # Vectorize over all candidate environments in the pool
            return jax.vmap(score_candidate_env)(pool_pixel_obs)

        # Vectorize over all policies
        policy2env_scores = jax.vmap(score_single_policy)(filtered_matched_env, filtered_step_perf)
        
        return policy2env_scores
    
    def update(self, policy_state: PolicyState, per_env_perf_history: dict, policy2env_pairing: jax.Array) -> None:
        """Saves current policy weights and appends outcomes to the tracking history.
        
        This history of pairings and solve rates is central to the PixelMatching's 
        scoring mechanism in the next `match` phase.
        
        Args:
            policy_state (PolicyState): The updated policy networks after PPO training.
            per_env_perf_history (dict): Performance metrics collected across the training phase.
            policy2env_pairing (jax.Array): The environments assigned in the recent training step.
        """
        self.policy_repo = policy_state # for now no pruning

        # update running stats
        policy2env_pairing = policy2env_pairing.reshape((self.num_policies,1))
        policy_step_solve_rate = jnp.zeros((self.num_policies,1))

        for policy_idx in range(self.num_policies):
            policy_solved = np.array(per_env_perf_history['solved'][policy_idx])
            solve_rate = np.mean(policy_solved) if len(policy_solved) > 0 else 0.0 # edge case for when no episode has finished
            policy_step_solve_rate = policy_step_solve_rate.at[policy_idx].set(float(solve_rate))

        
        if self.policy_perf_hist.is_empty():
            self.policy_perf_hist.matched_envs = policy2env_pairing
            self.policy_perf_hist.step_perf = policy_step_solve_rate
        else:
            # append across repo_step dim
            self.policy_perf_hist.matched_envs  = jnp.concatenate(
                [self.policy_perf_hist.matched_envs , policy2env_pairing],
                axis=1
            )
            self.policy_perf_hist.step_perf  = jnp.concatenate(
                [self.policy_perf_hist.step_perf , policy_step_solve_rate],
                axis=1
            )

    def evaluate(self) -> jax.Array:
        """Pairs each distinct environment with the policy most likely to solve it, based on visual similarity.
        
        It computes the visual similarity score of every environment against every policy's success history, 
        and assigns the highest-scoring policy to each environment for evaluation.
        
        Returns:
            jax.Array: Array of shape (num_distinct_envs, 2) containing [solve_rate, avg_return].
        """
        self.rng, sub_rng = jr.split(self.rng)
        
        if self.policy_perf_hist.is_empty():
            raise ValueError("The repo hasn't been run before evaluation attempt.")
        else:
            # Score all policies against all distinct environments
            all_envs = jnp.arange(self.num_distinct_envs)
            policy2env_scores = self.score_policy2env_pairing(all_envs) # Shape: (P, num_distinct_envs)
            
            # For tie-breaking when policies have similar scores (e.g. 0.0)
            self.rng, sub_rng = jr.split(self.rng)
            noise = jr.uniform(sub_rng, policy2env_scores.shape, minval=0.0, maxval=1e-5)
            
            # Each environment picks the policy with the highest score
            env2policy_pairing = jnp.argmax(policy2env_scores + noise, axis=0) # Shape: (num_distinct_envs,)
            
        return self.evaluate_pairing(env2policy_pairing)