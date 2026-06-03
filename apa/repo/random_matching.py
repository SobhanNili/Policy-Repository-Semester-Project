from apa.repo import PolicyRepo, PolicyState
import jax.random as jr
import jax

class RandomMatchingRepo(PolicyRepo):
    def __init__(self, config, unique_env_reset_states, rng) -> None:
        super().__init__(config, unique_env_reset_states, rng)

    def match(self) -> jax.Array:
        """Assigns each policy to a uniformly random environment.
        
        Returns:
            jax.Array: Assignment array of shape (num_policies,).
        """
        self.rng, sub_rng = jr.split(self.rng)
        
        # one env per policy
        policy2env_pairing = jr.choice(sub_rng,self.num_distinct_envs,shape=(self.num_policies,),replace=True)

        return policy2env_pairing
    
    def update(self, policy_state: PolicyState, per_env_perf_history: dict, policy2env_pairing: jax.Array) -> None:
        """Updates the repository with the trained policy state without tracking performance history.
        
        Args:
            policy_state (PolicyState): The updated policy networks after PPO training.
            per_env_perf_history (dict): Performance metrics collected across the training phase (ignored here).
            policy2env_pairing (jax.Array): The environments assigned in the recent training step (ignored here).
        """
        self.policy_repo = policy_state # for now not paying attention to results

    def evaluate(self) -> jax.Array:
        """Evaluates the repository by randomly assigning a policy to every distinct environment.
        
        Returns:
            jax.Array: Array of shape (num_distinct_envs, 2) containing [solve_rate, avg_return].
        """
        self.rng, sub_rng = jr.split(self.rng)
        
        # assign one random policy to each distinct environment
        env2policy_pairing = jr.choice(sub_rng, self.num_policies, shape=(self.num_distinct_envs,), replace=True)
        
        return self.evaluate_pairing(env2policy_pairing)

