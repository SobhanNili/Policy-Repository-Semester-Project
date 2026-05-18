from apa.repo import PolicyRepo, PolicyState
import jax.random as jr
import jax

class RandomMatchingRepo(PolicyRepo):
    def __init__(self, config, batched_env_reset_states, rng) -> None:
        super().__init__(config, batched_env_reset_states, rng)

    def match(self) -> jax.Array:
        self.rng, sub_rng = jr.split(self.rng)
        
        # one env per policy
        policy2env_pairing = jr.choice(sub_rng,self.num_distinct_envs,shape=(self.num_policies,),replace=True)

        return policy2env_pairing
    
    def update(self, policy_state: PolicyState, per_env_perf_history: dict, policy2env_pairing: jax.Array) -> None:
        self.policy_repo = policy_state # for now not paying attention to results

    def evaluate(self) -> jax.Array:
        self.rng, sub_rng = jr.split(self.rng)
        
        # assign one random policy to each distinct environment
        env2policy_pairing = jr.choice(sub_rng, self.num_policies, shape=(self.num_distinct_envs,), replace=True)
        
        return self.evaluate_pairing(env2policy_pairing)

