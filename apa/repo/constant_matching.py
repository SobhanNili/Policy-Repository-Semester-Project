from apa.repo import PolicyRepo, PolicyState
import jax.random as jr
import jax.numpy as jnp
import jax

class ConstantMatchingRepo(PolicyRepo):
    """Always pairs same env to a policy. Only to test if the training is being done properly."""
    def __init__(self, config, batched_env_reset_states, rng) -> None:
        super().__init__(config, batched_env_reset_states, rng)

        self.rng, sub_rng = jr.split(self.rng)
        self.constant_policy2env_pairing = jr.choice(sub_rng,self.num_distinct_envs,shape=(self.num_policies,),replace=False)

    def match(self) -> jax.Array:
        return self.constant_policy2env_pairing
    
    def update(self, policy_state: PolicyState, per_env_perf_history: dict, policy2env_pairing: jax.Array) -> None:
        self.policy_repo = policy_state # for now not paying attention to results

    def evaluate(self) -> jax.Array:
        raise NotImplementedError