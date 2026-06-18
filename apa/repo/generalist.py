from apa.repo import PolicyRepo
import jax.numpy as jnp
from apa.repo.base import get_pytree_batch_size
import jax

class GeneralistMatchingRepo(PolicyRepo):
    """A baseline matching where all envs are assigned to a single, fixed policy over all steps. This is a degenerate case of the matching problem, but it is useful for testing and debugging."""
    
    def __init__(self, config, train_env_reset_states, rng) -> None:
        super().__init__(config, train_env_reset_states, rng)

    def compute_affinity(self, env_reset_states) -> jax.Array:
        """One hot scoring of the first policy for all envs."""
        E_subset = get_pytree_batch_size(env_reset_states)
        affinity = jnp.zeros((self.num_policies, E_subset))
        affinity = affinity.at[0].set(1.0)
        return affinity