from apa.repo import PolicyRepo, PolicyState
from apa.repo.base import get_pytree_batch_size
import jax.numpy as jnp
import jax


class RandomMatchingRepo(PolicyRepo):
    """Random affinity: every candidate env is equally likely for every policy.

    Uniform affinity + categorical sampling in the base class produces truly
    random assignment — no override of ``derive_assignment_mask`` needed.
    """

    def __init__(self, config, train_env_reset_states, rng) -> None:
        super().__init__(config, train_env_reset_states, rng)

    def compute_affinity(self, env_reset_states) -> jax.Array:
        """Uniform affinity across policies — column-wise softmax of ones = 1/P."""
        E_subset = get_pytree_batch_size(env_reset_states)
        return jnp.ones((self.num_policies, E_subset)) / self.num_policies

