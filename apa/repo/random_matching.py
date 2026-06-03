from apa.repo import PolicyRepo, PolicyState
import jax.numpy as jnp
import jax


class RandomMatchingRepo(PolicyRepo):
    """Random affinity: every candidate env is equally likely for every policy.

    Uniform affinity + categorical sampling in the base class produces truly
    random assignment — no override of ``derive_assignment_mask`` needed.
    """

    def __init__(self, config, unique_env_reset_states, rng) -> None:
        super().__init__(config, unique_env_reset_states, rng)

    def compute_affinity(self, candidate_env_indices: jax.Array) -> jax.Array:
        """Uniform affinity across policies — column-wise softmax of ones = 1/P."""
        E_subset = candidate_env_indices.shape[0]
        return jnp.ones((self.num_policies, E_subset)) / self.num_policies

