from apa.repo import PolicyRepo, PolicyState
from apa.repo.base import get_pytree_batch_size
import jax.random as jr
import jax.numpy as jnp
import jax


class ConstantMatchingRepo(PolicyRepo):
    """Always pairs the same envs to the same policies.  A test harness that
    verifies the training pipeline works correctly.

    Flattens the constant (P, candidate_pool_size) pairing into a 1-D candidate
    set so every env in the pairing is present every step.  One-hot affinity
    routes each env's gradient to its owning policy.
    """

    def __init__(self, config, train_env_reset_states, rng) -> None:
        super().__init__(config, train_env_reset_states, rng)

        self.rng, sub_rng = jr.split(self.rng)
        # Each policy gets candidate_pool_size distinct envs, constant forever.
        def sample_for_policy(rng):
            return jr.choice(rng, self.num_distinct_envs,
                             shape=(self.candidate_pool_size,), replace=False)
        rngs = jr.split(sub_rng, self.num_policies)
        self.constant_pairing = jax.vmap(sample_for_policy)(rngs)  # (P, candidate_pool_size)

    def sample_candidate_envs(self) -> jax.Array:
        """Returns the flattened constant pairing — every pre-assigned env,
        every step.  Size = P * candidate_pool_size (preserves config validation)."""
        return self.constant_pairing.reshape(-1)  # (P * candidate_pool_size,)

    def compute_affinity(self, env_reset_states) -> jax.Array:
        """One-hot affinity built from the known (P, S) block structure.

        ``sample_candidate_envs`` returns ``constant_pairing.reshape(-1)``,
        so position *i* belongs to policy ``p = i // candidate_pool_size``.
        """
        P = self.num_policies
        S = self.candidate_pool_size
        E_subset = get_pytree_batch_size(env_reset_states)  # = P * S
        affinity = jnp.zeros((P, E_subset))
        for i in range(E_subset):
            p = i // S
            affinity = affinity.at[p, i].set(1.0)
        return affinity

    def derive_assignment_mask(self, affinity: jax.Array) -> jax.Array:
        """Pure argmax — no force-at-least-one post-processing.

        The constant pairing may produce policies with all-zero affinity rows
        when multiple policies happen to draw the same env.  Those policies
        should genuinely get no active envs rather than being forcibly assigned
        to a random env (which would break the test-harness invariant).
        """
        E_subset = affinity.shape[1]
        assignments = jnp.argmax(affinity, axis=0)  # (E_subset,)
        mask = jnp.zeros((E_subset, self.num_policies), dtype=bool)
        mask = mask.at[jnp.arange(E_subset), assignments].set(True)
        return mask
