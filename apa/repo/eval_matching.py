from apa.repo import PolicyRepo, PolicyState
from apa.repo.base import get_pytree_batch_size
import jax.numpy as jnp
import jax
import numpy as np


class EvalRepo(PolicyRepo):
    """Matches policies to environments based on learnability: P_success * (1 - P_success).

    Runs short evaluation rollouts on all candidate envs to estimate solve rates,
    then computes column-wise softmax of learnability as the affinity.
    """

    def __init__(self, config, train_env_reset_states, rng,
                 policy_eval_ep_count: int) -> None:
        """Args:
            policy_eval_ep_count: number of episodes (pessimistically 256 steps each)
                                  for the pre-matching evaluation rollout.
        """
        super().__init__(config, train_env_reset_states, rng)
        KINETIX_TRUNCATION_STEPS = 256
        self.num_eval_steps = KINETIX_TRUNCATION_STEPS * policy_eval_ep_count

    # ── Affinity computation ───────────────────────────────────────────────

    def compute_affinity(self, env_reset_states) -> jax.Array:
        """Run eval rollouts on all (policy × candidate env) pairs via
        ``evaluate_pairing``, compute learnability, then column-wise softmax."""
        P = self.num_policies
        E_subset = get_pytree_batch_size(env_reset_states)
        all_pairs = jnp.arange(E_subset)[None, :].repeat(P, axis=0)  # (P, E_subset)
        results = self.evaluate_pairing(all_pairs, env_reset_states,
                                        rollout_steps=self.num_eval_steps)  # (P, E_subset, 2)
        solve_rate = results[:, :, 0]
        # Learnability = P_success * (1 - P_success) — peaks at 0.5
        learnability = solve_rate * (1.0 - solve_rate)
        return jax.nn.softmax(learnability, axis=0)
