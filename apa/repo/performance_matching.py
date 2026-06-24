"""Repository that assigns policies to environments based on frozen-policy performance.

Runs evaluation rollouts on all P x E_subset pairs at each repo step, then
derives affinity scores from the results.
"""

from apa.repo import PolicyRepo
from apa.repo.base import get_pytree_batch_size
import jax.numpy as jnp
import jax


class PerformanceMatchingRepo(PolicyRepo):
    """Matches policies to environments based on frozen-policy evaluation rollouts.
    Acts as sort of of exhaustive search over the policy space, scoring each policy on each candidate environment.  
    
    Three scoring modes:
      - ``"return_rank"``: Rank policies per env by average return (1 = best).
        Nonparametric and scale-invariant. Handles negative returns.
      - ``"solve_rate"``: Use the fraction of episodes solved during evaluation period.
        With few eval episodes or difficult environments, many policies may
        tie at zero, producing uniform scores especially early in training.
      - ``"hybrid"``: Mixing the solve rate and returns: 
        in the score, solve rate dominates, with a small return-based
        term to break ties.  Useful when multiple policies achieve the same
        solve rate (usally tieing at zero) but show different progress in terms of return.
    """

    def __init__(
        self,
        config,
        train_env_reset_states,
        rng,
        mode: str = "return_rank",
        policy_eval_ep_count: int = 3,
        hybrid_return_weight: float = 0.1,
    ) -> None:
        """Args:
            config: Configuration dictionary.
            train_env_reset_states: Training level states, shape ``(E, ...)``.
            rng: JAX random key.
            mode: Scoring strategy -> ``"return_rank"``, ``"solve_rate"``, or
                ``"hybrid"``.  See class docstring for details.
            policy_eval_ep_count: Number of episodes per (policy, env) pair
                for the affinity evaluation rollout.  Each episode is
                is at maximum 256 steps by kinetix default (the Kinetix truncation horizon).
            hybrid_return_weight: Weight of the return-based tiebreaker in
                ``"hybrid"`` mode.  Ignored in other modes.  The return term
                is bounded to ``[0, hybrid_return_weight]`` per column, so
                solve rate dominates at typical values, making the return term a small tiebreaker.
        """
        super().__init__(config, train_env_reset_states, rng)
        if mode not in ("return_rank", "solve_rate", "hybrid"):
            raise ValueError(
                f"Unknown mode '{mode}'; expected 'return_rank', 'solve_rate', or 'hybrid'"
            )
        self.mode = mode
        self.hybrid_return_weight = hybrid_return_weight

        # Each episode is capped at 256 steps (Kinetix truncation horizon).
        KINETIX_TRUNCATION_STEPS = 256
        self.num_eval_steps = KINETIX_TRUNCATION_STEPS * policy_eval_ep_count

    def compute_affinity(self, env_reset_states) -> jax.Array:
        P = self.num_policies
        E_subset = get_pytree_batch_size(env_reset_states)

        # Evaluate every policy on every candidate env.
        # all_pairs: (P, E_subset)
        all_pairs = jnp.arange(E_subset)[None, :].repeat(P, axis=0)
        results = self.evaluate_pairing(
            all_pairs, env_reset_states, rollout_steps=self.num_eval_steps,
            params=self.policy_repo.train_state.params,
            hstate=self.policy_repo.hstate,
        )  # (P, E_subset, 2)

        solve_rate = results[:, :, 0]   # (P, E_subset) (within [0, 1])
        avg_return = results[:, :, 1]   # (P, E_subset) usually bounded between -3,3

        if self.mode == "return_rank":
            # Best policy gets rank 0, worst gets rank P-1.
            # Normalise to [0, 1]: (P-1 - rank) / (P-1)
            #   e.g. if for P=4 we have ranks for a certain env as [0,1,2,3] → scores would be [1.0, 0.67, 0.33, 0.0]
            ranks = jnp.argsort(jnp.argsort(-avg_return, axis=0), axis=0).astype(jnp.float32)
            raw = (P - 1 - ranks) / (P - 1)

        elif self.mode == "solve_rate":
            raw = solve_rate  # already in [0, 1]

        elif self.mode == "hybrid":
            # Solve rate dominates; softmax of returns breaks ties.
            return_softmax = jax.nn.softmax(avg_return, axis=0)  # (P, E_subset)
            raw = solve_rate + self.hybrid_return_weight * return_softmax

        return raw
