"""Repository that assigns policies to environments based on frozen-policy performance.

Runs evaluation rollouts on all P × E_subset pairs at each repo step, then
derives affinity scores from the results.  Three scoring modes are available,
selected via the ``mode`` constructor argument.
"""

from apa.repo import PolicyRepo
from apa.repo.base import get_pytree_batch_size
import jax.numpy as jnp
import jax


class PerformanceMatchingRepo(PolicyRepo):
    """Matches policies to environments based on frozen-policy evaluation rollouts.

    Unlike proxy-based repos (pixel similarity, morphology), this measures
    actual policy competence on each environment.  The affinity matrix answers
    "which policy is best for this environment?" — the opposite question from
    ``EvalRepo``, which asks "which environment is in the sweet spot for this
    policy?"

    Three scoring modes:
      - ``"return_rank"``: Rank policies per env by average return (1 = best).
        Nonparametric and scale-invariant — handles negative returns and
        dense-reward noise gracefully.  Always produces P distinct affinity
        values per environment.
      - ``"solve_rate"``: Use the fraction of episodes solved directly.
        Coarse but interpretable — with few eval episodes, many policies may
        tie at zero, producing uniform affinity early in training.
      - ``"hybrid"``: Solve rate dominates, with a small return-based softmax
        term to break ties.  Useful when multiple policies achieve the same
        solve rate and you still want the better-returning one preferred.
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
            mode: Scoring strategy — ``"return_rank"``, ``"solve_rate"``, or
                ``"hybrid"``.  See class docstring for details.
            policy_eval_ep_count: Number of episodes per (policy, env) pair
                for the affinity evaluation rollout.  Each episode is
                pessimistically 256 steps (the Kinetix truncation horizon).
            hybrid_return_weight: Weight of the return-based tiebreaker in
                ``"hybrid"`` mode.  Ignored in other modes.  The return term
                is bounded to ``[0, hybrid_return_weight]`` per column, so
                solve rate dominates at typical values.
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

    # ── Affinity computation ───────────────────────────────────────────────

    def compute_affinity(self, env_reset_states) -> jax.Array:
        """Run eval rollouts on all (policy × candidate env) pairs and return
        raw affinity scores.

        Args:
            env_reset_states: pytree of environment reset states to score.

        Returns:
            ``(P, E_subset)`` float array of raw scores (higher = better match).
            Values are normalised to [0, 1] where possible.
        """
        P = self.num_policies
        E_subset = get_pytree_batch_size(env_reset_states)

        # Evaluate every policy on every candidate env.
        # all_pairs: (P, E_subset) — row p is [0, 1, …, E_subset-1].
        all_pairs = jnp.arange(E_subset)[None, :].repeat(P, axis=0)
        results = self.evaluate_pairing(
            all_pairs, env_reset_states, rollout_steps=self.num_eval_steps,
            params=self.policy_repo.train_state.params,
            hstate=self.policy_repo.hstate,
        )  # (P, E_subset, 2)

        solve_rate = results[:, :, 0]   # (P, E_subset) ∈ [0, 1]
        avg_return = results[:, :, 1]   # (P, E_subset) — may be negative

        if self.mode == "return_rank":
            # Best policy gets rank 0, worst gets rank P-1.
            # Normalise to [0, 1]: (P-1 - rank) / (P-1)
            #   P=4 → ranks [0,1,2,3] → scores [1.0, 0.67, 0.33, 0.0]
            ranks = jnp.argsort(jnp.argsort(-avg_return, axis=0), axis=0).astype(jnp.float32)
            raw = (P - 1 - ranks) / (P - 1)

        elif self.mode == "solve_rate":
            raw = solve_rate  # already in [0, 1]

        elif self.mode == "hybrid":
            # Solve rate dominates; softmax of returns breaks ties.
            return_softmax = jax.nn.softmax(avg_return, axis=0)  # (P, E_subset)
            raw = solve_rate + self.hybrid_return_weight * return_softmax

        return raw
