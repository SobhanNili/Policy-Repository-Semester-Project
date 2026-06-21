"""Base class for embedding-based policy <-> environment matching.

Provides the shared affinity scoring calcluation logic used by reposiotries relying on env-embedding similarity to recent history of envs solves by the policy.
``DinoMatchingRepo``, ``MorphologyMatchingRepo``, and ``PolicyLatentMatchingRepo``
"""

from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

from apa.repo import PolicyRepo, PolicyState
from apa.repo.base import get_pytree_batch_size

class EmbeddingMatchingRepo(PolicyRepo, ABC):
    """Shared skeleton for embedding-based affinity scoring. Matches environments to policies that have historically performed well on similar environments.
    Similarity of envrionments is measured in an embedding space via an RBF kernel, and the history of past envs is soft-weighted by recency and solve rate.

    Subclasses must implement the mapping between env intial states and embedding vectors via ``_encode_envs``.:
        ``_encode_envs(env_reset_states) → (N, D)`` where N is the number of environments and D is the embedding dimension.
    """

    def __init__(
        self,
        config,
        train_env_reset_states,
        rng,
        history_decay: float = 0.1,
        sigma: float = 0.3,
    ) -> None:
        super().__init__(config, train_env_reset_states, rng)

        self.history_decay = history_decay
        self.sigma = sigma

        self._env_embeddings = self._encode_envs(train_env_reset_states)  # (E, D)

        # Flat history arrays, storing the candidate env indices, solve rates, and ages of all past step-log entries, per policy.
        self._hist_envs: Optional[jnp.ndarray] = None  # (P, H)
        self._hist_perf: Optional[jnp.ndarray] = None  # (P, H)
        self._hist_ages: Optional[jnp.ndarray] = None  # (P, H)

    @abstractmethod
    def _encode_envs(self, env_reset_states) -> jnp.ndarray:
        """Produce an embedding vector for each environment.

        Args:
            env_reset_states: pytree of environment reset states with leading
                dimension N.

        Returns:
            ``(N, D)`` float array of embeddings ready for L2-distance
            comparison.
        """
        ...

    def compute_affinity(self, env_reset_states) -> jax.Array:
        """RBF-kernel affinity of an env embedding with previous envs with weighting based on solving history.
        Env embeddings are computed via ``_encode_envs``.
        """
        P = self.num_policies
        E_subset = get_pytree_batch_size(env_reset_states)

        # Encode environments: (E_subset, D)
        pool_embeddings = self._encode_envs(env_reset_states)

        # Before any history has accumulated, all scores are uniform.
        if self._hist_envs is None or self._hist_envs.shape[1] == 0:
            return jnp.ones((P, E_subset)) / P

        def _score_single_policy(past_envs, past_perfs, past_ages):
            """past_envs: (H,), past_perfs: (H,), past_ages: (H,)

            Logic: 
            - Compute RBF kernel between candidate envs and past envs
            - Weight each past env by its solve_rate and recency decay (to favor "recent" "solves")
            - Return the weighted average of the RBF kernel values for each candidate env

            In essence, this is a soft nearest-neighbor lookup in env embedding space of policy encounters, where the "neighbors" are weighted by how well the policy performed on them recently.
            """
            # Soft weight: solve_rate x recency decay
            weights = past_perfs * jnp.exp(-self.history_decay * past_ages)
            # Look up cached embeddings for past environments: (H, D)
            past_embs = self._env_embeddings[past_envs]

            def _score_candidate(candidate_emb):
                """candidate_emb: (D,) → scalar"""
                sq_diffs = jnp.square(past_embs - candidate_emb)  # (H, D)
                dist2 = jnp.sum(sq_diffs, axis=-1)                  # (H,)
                rbf = jnp.exp(-dist2 / (2.0 * self.sigma ** 2))     # (H,)
                num = jnp.sum(weights * rbf)
                den = jnp.sum(weights) + 1e-12
                return num / den

            return jax.vmap(_score_candidate)(pool_embeddings)  # (E_subset,)

        return jax.vmap(_score_single_policy)(
            self._hist_envs, self._hist_perf, self._hist_ages
        )  # (P, E_subset)

    # History maintenance
    def update(
        self,
        policy_state: PolicyState,
        candidate_env_indices: jax.Array,
        affinity: jax.Array,
        active_mask: jax.Array,
        perf_metrics: jax.Array,
    ) -> None:
        """Called after each repo step — rebuild flat history arrays."""
        self.policy_repo = policy_state
        self._rebuild_history()

    def _rebuild_history(self):
        """Flatten ``step_logs`` into ``(P, total_entries)`` arrays.

        Each step-log entry is broadcast across policies.  Ages increase
        linearly with the number of steps since the entry was recorded.
        """
        if not self.step_logs:
            self._hist_envs = None
            self._hist_perf = None
            self._hist_ages = None
            return

        T = len(self.step_logs)
        envs_list, perf_list, ages_list = [], [], []

        for t, log in enumerate(self.step_logs):
            age = float(T - 1 - t)
            E = log.candidate_env_indices.shape[0]
            # Broadcast env indices across policies: (P, E)
            envs_list.append(
                jnp.tile(log.candidate_env_indices[None, :], (self.num_policies, 1))
            )
            # Replace NaN solve_rate with 0.0 (in case of not finishing the env in time limit)
            perf_list.append(jnp.nan_to_num(log.solve_rate, nan=0.0))
            ages_list.append(
                jnp.full((self.num_policies, E), age, dtype=jnp.float32)
            )

        self._hist_envs = jnp.concatenate(envs_list, axis=1)  # (P, total_entries)
        self._hist_perf = jnp.concatenate(perf_list, axis=1)
        self._hist_ages = jnp.concatenate(ages_list, axis=1)
