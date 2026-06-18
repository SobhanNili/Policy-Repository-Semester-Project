"""Base class for embedding-based policy–environment matching.

Provides the shared RBF-kernel + soft-weighted history architecture used by
``DinoMatchingRepo``, ``MorphologyMatchingRepo``, and ``PolicyLatentMatchingRepo``.

Subclasses only implement ``_encode_envs`` to produce an embedding vector per
environment.  Everything else — affinity scoring, history maintenance, and
the ``update`` hook — is inherited from this class.
"""

from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

from apa.repo import PolicyRepo, PolicyState
from apa.repo.base import get_pytree_batch_size


class EmbeddingMatchingRepo(PolicyRepo, ABC):
    """Shared skeleton for embedding-based affinity scoring.

    Caches precomputed embeddings for all training environments at init and
    encodes candidate environments on the fly during ``compute_affinity``.
    The RBF kernel on L2 distance between embeddings is soft-weighted by
    ``solve_rate × exp(−decay × age)`` so that successful, recent matches
    dominate the score for each policy.

    Subclasses must implement:
        ``_encode_envs(env_reset_states) → (N, D)``
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

        # Precompute embeddings for all training environments.
        # Subclass _encode_envs is responsible for any model loading
        # or feature extraction needed — set those up *before* calling
        # super().__init__() in the subclass.
        self._env_embeddings = self._encode_envs(train_env_reset_states)  # (E, D)

        # Flat history arrays — built lazily by _rebuild_history.
        self._hist_envs: Optional[jnp.ndarray] = None  # (P, H)
        self._hist_perf: Optional[jnp.ndarray] = None  # (P, H)
        self._hist_ages: Optional[jnp.ndarray] = None  # (P, H)

    # ── Subclass contract ──────────────────────────────────────────────────

    @abstractmethod
    def _encode_envs(self, env_reset_states) -> jnp.ndarray:
        """Produce an embedding vector for each environment.

        Args:
            env_reset_states: pytree of environment reset states with leading
                dimension N.

        Returns:
            ``(N, D)`` float array of embeddings ready for L2-distance
            comparison.  Embedding vectors should already be L2-normalized
            if the embedding space is isotropic (e.g. DINO, Kinetix); for
            feature-count vectors a bounded [0, 1] range is sufficient.
        """
        ...

    # ── Affinity (RBF kernel + soft-weighted history) ──────────────────────

    def compute_affinity(self, env_reset_states) -> jax.Array:
        """RBF-kernel affinity with soft-weighted history.

        Candidate embeddings are computed on the fly via ``_encode_envs``;
        past-environment embeddings are looked up from the precomputed cache.
        """
        P = self.num_policies
        E_subset = get_pytree_batch_size(env_reset_states)

        # Encode candidate environments: (E_subset, D)
        pool_embeddings = self._encode_envs(env_reset_states)

        # Before any history has accumulated, all scores are uniform.
        if self._hist_envs is None or self._hist_envs.shape[1] == 0:
            return jnp.ones((P, E_subset)) / P

        def _score_single_policy(past_envs, past_perfs, past_ages):
            """past_envs: (H,), past_perfs: (H,), past_ages: (H,)"""
            # Soft weight: solve_rate × recency decay
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

    # ── History maintenance ────────────────────────────────────────────────

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
            # Replace NaN solve_rate with 0.0 (near-zero weight)
            perf_list.append(jnp.nan_to_num(log.solve_rate, nan=0.0))
            ages_list.append(
                jnp.full((self.num_policies, E), age, dtype=jnp.float32)
            )

        self._hist_envs = jnp.concatenate(envs_list, axis=1)  # (P, total_entries)
        self._hist_perf = jnp.concatenate(perf_list, axis=1)
        self._hist_ages = jnp.concatenate(ages_list, axis=1)
