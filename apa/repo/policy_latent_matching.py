"""Repository using Kinetix transformer backbone embeddings for policy-env matching.

Same RBF-kernel + soft-weighted history architecture as DinoMatchingRepo,
but uses a frozen Kinetix checkpoint instead of a DINO vision model.
"""

from apa.repo import PolicyRepo, PolicyState
from apa.repo.base import get_pytree_batch_size
import jax.numpy as jnp
import jax
import numpy as np
from typing import Optional


class PolicyLatentMatchingRepo(PolicyRepo):

    def __init__(
        self,
        config,
        train_env_reset_states,
        rng,
        checkpoint_path: str,
        history_decay: float = 0.2,
        sigma: float = 0.3,
    ) -> None:
        super().__init__(config, train_env_reset_states, rng)

        # ── Build the embedding function from checkpoint ──────────────────
        from apa.utils.embedding import make_embedding_fn

        self._encode = make_embedding_fn(config, checkpoint_path)
        self._encode = jax.jit(self._encode)

        # ── Precompute embeddings for all training envs ───────────────────
        all_obs = jax.vmap(self.env.get_obs)(self.train_env_reset_states.env_state)
        self._env_embeddings = self._encode(all_obs)  # (E, D), L2-normalized

        self.history_decay = history_decay
        self.sigma = sigma

        self._hist_envs: Optional[jnp.ndarray] = None  # (P, H)
        self._hist_perf: Optional[jnp.ndarray] = None  # (P, H)
        self._hist_ages: Optional[jnp.ndarray] = None  # (P, H)

    # ── Affinity (identical logic to DinoMatchingRepo) ────────────────────

    def compute_affinity(self, env_reset_states) -> jax.Array:
        P = self.num_policies
        E_subset = get_pytree_batch_size(env_reset_states)

        cand_obs = jax.vmap(self.env.get_obs)(env_reset_states.env_state)
        pool_embeddings = self._encode(cand_obs)  # (E_subset, D)

        if self._hist_envs is None or self._hist_envs.shape[1] == 0:
            return jnp.ones((P, E_subset)) / P

        def score_single_policy(past_envs, past_perfs, past_ages):
            weights = past_perfs * jnp.exp(-self.history_decay * past_ages)
            past_embs = self._env_embeddings[past_envs]

            def score_candidate(candidate_emb):
                sq_diffs = jnp.square(past_embs - candidate_emb)
                dist2 = jnp.sum(sq_diffs, axis=-1)
                rbf = jnp.exp(-dist2 / (2.0 * self.sigma ** 2))
                num = jnp.sum(weights * rbf)
                den = jnp.sum(weights) + 1e-12
                return num / den

            return jax.vmap(score_candidate)(pool_embeddings)

        return jax.vmap(score_single_policy)(
            self._hist_envs, self._hist_perf, self._hist_ages
        )  # (P, E_subset)

    # ── History ──────────────────────────────────────────────────────────

    def update(self, policy_state, candidate_env_indices, affinity,
               active_mask, perf_metrics):
        self.policy_repo = policy_state
        self._rebuild_history()

    def _rebuild_history(self):
        if not self.step_logs:
            self._hist_envs = self._hist_perf = self._hist_ages = None
            return

        T = len(self.step_logs)
        envs_list, perf_list, ages_list = [], [], []

        for t, log in enumerate(self.step_logs):
            age = float(T - 1 - t)
            E = log.candidate_env_indices.shape[0]
            envs_list.append(jnp.tile(log.candidate_env_indices[None, :],
                                      (self.num_policies, 1)))
            perf_list.append(jnp.nan_to_num(log.solve_rate, nan=0.0))
            ages_list.append(jnp.full((self.num_policies, E), age,
                                      dtype=jnp.float32))

        self._hist_envs = jnp.concatenate(envs_list, axis=1)
        self._hist_perf = jnp.concatenate(perf_list, axis=1)
        self._hist_ages = jnp.concatenate(ages_list, axis=1)
