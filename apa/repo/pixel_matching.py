from apa.repo import PolicyRepo, PolicyState
import jax.random as jr
import jax.numpy as jnp
import jax
from typing import Optional
import numpy as np


class PixelMatchingRepo(PolicyRepo):
    def __init__(self, config, unique_env_reset_states, rng,
                 history_window: int, policy_success_threshold: float = 0.5) -> None:
        """Repository that matches policies to visually similar environments.

        Args:
            config: Configuration dictionary.
            unique_env_reset_states: Unique initial level states, shape ``(E, ...)``.
            rng: JAX random key.
            history_window: Number of past repo steps to consider for success history.
            policy_success_threshold: Minimum solve rate for an env to be a 'success'.
        """
        super().__init__(config, unique_env_reset_states, rng)
        self.initial_pixel_obs = jax.vmap(
            lambda s: self.pixel_renderer(s.env_state)
        )(self.unique_env_reset_states).image  # (E, 125, 125, 3)
        self.history_window = history_window
        self.policy_success_threshold = policy_success_threshold

        # Flat arrays built from step_logs for efficient jitted affinity computation.
        # Shape: (P, total_entries), concatenated across repo steps.
        self._hist_envs: Optional[jnp.ndarray] = None   # (P, H) — env indices
        self._hist_perf: Optional[jnp.ndarray] = None   # (P, H) — solve rates

    # ── Affinity computation ───────────────────────────────────────────────

    def compute_affinity(self, candidate_env_indices: jax.Array) -> jax.Array:
        """Pixel-MSE-based affinity with column-wise softmax.

        For first step (no history): uniform affinity.
        Otherwise: each policy's raw score for an env is 1 / (min MSE to past
        successful envs + ε).  Scores are then softmax-normalised per env column.
        """
        P = self.num_policies
        E_subset = candidate_env_indices.shape[0]
        pool_pixel_obs = self.initial_pixel_obs[candidate_env_indices]  # (E_subset, H, W, C)

        if self._hist_envs is None or self._hist_envs.shape[1] == 0:
            # No history yet — uniform
            return jnp.ones((P, E_subset)) / P

        # Slice last history_window steps worth of columns
        window_cols = self.history_window * self.candidate_pool_size
        if self._hist_envs.shape[1] >= window_cols:
            hist_envs = self._hist_envs[:, -window_cols:]
            hist_perf = self._hist_perf[:, -window_cols:]
        else:
            hist_envs = self._hist_envs
            hist_perf = self._hist_perf

        def score_single_policy(past_envs, past_perfs):
            """past_envs: (H,), past_perfs: (H,)"""
            success_mask = past_perfs > self.policy_success_threshold  # (H,)
            past_obs = self.initial_pixel_obs[past_envs]  # (H, Hpx, Wpx, C)

            def score_candidate_env(candidate_obs):
                mse_dist = jnp.mean(
                    jnp.square(past_obs.astype(jnp.float32) - candidate_obs.astype(jnp.float32)),
                    axis=(-3, -2, -1)
                )  # (H,)
                valid_dists = jnp.where(success_mask, mse_dist, jnp.inf)
                min_dist = jnp.min(valid_dists)
                score = 1.0 / (min_dist + 1e-8)
                return jnp.where(jnp.any(success_mask), score, 0.0)

            return jax.vmap(score_candidate_env)(pool_pixel_obs)

        raw_scores = jax.vmap(score_single_policy)(hist_envs, hist_perf)  # (P, E_subset)

        # Column-wise softmax
        affinity = jax.nn.softmax(raw_scores, axis=0)  # softmax over policies (axis 0)
        return affinity

    # ── History maintenance ─────────────────────────────────────────────────

    def update(self, policy_state: PolicyState,
               candidate_env_indices: jax.Array,
               affinity: jax.Array,
               active_mask: jax.Array,
               perf_metrics: jax.Array) -> None:
        """Rebuild flat history arrays from step_logs for efficient scoring."""
        self.policy_repo = policy_state
        self._rebuild_history()

    def _rebuild_history(self):
        """Rebuild (P, total_entries) flat arrays from self.step_logs."""
        if not self.step_logs:
            self._hist_envs = None
            self._hist_perf = None
            return

        envs_list = []
        perf_list = []
        for log in self.step_logs:
            # log.candidate_env_indices: (E_subset,)
            # log.solve_rate: (P, E_subset) with NaN where inactive
            E_subset = log.candidate_env_indices.shape[0]
            # Broadcast env indices across policies: (P, E_subset)
            envs_broadcast = jnp.tile(log.candidate_env_indices[None, :], (self.num_policies, 1))
            # Replace NaN solve_rate with 0.0 (they'll be below threshold anyway)
            perf_clean = jnp.nan_to_num(log.solve_rate, nan=0.0)
            envs_list.append(envs_broadcast)
            perf_list.append(perf_clean)

        self._hist_envs = jnp.concatenate(envs_list, axis=1)  # (P, total_entries)
        self._hist_perf = jnp.concatenate(perf_list, axis=1)  # (P, total_entries)