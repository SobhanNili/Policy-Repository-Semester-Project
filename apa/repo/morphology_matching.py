"""Repository that assigns policies to environments based on component-count similarity.

Follows the same architecture as ``PixelMatchingRepo`` but uses morphological
feature vectors (6 selected components from ``count_active_components``)
instead of pixel observations.

Key design differences from pixel matching:
- **RBF kernel** for distance→score mapping: ``exp(-d² / 2σ²)``, bounded [0, 1].
  No singularity at d=0 (common when two envs share identical component counts).
- **Soft-weighted history**: ``weight = solve_rate × exp(-decay × age)`` replaces
  the binary success threshold + hard history window of pixel matching.
- **Temperature** in the column-wise softmax for controllable sharpness.
- **Feature normalization** by maximum component counts from the config's env_size.
"""

from apa.repo import PolicyRepo, PolicyState
from apa.repo.base import get_pytree_batch_size
from apa.utils.env import count_active_components
import jax.numpy as jnp
import jax
from typing import Optional


class MorphologyMatchingRepo(PolicyRepo):
    def __init__(
        self,
        config,
        train_env_reset_states,
        rng,
        history_decay: float = 0.1,
        sigma: float = 0.15,
        temperature: float = 1.0,
        feature_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ) -> None:
        """Repository that matches policies to structurally similar environments.

        Args:
            config: Configuration dictionary.  Must contain ``num_polygons``,
                ``num_circles``, ``num_joints``, ``num_thrusters`` from the
                ``env_size`` section.
            train_env_reset_states: Training level states, shape ``(E, ...)``.
            rng: JAX random key.
            history_decay: Exponential decay rate for past-step weighting.
                ``0`` = all history equally weighted forever;
                ``1`` = very fast forgetting.  Default ``0.2``.
            sigma: RBF kernel bandwidth in *normalised* L2 space.
                Smaller values require closer component-count match.
                Default ``0.15`` (suitable when envs differ by 1–2 components).
            temperature: Softmax temperature.  ``1.0`` = standard softmax;
                ``< 1`` sharper, ``> 1`` flatter.  Default ``1.0``.
            feature_weights: Per-feature multipliers applied before distance
                computation.  Length-6 tuple for ``[polygons, circles, joints,
                thrusters, green_shapes, blue_shapes]``.  Default all ones.
        """
        super().__init__(config, train_env_reset_states, rng)

        # ── Build normalised component vectors for all training envs ──────
        # Indices into the 9-dim count_active_components output:
        #   0: n_polygons   1: n_circles   2: n_joints   3: n_thrusters
        #   6: n_green_shapes   7: n_blue_shapes
        self._feature_indices = jnp.array([0, 1, 2, 3, 6, 7], dtype=jnp.int32)

        # Max capacities from the env_size config (merged into flat config by load_config).
        num_poly = float(config["num_polygons"])
        num_circ = float(config["num_circles"])
        num_joint = float(config["num_joints"])
        num_thrust = float(config["num_thrusters"])
        num_total = num_poly + num_circ  # max possible for green or blue shapes

        self._max_vals = jnp.array(
            [num_poly, num_circ, num_joint, num_thrust, num_total, num_total],
            dtype=jnp.float32,
        )

        self._feature_weights = jnp.array(feature_weights, dtype=jnp.float32)

        # Precompute: (E, 6) normalised component vectors
        raw_components = jax.vmap(count_active_components)(
            self.train_env_reset_states.env_state
        )  # (E, 9)
        selected = raw_components[:, self._feature_indices].astype(jnp.float32)  # (E, 6)
        self.component_vectors = selected / self._max_vals  # (E, 6) in [0, 1]

        # ── Hyper-parameters ──────────────────────────────────────────────
        self.history_decay = history_decay
        self.sigma = sigma
        self.temperature = temperature

        # ── Flat history arrays (P, total_entries) for jitted scoring ────
        self._hist_envs: Optional[jnp.ndarray] = None  # (P, H) — env indices
        self._hist_perf: Optional[jnp.ndarray] = None  # (P, H) — solve rates
        self._hist_ages: Optional[jnp.ndarray] = None  # (P, H) — steps since recorded

    # ── Affinity computation ───────────────────────────────────────────────

    def compute_affinity(self, env_reset_states) -> jax.Array:
        """RBF-kernel affinity with soft-weighted history and column-wise softmax.

        Candidate component vectors are computed on-the-fly from
        ``env_reset_states``.  History lookups always use cached training-level
        component vectors.
        """
        P = self.num_policies
        E_subset = get_pytree_batch_size(env_reset_states)

        # Normalised component vectors for the candidate envs
        raw_candidates = jax.vmap(count_active_components)(
            env_reset_states.env_state
        )  # (E_subset, 9)
        pool_comps = (
            raw_candidates[:, self._feature_indices].astype(jnp.float32)
            / self._max_vals
        )  # (E_subset, 6)
        pool_comps = pool_comps * self._feature_weights  # apply per-feature weights

        if self._hist_envs is None or self._hist_envs.shape[1] == 0:
            return jnp.ones((P, E_subset)) / P

        def score_single_policy(past_envs, past_perfs, past_ages):
            """past_envs: (H,), past_perfs: (H,), past_ages: (H,)"""
            # Soft weight: solve_rate × recency decay
            weights = past_perfs * jnp.exp(-self.history_decay * past_ages)  # (H,)
            # Look up cached normalised vectors and apply feature weights
            past_comps = self.component_vectors[past_envs] * self._feature_weights  # (H, 6)

            def score_candidate(candidate_comp):
                """candidate_comp: (6,) → scalar"""
                sq_diffs = jnp.square(past_comps - candidate_comp)  # (H, 6)
                dist2 = jnp.sum(sq_diffs, axis=-1)  # (H,)
                rbf = jnp.exp(-dist2 / (2.0 * self.sigma ** 2))  # (H,)
                num = jnp.sum(weights * rbf)
                den = jnp.sum(weights) + 1e-12
                return num / den

            return jax.vmap(score_candidate)(pool_comps)  # (E_subset,)

        raw_scores = jax.vmap(score_single_policy)(
            self._hist_envs, self._hist_perf, self._hist_ages
        )  # (P, E_subset)

        # Column-wise temperature-scaled softmax
        affinity = jax.nn.softmax(raw_scores / self.temperature, axis=0)
        return affinity

    # ── History maintenance ─────────────────────────────────────────────────

    def update(
        self,
        policy_state: PolicyState,
        candidate_env_indices: jax.Array,
        affinity: jax.Array,
        active_mask: jax.Array,
        perf_metrics: jax.Array,
    ) -> None:
        """Rebuild flat history arrays from step_logs for efficient scoring."""
        self.policy_repo = policy_state
        self._rebuild_history()

    def _rebuild_history(self):
        """Rebuild ``(P, total_entries)`` flat arrays for envs, perfs, and ages."""
        if not self.step_logs:
            self._hist_envs = None
            self._hist_perf = None
            self._hist_ages = None
            return

        T = len(self.step_logs)
        envs_list = []
        perf_list = []
        ages_list = []

        for t, log in enumerate(self.step_logs):
            age = float(T - 1 - t)  # steps ago
            E_subset = log.candidate_env_indices.shape[0]
            # Broadcast env indices across policies: (P, E_subset)
            envs_broadcast = jnp.tile(
                log.candidate_env_indices[None, :], (self.num_policies, 1)
            )
            # Replace NaN solve_rate with 0.0 (they will get near-zero weight)
            perf_clean = jnp.nan_to_num(log.solve_rate, nan=0.0)
            ages_full = jnp.full(
                (self.num_policies, E_subset), age, dtype=jnp.float32
            )

            envs_list.append(envs_broadcast)
            perf_list.append(perf_clean)
            ages_list.append(ages_full)

        self._hist_envs = jnp.concatenate(envs_list, axis=1)  # (P, total_entries)
        self._hist_perf = jnp.concatenate(perf_list, axis=1)
        self._hist_ages = jnp.concatenate(ages_list, axis=1)
