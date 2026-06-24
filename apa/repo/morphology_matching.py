"""
Repository that assigns policies to environments based on component-count similarity.
"""

from apa.repo.embedding_matching import EmbeddingMatchingRepo
from apa.utils.env import count_active_components
import jax.numpy as jnp
import jax

class MorphologyMatchingRepo(EmbeddingMatchingRepo):
    """Matches policies to structurally similar environments.

    Inherits the RBF-kernel affinity + soft-weighted history from
    ``EmbeddingMatchingRepo``.  Only implements ``_encode_envs`` — extracting
    6 normalised, weighted component-count features per environment.
    """

    def __init__(
        self,
        config,
        train_env_reset_states,
        rng,
        history_decay: float = 0.1,
        sigma: float = 0.15,
        feature_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ) -> None:
        """Args:
            config: Configuration dictionary.  Must contain ``num_polygons``,
                ``num_circles``, ``num_joints``, ``num_thrusters`` from the
                ``env_size`` section.
            train_env_reset_states: Training level states, shape ``(E, ...)``.
            rng: JAX random key.
            history_decay: Exponential decay rate for past-step weighting.
                ``0`` = all history equally weighted forever (no forgetting / putting more emphasis on recent history)
                ``1`` = very fast forgetting.
            sigma: RBF kernel bandwidth on L2 distance between embeddings.
            feature_weights: Per-feature multipliers applied before distance
                computation.  Length-6 tuple for ``[polygons, circles, joints,
                thrusters, green_shapes, blue_shapes]``.
        """
        # Indices into the 9-dim count_active_components output:
        #   0: n_polygons   1: n_circles   2: n_joints   3: n_thrusters
        #   6: n_green_shapes   7: n_blue_shapes
        # FIXME: I should`ve` probably include more fetures such as red objects count.
        self._feature_indices = jnp.array([0, 1, 2, 3, 6, 7], dtype=jnp.int32)

        # Max capacities for each type of entity (determined from the env_size config)
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

        super().__init__(config, train_env_reset_states, rng,
                         history_decay=history_decay, sigma=sigma)

    def _encode_envs(self, env_reset_states) -> jnp.ndarray:
        """Extract normalised, weighted component-count vectors.

        Returns ``(N, 6)`` float array with features in [0, 1] after
        normalisation by max capacities per that row, multiplied by per-feature weights.
        """
        raw = jax.vmap(count_active_components)(
            env_reset_states.env_state
        )  # (N, 9)
        selected = raw[:, self._feature_indices].astype(jnp.float32) # (N, 6)
        normalised = selected / self._max_vals                       # (N, 6) in [0, 1]
        return normalised * self._feature_weights                    # (N, 6) weighted
