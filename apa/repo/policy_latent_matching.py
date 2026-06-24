"""
Repository using Kinetix transformer policy's backbone activations for policy-env matching.
"""

from apa.repo.embedding_matching import EmbeddingMatchingRepo
import jax.numpy as jnp
import jax


class PolicyLatentMatchingRepo(EmbeddingMatchingRepo):
    """
    Matches policies using a pre-trained transformer policy activations.
    """

    def __init__(
        self,
        config,
        train_env_reset_states,
        rng,
        checkpoint_path: str,
        history_decay: float = 0.2,
        sigma: float = 0.3,
    ) -> None:
        """Args:
            config: Configuration dictionary.
            train_env_reset_states: Training level states, shape ``(E, ...)``.
            rng: JAX random key.
            checkpoint_path: Path to a trained Kinetix ``full_model.pbz2`` checkpoint.
            history_decay: Exponential decay rate for past-step weighting.
                ``0`` = all history equally weighted forever;
                ``1`` = very fast forgetting.
            sigma: RBF kernel bandwidth on L2 distance between L2-normalized
                embeddings (range [0, 2] for unit vectors).
        """
        from apa.utils.transformer_policy_embedding import make_embedding_fn

        self._encode_fn = make_embedding_fn(config, checkpoint_path)
        self._encode_fn = jax.jit(self._encode_fn)

        super().__init__(config, train_env_reset_states, rng,
                         history_decay=history_decay, sigma=sigma)

    def _encode_envs(self, env_reset_states) -> jnp.ndarray:
        """Encode entity observations via frozen Kinetix transformer backbone.

        Returns ``(N, D)`` L2-normalized embeddings.
        """
        obs = jax.vmap(self.env.get_obs)(env_reset_states.env_state)
        return self._encode_fn(obs)
