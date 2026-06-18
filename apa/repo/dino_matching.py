"""Repository that assigns policies to environments based on DINO visual embedding similarity.

Uses a frozen DINOv3 model to encode rendered environment images into semantic
embedding vectors.  The RBF-kernel + soft-weighted history scoring is inherited
from ``EmbeddingMatchingRepo``.
"""

from apa.repo.embedding_matching import EmbeddingMatchingRepo
import jax.numpy as jnp
import jax
import numpy as np


class DinoMatchingRepo(EmbeddingMatchingRepo):
    """Matches policies to visually-similar environments using DINO embeddings.

    Inherits the RBF-kernel affinity + soft-weighted history from
    ``EmbeddingMatchingRepo``.  Only implements ``_encode_envs`` — rendering
    each environment, passing pixels through a frozen DINOv3 ViT, and
    L2-normalizing the CLS embedding.

    The DINO model (PyTorch) is loaded once at init and used for inference only.
    """

    def __init__(
        self,
        config,
        train_env_reset_states,
        rng,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        history_decay: float = 0.1,
        sigma: float = 0.3,
    ) -> None:
        """Args:
            config: Configuration dictionary.
            train_env_reset_states: Training level states, shape ``(E, ...)``.
            rng: JAX random key.
            model_name: HuggingFace model ID for DINO.
            history_decay: Exponential decay rate for past-step weighting.
                ``0`` = all history equally weighted forever;
                ``1`` = very fast forgetting.
            sigma: RBF kernel bandwidth on L2 distance between L2-normalized
                embeddings (range [0, 2] for unit vectors).
        """
        # ── Load DINO model (PyTorch) before super().__init__ calls _encode_envs
        # First download is ~330 MB — cached by HuggingFace thereafter.
        from transformers import AutoImageProcessor, AutoModel
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dino_processor = AutoImageProcessor.from_pretrained(model_name)
        self._dino_model = AutoModel.from_pretrained(model_name).to(self._device).eval()

        super().__init__(config, train_env_reset_states, rng,
                         history_decay=history_decay, sigma=sigma)

    # ── Embedding (implements EmbeddingMatchingRepo._encode_envs) ──────────

    def _encode_envs(self, env_reset_states) -> jnp.ndarray:
        """Render and DINO-encode a batch of environment reset states.

        Returns ``(N, D)`` L2-normalized CLS embeddings.
        """
        pixels = jax.vmap(lambda s: self.pixel_renderer(s.env_state))(
            env_reset_states
        ).image  # (N, H, W, C)
        return self._encode_pixels(pixels)

    def _encode_pixels(self, pixels) -> jnp.ndarray:
        """Encode pixel images into L2-normalized DINO CLS embeddings.

        Args:
            pixels: JAX array of shape ``(N, H, W, C)`` in [0, 1].

        Returns:
            ``(N, D)`` JAX array of L2-normalized DINO embeddings.
        """
        import torch

        # JAX → numpy, rotate 90° CCW (fixes rendering orientation)
        pixels_np = np.array(pixels)
        pixels_np = np.rot90(pixels_np, k=1, axes=(1, 2)).copy()

        # PyTorch inference
        inputs = self._dino_processor(images=pixels_np, return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            outputs = self._dino_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (N, D)

        # L2 normalize
        embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        )
        return jnp.asarray(embeddings, dtype=jnp.float32)
