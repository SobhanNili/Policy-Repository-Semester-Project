"""Repository that assigns policies to environments based on DINO visual embedding similarity.

Uses a frozen DINOv2 model to encode rendered environment images into semantic
embedding vectors, then matches policies to environments based on similarity
to previously-solved environments — following the same RBF-kernel + soft-weighted
history pattern as ``MorphologyMatchingRepo``.
"""

from apa.repo import PolicyRepo, PolicyState
from apa.repo.base import get_pytree_batch_size
import jax.numpy as jnp
import jax
import numpy as np
from typing import Optional


class DinoMatchingRepo(PolicyRepo):
    """Matches policies to visually-similar environments using DINO embeddings.

    Architecture mirrors ``MorphologyMatchingRepo`` exactly:
      - Precompute DINO embeddings for all training envs at init.
      - Encode candidate envs on-the-fly in ``compute_affinity``.
      - RBF kernel on L2 distance between L2-normalized embeddings.
      - Soft-weighted history: ``weight = solve_rate × exp(-decay × age)``.
      - Column-wise temperature-scaled softmax for final affinity.

    The DINO model (PyTorch) is loaded once and used for inference only.
    Embeddings are cached as JAX arrays for efficient scoring.
    """

    def __init__(
        self,
        config,
        train_env_reset_states,
        rng,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", # "facebook/dinov2-base"
        history_decay: float = 0.1,
        sigma: float = 0.3,
        temperature: float = 1.0,
    ) -> None:
        """Args:
            config: Configuration dictionary.
            train_env_reset_states: Training level states, shape ``(E, ...)``.
            rng: JAX random key.
            model_name: HuggingFace model ID for DINO.  Default
                ``"facebook/dinov2-base"`` (768-dim embeddings).
            history_decay: Exponential decay rate for past-step weighting.
                ``0`` = all history equally weighted forever;
                ``1`` = very fast forgetting.  Default ``0.1``.
            sigma: RBF kernel bandwidth for L2 distance on unit vectors.
                L2 distance ranges [0, 2] for normalized vectors.
                Default ``0.3`` gives good discrimination.
            temperature: Softmax temperature.  ``1.0`` = standard softmax;
                ``< 1`` sharper, ``> 1`` flatter.  Default ``1.0``.
        """
        super().__init__(config, train_env_reset_states, rng)

        # ── Load DINO model (PyTorch) ────────────────────────────────────
        # First download is ~330 MB — cached by HuggingFace thereafter.
        from transformers import AutoImageProcessor, AutoModel
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dino_processor = AutoImageProcessor.from_pretrained(model_name)
        self._dino_model = AutoModel.from_pretrained(model_name).to(self._device).eval()

        # ── Precompute DINO embeddings for all training envs ─────────────
        # Render pixels: (E, H, W, C), rotate 90° CCW, encode via DINO, L2-normalize.
        pixels = jax.vmap(lambda s: self.pixel_renderer(s.env_state))(
            self.train_env_reset_states
        ).image  # (E, H, W, C)
        self.dino_embeddings = self._encode_pixels(pixels)  # (E, D), L2-normalized

        # ── Hyper-parameters ──────────────────────────────────────────────
        self.history_decay = history_decay
        self.sigma = sigma
        self.temperature = temperature

        # ── Flat history arrays (P, total_entries) for jitted scoring ────
        self._hist_envs: Optional[jnp.ndarray] = None  # (P, H) — env indices
        self._hist_perf: Optional[jnp.ndarray] = None  # (P, H) — solve rates
        self._hist_ages: Optional[jnp.ndarray] = None  # (P, H) — steps since recorded

    # ── DINO encoding bridge (JAX → PyTorch → JAX) ────────────────────────

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

    def _encode_envs(self, env_reset_states) -> jnp.ndarray:
        """Render and DINO-encode a batch of environment reset states.

        Args:
            env_reset_states: Pytree of ``LogEnvState`` with leading dim N.

        Returns:
            ``(N, D)`` JAX array of L2-normalized DINO embeddings.
        """
        pixels = jax.vmap(lambda s: self.pixel_renderer(s.env_state))(
            env_reset_states
        ).image  # (N, H, W, C)
        return self._encode_pixels(pixels)

    # ── Affinity computation ───────────────────────────────────────────────

    def compute_affinity(self, env_reset_states) -> jax.Array:
        """RBF-kernel affinity on DINO embeddings with soft-weighted history.

        Candidate embeddings are computed on-the-fly from ``env_reset_states``.
        History lookups use cached training-level DINO embeddings.
        """
        P = self.num_policies
        E_subset = get_pytree_batch_size(env_reset_states)

        # DINO embeddings for the candidate envs
        pool_embeddings = self._encode_envs(env_reset_states)  # (E_subset, D)

        if self._hist_envs is None or self._hist_envs.shape[1] == 0:
            return jnp.ones((P, E_subset)) / P

        def score_single_policy(past_envs, past_perfs, past_ages):
            """past_envs: (H,), past_perfs: (H,), past_ages: (H,)"""
            # Soft weight: solve_rate × recency decay
            weights = past_perfs * jnp.exp(-self.history_decay * past_ages)  # (H,)
            # Look up cached DINO embeddings for past envs
            past_embs = self.dino_embeddings[past_envs]  # (H, D)

            def score_candidate(candidate_emb):
                """candidate_emb: (D,) → scalar"""
                sq_diffs = jnp.square(past_embs - candidate_emb)  # (H, D)
                dist2 = jnp.sum(sq_diffs, axis=-1)  # (H,)
                rbf = jnp.exp(-dist2 / (2.0 * self.sigma ** 2))  # (H,)
                num = jnp.sum(weights * rbf)
                den = jnp.sum(weights) + 1e-12
                return num / den

            return jax.vmap(score_candidate)(pool_embeddings)  # (E_subset,)

        raw_scores = jax.vmap(score_single_policy)(
            self._hist_envs, self._hist_perf, self._hist_ages
        )  # (P, E_subset)

        # Column-wise temperature-scaled softmax
        affinity = jax.nn.softmax(raw_scores / self.temperature, axis=0)
        return affinity

    # ── History maintenance (identical to MorphologyMatchingRepo) ─────────

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
