"""Extract pre-head embeddings from a Kinetix ActorCriticTransformer checkpoint.

This module loads a trained checkpoint, strips the actor/critic head parameters,
and exposes a pure JAX function that maps ``EntityObservation`` → L2-normalized
embedding vectors suitable for environment similarity matching (analogous to how
``DinoMatchingRepo`` uses DINO embeddings).
"""

from __future__ import annotations

from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from kinetix.models.transformer_model import Transformer


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mask_out_inactives(flat_active_mask, matrix_attention_mask):
    """Replicate of the local function inside ActorCriticTransformer.__call__."""
    return matrix_attention_mask & (flat_active_mask[:, None]) & (flat_active_mask[None, :])


# ── Param filtering ─────────────────────────────────────────────────────────

def _is_head_param(path: tuple[str, ...]) -> bool:
    """Return True if *path* belongs to the actor/critic head (GeneralActorCriticRNN).

    Flax params are stored as ``{'params': {'Dense_0': ..., 'GeneralActorCriticRNN_0': ...}}``,
    so flattened keys look like ``('GeneralActorCriticRNN_0', 'Dense_0', 'bias')``.
    """
    return (
        len(path) >= 2
        and path[0] == "GeneralActorCriticRNN_0"
        and (path[1].startswith("Dense_") or path[1] == "log_std")
    )


def _filter_backbone_params(full_params: dict) -> dict:
    """Strip ``GeneralActorCriticRNN_0`` entries from a Flatten-able param dict."""
    from flax.traverse_util import flatten_dict, unflatten_dict

    flat = flatten_dict(full_params)
    backbone_flat = {k: v for k, v in flat.items() if not _is_head_param(k)}
    return unflatten_dict(backbone_flat)


# ── Extractor module ────────────────────────────────────────────────────────

class TransformerEmbeddingExtractor(nn.Module):
    """Replicates the ``ActorCriticTransformer`` backbone exactly.

    The submodule structure (encoder ``Dense`` layers, ``Transformer`` block)
    is identical to ``ActorCriticTransformer`` so that checkpoint parameter
    names align automatically after stripping the head entries.

    Stops *before* ``GeneralActorCriticRNN`` — returns the pooled embedding.
    """

    transformer_size: int
    transformer_encoder_size: int
    transformer_depth: int
    num_heads: int
    activation: str
    aggregate_mode: str          # "dummy" | "mean" | "dummy_and_mean"
    full_attention_mask: bool

    @nn.compact
    def __call__(self, hidden, x):
        del hidden  # not used by the backbone; kept for API compatibility

        activation = nn.relu if self.activation == "relu" else nn.tanh

        og_obs, dones = x
        obs = og_obs  # EntityObservation  (T, B, N, L)

        # ── Entity encoders (identical to ActorCriticTransformer) ──────────
        def _single_encoder(features, entity_id, concat=True):
            num_to_remove = 1 if concat else 0
            emb = activation(
                nn.Dense(
                    self.transformer_encoder_size - num_to_remove,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(features)
            )
            if concat:
                id_1h = jnp.zeros((*emb.shape[:3], 1)).at[:, :, :, 0].set(entity_id)
                return jnp.concatenate([emb, id_1h], axis=-1)
            return emb

        circle_encodings = _single_encoder(obs.circles, 0)
        polygon_encodings = _single_encoder(obs.polygons, 1)
        joint_encodings = _single_encoder(obs.joints, -1, False)
        thruster_encodings = _single_encoder(obs.thrusters, -1, False)

        shape_encodings = jnp.concatenate([polygon_encodings, circle_encodings], axis=2)
        shape_mask = jnp.concatenate([obs.polygon_mask, obs.circle_mask], axis=2)

        joint_indexes = obs.joint_indexes
        thruster_indexes = obs.thruster_indexes

        # ── Optional dummy token ───────────────────────────────────────────
        if self.aggregate_mode in ("dummy", "dummy_and_mean"):
            T, B, _, K = circle_encodings.shape
            dummy = jnp.ones((T, B, 1, K))
            shape_encodings = jnp.concatenate([dummy, shape_encodings], axis=2)
            shape_mask = jnp.concatenate(
                [jnp.ones((T, B, 1), dtype=bool), shape_mask], axis=2
            )
            N = obs.attention_mask.shape[-1]
            overall_mask = (
                jnp.ones((T, B, obs.attention_mask.shape[2], N + 1, N + 1), dtype=bool)
                .at[:, :, :, 1:, 1:]
                .set(obs.attention_mask)
            )
            overall_mask = jax.vmap(jax.vmap(_mask_out_inactives))(shape_mask, overall_mask)
            joint_indexes = joint_indexes + 1
            thruster_indexes = thruster_indexes + 1
        else:
            overall_mask = obs.attention_mask

        if self.full_attention_mask:
            overall_mask = jnp.ones(overall_mask.shape, dtype=bool)
            overall_mask = jax.vmap(jax.vmap(_mask_out_inactives))(shape_mask, overall_mask)

        # ── Transformer ────────────────────────────────────────────────────
        embedding = Transformer(
            num_layers=self.transformer_depth,
            num_heads=self.num_heads,
            qkv_features=self.transformer_size,
            encoder_size=self.transformer_encoder_size,
            gating=True,
            gating_bias=0.0,
        )(
            shape_encodings,
            jnp.repeat(overall_mask, repeats=self.num_heads // overall_mask.shape[2], axis=2),
            joint_encodings,
            obs.joint_mask,
            joint_indexes,
            thruster_encodings,
            obs.thruster_mask,
            thruster_indexes,
        )

        # ── Pool ───────────────────────────────────────────────────────────
        if self.aggregate_mode in ("mean", "dummy_and_mean"):
            embedding = jnp.mean(embedding, axis=2, where=shape_mask[..., None])
        else:
            embedding = embedding[:, :, 0]

        return embedding  # (T, B, transformer_encoder_size)


# ── Public API ──────────────────────────────────────────────────────────────

def make_embedding_fn(
    config: dict[str, Any],
    checkpoint_path: str,
) -> Callable[[Any], jnp.ndarray]:
    """Build a function that encodes entity observations → L2-normalized embeddings.

    Args:
        config: Flat configuration dict (from ``load_config``).  Must contain
            ``transformer_size``, ``transformer_encoder_size``,
            ``transformer_depth``, ``num_heads``, ``activation``,
            ``aggregate_mode``, ``full_attention_mask``.
        checkpoint_path: Path to a ``full_model.pbz2`` checkpoint.

    Returns:
        A callable ``encode(obs) -> embeddings`` where embeddings are
        ``(B, D)`` float32, L2-normalized.  The returned function is pure
        and can be wrapped with ``jax.jit``.
    """
    from kinetix.util.saving import load_params

    payload = load_params(checkpoint_path)
    raw_params = payload["params"]

    # Unwrap the Flax 'params' wrapper if present, filter, then re-wrap.
    if isinstance(raw_params, dict) and "params" in raw_params:
        inner = raw_params["params"]
        backbone_params = {"params": _filter_backbone_params(inner)}
    else:
        backbone_params = _filter_backbone_params(raw_params)

    extractor = TransformerEmbeddingExtractor(
        transformer_size=int(config["transformer_size"]),
        transformer_encoder_size=int(config["transformer_encoder_size"]),
        transformer_depth=int(config["transformer_depth"]),
        num_heads=int(config["num_heads"]),
        activation=config["activation"],
        aggregate_mode=config.get("aggregate_mode", "dummy_and_mean"),
        full_attention_mask=bool(config.get("full_attention_mask", False)),
    )

    def encode(obs):
        """obs: EntityObservation with leading (B,) batch dim → (B, D) L2-normed."""
        obs_t = jax.tree.map(lambda x: jnp.expand_dims(x, 0), obs)
        B = obs.circles.shape[0]
        dones = jnp.zeros((1, B), dtype=bool)
        from kinetix.models import ScannedRNN
        hidden = ScannedRNN.initialize_carry(B)

        emb = extractor.apply(backbone_params, hidden, (obs_t, dones))
        emb = jnp.squeeze(emb, axis=0)
        emb = emb / (jnp.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12)
        return emb

    return encode


def load_and_validate(checkpoint_path: str, config: dict[str, Any]) -> dict:
    """Load checkpoint and print diagnostic info.  Returns raw params dict."""
    from kinetix.util.saving import load_params
    from flax.traverse_util import flatten_dict

    payload = load_params(checkpoint_path)
    raw_params = payload["params"]

    if isinstance(raw_params, dict) and "params" in raw_params:
        inner_params = raw_params["params"]
        nested = True
    else:
        inner_params = raw_params
        nested = False

    flat = flatten_dict(inner_params)
    head_keys = [k for k in flat if _is_head_param(k)]
    backbone_keys = [k for k in flat if not _is_head_param(k)]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"  Step: {payload.get('step', 'N/A')}")
    print(f"  Params nesting: {'wrapped' if nested else 'flat'}")
    print(f"  Total param leaves: {len(flat)}")
    print(f"  Backbone leaves:    {len(backbone_keys)}")
    print(f"  Head leaves:        {len(head_keys)}")
    print(f"  Embedding dim:      {int(config['transformer_encoder_size'])}")

    return raw_params
