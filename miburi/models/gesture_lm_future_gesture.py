"""Leak-free future-gesture teacher variants for MIBURI.

The released temporal stream remains causal and consumes the shifted past
gesture sequence. A separate reverse-causal encoder summarizes the target
gesture suffix. Its output is shifted by one frame and fused only after the
causal temporal stack, so prediction at time ``t`` can use only gestures
``g[t+1:]`` and can never recover its own target ``g[t]`` directly or through
an earlier temporal position.

Two teachers share this exact future-gesture path:

* ``GTemporalDepthModel3FutureGesture`` keeps paired audio/text causal.
* ``GTemporalDepthModel3FutureGestureFullCondition`` exposes complete paired
  audio/text sequences to temporal cross-attention.

The kinematic/depth transformer is unchanged and remains locally conditioned.
These are privileged offline teachers, not standalone generators.
"""

import logging
import typing as tp

import torch
from torch import nn

from ..modules.transformer import StreamingTransformer
from .gesture_lm import GTemporalDepthModel3
from .gesture_lm_offline import _as_static_memory_attention
from .lm_utils import _init_layer


logger = logging.getLogger(__name__)


class GTemporalDepthModel3FutureGesture(GTemporalDepthModel3):
    """Future gesture with causal paired audio/text conditioning."""

    def __init__(
        self,
        *args,
        future_gesture_layers: int = 4,
        future_gesture_heads: int = 2,
        future_gesture_context: int | None = None,
        future_gesture_gate_init: float = 0.05,
        **kwargs,
    ):
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        super().__init__(*args, **kwargs)
        if not self.causal:
            raise ValueError(
                "The deployable past-gesture stream must remain causal."
            )
        if future_gesture_layers <= 0:
            raise ValueError("future_gesture_layers must be positive.")
        if future_gesture_heads <= 0:
            raise ValueError("future_gesture_heads must be positive.")
        if self.dim % future_gesture_heads:
            raise ValueError(
                f"Model dim {self.dim} is not divisible by "
                f"{future_gesture_heads} future-gesture heads."
            )
        if (
            future_gesture_context is not None
            and future_gesture_context <= 0
        ):
            future_gesture_context = None

        self.future_gesture_layers = future_gesture_layers
        self.future_gesture_context = future_gesture_context
        self.future_gesture_transformer = StreamingTransformer(
            d_model=self.dim,
            num_heads=future_gesture_heads,
            num_layers=future_gesture_layers,
            dim_feedforward=4 * self.dim,
            causal=True,
            context=future_gesture_context,
            positional_embedding="rope",
            gating="silu",
            norm="layer_norm",
            dropout=0.01,
            device=device,
            dtype=dtype,
        )
        self.future_gesture_fusion = nn.Linear(
            self.dim,
            self.dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.future_gesture_gate = nn.Parameter(
            torch.full(
                (self.dim,),
                float(future_gesture_gate_init),
                device=device,
                dtype=dtype,
            )
        )

        self.future_gesture_transformer.apply(_init_layer)
        _init_layer(self.future_gesture_fusion)
        self.to(device=device, dtype=dtype)
        self.temporal_condition_mode = "causal_audio_text"

    def _embed_gesture_frames(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() != 3:
            raise ValueError(
                f"Expected gesture codes [B,K,T], got {tuple(codes.shape)}."
            )
        if codes.shape[1] != self.n_q:
            raise ValueError(
                f"Expected {self.n_q} gesture codebooks, got "
                f"{codes.shape[1]}."
            )

        frame_embeddings = None
        for codebook_index in range(self.n_q):
            embedding = self.temporal_gemb[codebook_index](
                codes[:, codebook_index]
            )
            embedding = self.temporal_gproj[codebook_index](embedding)
            frame_embeddings = (
                embedding
                if frame_embeddings is None
                else frame_embeddings + embedding
            )
        if frame_embeddings is None:
            raise RuntimeError("No gesture frame embeddings were produced.")
        return frame_embeddings

    def encode_strict_future_gesture(
        self,
        target_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Return suffix context where position ``t`` uses only ``g[t+1:]``."""

        frame_embeddings = self._embed_gesture_frames(target_codes)
        reversed_embeddings = torch.flip(frame_embeddings, dims=[1])
        reversed_hidden = self.future_gesture_transformer(
            reversed_embeddings
        )
        suffix_hidden = torch.flip(reversed_hidden, dims=[1])

        # suffix_hidden[:, j] contains g[j:] because the encoder was causal
        # in reversed time. Shift it left so query t receives the state at
        # j=t+1. The final query has no future gesture and receives zero.
        no_future = torch.zeros_like(suffix_hidden[:, :1])
        return torch.cat(
            [suffix_hidden[:, 1:], no_future],
            dim=1,
        )

    def augment_temporal_output(
        self,
        temporal_output: torch.Tensor,
        temporal_target_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temporal_target_codes is None:
            raise RuntimeError(
                "Future-gesture teachers require complete ground-truth "
                "gesture tokens. They cannot run standalone autoregressive "
                "generation."
            )
        future_context = self.encode_strict_future_gesture(
            temporal_target_codes
        )
        if future_context.shape != temporal_output.shape:
            raise ValueError(
                "Future context and temporal output must align, got "
                f"{tuple(future_context.shape)} and "
                f"{tuple(temporal_output.shape)}."
            )
        future_update = self.future_gesture_fusion(
            future_context.to(temporal_output)
        )
        gate = torch.tanh(self.future_gesture_gate).to(temporal_output)
        return temporal_output + gate * future_update

    def _initialize_future_encoder_from_temporal(self):
        """Warm-start reverse layers from compatible released layer weights."""

        source_layers = self.temporal_transformer.layers
        if not source_layers:
            return
        with torch.no_grad():
            for index, future_layer in enumerate(
                self.future_gesture_transformer.layers
            ):
                source_state = source_layers[
                    index % len(source_layers)
                ].state_dict()
                target_state = future_layer.state_dict()
                compatible = {
                    key: value
                    for key, value in source_state.items()
                    if key in target_state
                    and target_state[key].shape == value.shape
                }
                future_layer.load_state_dict(compatible, strict=False)

    def load_state_dict(
        self,
        state_dict: tp.Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Accept released/offline checkpoints as a deliberate warm start."""

        has_future_weights = any(
            key.startswith("future_gesture_") for key in state_dict
        )
        if has_future_weights:
            return super().load_state_dict(
                state_dict,
                strict=strict,
                assign=assign,
            )

        incompatible = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        allowed_prefixes = (
            "future_gesture_transformer.",
            "future_gesture_fusion.",
            "future_gesture_gate",
        )
        unexpected_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if incompatible.unexpected_keys or unexpected_missing:
            raise RuntimeError(
                "Base checkpoint is not compatible with the future-gesture "
                f"teacher. Missing={unexpected_missing}, "
                f"unexpected={incompatible.unexpected_keys}."
            )
        self._initialize_future_encoder_from_temporal()
        logger.info(
            "Warm-started future-gesture teacher from a base MIBURI "
            "checkpoint; reverse-causal layers copied compatible temporal "
            "self-attention/FFN weights."
        )
        return incompatible


class GTemporalDepthModel3FutureGestureFullCondition(
    GTemporalDepthModel3FutureGesture
):
    """Future gesture plus complete paired audio/text temporal context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.temporal_transformer.layers:
            layer.cross_attns = nn.ModuleList(
                [
                    _as_static_memory_attention(attention)
                    for attention in layer.cross_attns
                ]
            )
        self.temporal_condition_mode = "full_audio_text"
