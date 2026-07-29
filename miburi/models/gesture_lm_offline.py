"""Offline-conditioned variant of the released MIBURI gesture LM.

The gesture factorization is unchanged:

* the temporal transformer consumes complete preceding gesture frames and
  predicts only the first upper-body RVQ token;
* the kinematic/depth transformer predicts the remaining 19 tokens inside
  the current frame.

Only temporal conditioning changes. Gesture self-attention stays causal,
while temporal audio/text cross-attention can inspect the complete condition
sequence. During autoregressive evaluation, the full condition is installed
as static memory and its projected keys/values are cached once per layer.
"""

from dataclasses import dataclass
import typing as tp

from einops import rearrange
import torch
import torch.nn.functional as F

from ..modules.streaming import State
from ..modules.transformer import (
    StreamingMultiheadCrossAttention,
    apply_weights_per_step,
)
from ..utils.compile import CUDAGraphed
from ..utils.sampling import sample_token
from .gesture_lm import GTemporalDepthModel3, GestureLMGen, _GLMGenState


@dataclass
class _StaticMemoryCrossAttentionState(State):
    """Streaming query position plus one immutable full-memory KV cache."""

    query_offset: torch.Tensor
    query_offset_cpu: int = 0
    projected_keys: torch.Tensor | None = None
    projected_values: torch.Tensor | None = None
    memory_length: int = 0

    def reset(self):
        self.query_offset.zero_()
        self.query_offset_cpu = 0
        self.projected_keys = None
        self.projected_values = None
        self.memory_length = 0


class StaticMemoryCrossAttention(StreamingMultiheadCrossAttention):
    """Noncausal cross-attention with static-memory streaming inference.

    In ordinary (training/validation) mode this is exactly the parent's
    noncausal full-sequence attention. In streaming mode, queries arrive one
    gesture frame at a time, but the audio/text memory is the same complete
    sequence on every call. Its projected K/V tensors are therefore computed
    once, while the RoPE query offset continues to advance autoregressively.
    """

    def _init_streaming_state(
        self, batch_size: int
    ) -> _StaticMemoryCrossAttentionState:
        parameter = next(self.parameters())
        return _StaticMemoryCrossAttentionState(
            batch_size=batch_size,
            device=parameter.device,
            query_offset=torch.zeros(
                batch_size,
                device=parameter.device,
                dtype=torch.long,
            ),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_padding_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ):
        state = self._streaming_state
        if state is None:
            return super().forward(
                query,
                key,
                value,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
            )

        if self.causal:
            raise RuntimeError(
                "StaticMemoryCrossAttention is reserved for noncausal "
                "offline conditioning."
            )
        if not isinstance(state, _StaticMemoryCrossAttentionState):
            raise TypeError(f"Unexpected streaming state: {type(state)!r}")

        batch_size, query_length = query.shape[:2]
        if batch_size != state.batch_size:
            raise ValueError(
                f"Expected batch size {state.batch_size}, got {batch_size}."
            )

        projected_query = apply_weights_per_step(
            self.in_projs,
            self.weights_per_step_schedule,
            query,
            state.query_offset_cpu,
        )
        q = rearrange(
            projected_query,
            "b t (h d) -> b h t d",
            h=self.num_heads,
        )

        if state.projected_keys is None:
            zero_key_offset = torch.zeros_like(state.query_offset)
            projected_key = apply_weights_per_step(
                self.key_projs,
                self.weights_per_step_schedule,
                key,
                0,
            )
            projected_value = apply_weights_per_step(
                self.val_projs,
                self.weights_per_step_schedule,
                value,
                0,
            )
            k = rearrange(
                projected_key,
                "b t (h d) -> b h t d",
                h=self.num_heads,
            )
            v = rearrange(
                projected_value,
                "b t (h d) -> b h t d",
                h=self.num_heads,
            )
            if self.rope is not None:
                q, k = self.rope(
                    q,
                    k,
                    state.query_offset,
                    offsetk=zero_key_offset,
                    time_before_heads=False,
                )
            state.projected_keys = k
            state.projected_values = v
            state.memory_length = key.shape[1]
        else:
            if key.shape[1] != state.memory_length:
                raise ValueError(
                    "Static conditioning length changed without resetting "
                    f"the generator: {state.memory_length} -> {key.shape[1]}."
                )
            if self.rope is not None:
                # The cached keys have already received their position-0 RoPE
                # rotation. Rotate only the new query by pairing it with an
                # empty key sequence.
                empty_key = state.projected_keys[:, :, :0]
                zero_key_offset = torch.zeros_like(state.query_offset)
                q, _ = self.rope(
                    q,
                    empty_key,
                    state.query_offset,
                    offsetk=zero_key_offset,
                    time_before_heads=False,
                )

        k = state.projected_keys
        v = state.projected_values
        if k is None or v is None:
            raise RuntimeError("Static K/V memory was not initialized.")
        memory_length = k.shape[2]

        attention_mask = None
        if query_padding_mask is not None:
            if query_padding_mask.shape != (batch_size, query_length):
                raise ValueError(
                    "Expected query padding mask shape "
                    f"{(batch_size, query_length)}, got "
                    f"{tuple(query_padding_mask.shape)}."
                )
            attention_mask = (
                ~query_padding_mask[:, None, :, None]
            ).expand(-1, -1, -1, memory_length)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, memory_length):
                raise ValueError(
                    "Expected key padding mask shape "
                    f"{(batch_size, memory_length)}, got "
                    f"{tuple(key_padding_mask.shape)}."
                )
            valid_keys = (
                ~key_padding_mask[:, None, None, :]
            ).expand(-1, -1, query_length, -1)
            attention_mask = (
                valid_keys
                if attention_mask is None
                else attention_mask & valid_keys
            )

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask,
            dropout_p=0.0,
        )
        output = rearrange(output, "b h t d -> b t (h d)")
        output = apply_weights_per_step(
            self.out_projs,
            self.weights_per_step_schedule,
            output,
            state.query_offset_cpu,
        )

        state.query_offset[:] = torch.where(
            state.exec_mask,
            state.query_offset + query_length,
            state.query_offset,
        )
        state.query_offset_cpu += query_length
        return output


def _as_static_memory_attention(
    source: StreamingMultiheadCrossAttention,
) -> StaticMemoryCrossAttention:
    """Create a state-dict-compatible static-memory attention module."""

    parameter = next(source.parameters())
    replacement = StaticMemoryCrossAttention(
        embed_dim=source.embed_dim,
        num_heads=source.num_heads,
        causal=False,
        context=source.context,
        rope=source.rope,
        weights_per_step=source.weights_per_step,
        weights_per_step_schedule=source.weights_per_step_schedule,
        upsample_factor=source.upsample_factor,
        query_upsample_factor=source.query_upsample_factor,
        key_width=source.key_width,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    replacement.load_state_dict(source.state_dict())
    return replacement


class GTemporalDepthModel3Offline(GTemporalDepthModel3):
    """Original MIBURI architecture with offline temporal conditioning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.causal:
            raise ValueError(
                "Gesture self-attention must remain causal in the offline "
                "conditioning variant."
            )

        for layer in self.temporal_transformer.layers:
            layer.cross_attns = torch.nn.ModuleList(
                [
                    _as_static_memory_attention(attention)
                    for attention in layer.cross_attns
                ]
            )
        self.temporal_crossattn_causal = False

    def project_temporal_conditions(
        self,
        audio_condition: torch.Tensor,
        text_condition: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference is None:
            reference = next(self.temp_condproj.parameters())
        audio_memory = self.temp_condproj[0](
            audio_condition.to(reference)
        )
        text_memory = self.temp_condproj[1](
            text_condition.to(reference)
        )
        return audio_memory, text_memory

    def forward_temporal_projected(
        self,
        sequence: torch.Tensor,
        audio_memory: torch.Tensor,
        text_memory: torch.Tensor,
        sum_condition: torch.Tensor | None = None,
    ):
        """Temporal step accepting already projected static condition memory."""

        _, codebooks, _ = sequence.shape
        if codebooks != self.n_q:
            raise ValueError(
                f"Expected {self.n_q} codebooks, got {codebooks}."
            )

        temporal_input = None
        for codebook_index in range(self.n_q):
            embedding = self.temporal_gemb[codebook_index](
                sequence[:, codebook_index]
            )
            embedding = self.temporal_gproj[codebook_index](embedding)
            temporal_input = (
                embedding
                if temporal_input is None
                else temporal_input + embedding
            )
        if temporal_input is None:
            raise RuntimeError("Temporal input has no codebook embeddings.")

        if sum_condition is not None:
            speaker_embedding = self.spk_tempemb(sum_condition)
            temporal_input = temporal_input + speaker_embedding.to(
                temporal_input
            )

        transformer_out = self.temporal_transformer(
            temporal_input,
            memories=[
                audio_memory.to(temporal_input),
                text_memory.to(temporal_input),
            ],
        )
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        logits = self.temporal_classifier(transformer_out).unsqueeze(1)
        return transformer_out, logits


@dataclass
class _OfflineGLMGenState(_GLMGenState):
    audio_condition: torch.Tensor | None = None
    text_condition: torch.Tensor | None = None
    temporal_audio_memory: torch.Tensor | None = None
    temporal_text_memory: torch.Tensor | None = None
    condition_steps: int = 0


class GestureLMOfflineGen(GestureLMGen):
    """Autoregressive gesture generator with full offline audio/text memory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.glm_model, GTemporalDepthModel3Offline):
            raise TypeError(
                "GestureLMOfflineGen requires GTemporalDepthModel3Offline."
            )
        self._full_condition_tokens: torch.Tensor | None = None

    def set_full_condition(self, condition: torch.Tensor):
        """Install `[B, 1 + n_audio_codebooks, S]` condition tokens."""

        if self.is_streaming:
            raise RuntimeError(
                "Full conditioning must be installed before streaming starts."
            )
        if condition.dim() != 3 or condition.shape[1] < 2:
            raise ValueError(
                "Expected full condition shaped [B, 1+K_audio, S], got "
                f"{tuple(condition.shape)}."
            )
        self._full_condition_tokens = condition

    def _init_streaming_state(self, batch_size: int) -> _OfflineGLMGenState:
        if self._full_condition_tokens is None:
            raise RuntimeError(
                "Call set_full_condition(...) before generator.streaming(...)."
            )

        model = self.glm_model
        full_condition = self._full_condition_tokens.to(self.device)
        if full_condition.shape[0] != batch_size:
            raise ValueError(
                f"Expected condition batch {batch_size}, got "
                f"{full_condition.shape[0]}."
            )
        if (
            full_condition.shape[-1] % model.query2mem_scale
        ) != 0:
            raise ValueError(
                "Condition length must be divisible by query2mem_scale: "
                f"{full_condition.shape[-1]} vs {model.query2mem_scale}."
            )

        audio_codes = full_condition[:, 1:]
        text_codes = full_condition[:, :1]
        if self.cfg_coef != 1.0:
            audio_codes = torch.cat(
                [
                    audio_codes,
                    torch.full_like(
                        audio_codes, self.audio_codec_nulltoken
                    ),
                ],
                dim=0,
            )
            text_codes = torch.cat(
                [
                    text_codes,
                    torch.full_like(
                        text_codes, self.text_codec_nulltoken
                    ),
                ],
                dim=0,
            )

        audio_condition, text_condition = self.process_conditions(
            audio_codes,
            text_codes,
        )
        temporal_audio, temporal_text = (
            model.project_temporal_conditions(
                audio_condition,
                text_condition,
            )
        )

        condition_sum = (
            self.condition_tensors.to(self.device)
            if self.condition_tensors is not None
            else None
        )
        if condition_sum is not None:
            if condition_sum.shape[0] != batch_size:
                raise ValueError(
                    f"Expected speaker batch {batch_size}, got "
                    f"{condition_sum.shape[0]}."
                )
            if self.cfg_coef != 1.0:
                condition_sum = torch.cat(
                    [condition_sum, condition_sum],
                    dim=0,
                )

        cache = torch.full(
            (batch_size, model.n_q, 1),
            model.ungenerated_token_id,
            device=self.device,
            dtype=torch.long,
        )
        # Static K/V state is mutated outside CUDA graph capture. The depth
        # model retains the released generator's graph optimization.
        graphed_temp = CUDAGraphed(
            model.forward_temporal_projected,
            disable=True,
        )
        graphed_depth = CUDAGraphed(
            self.depformer_step,
            disable=self.device.type != "cuda",
        )
        state = _OfflineGLMGenState(
            batch_size=batch_size,
            device=self.device,
            cache=cache,
            initial=model._get_initial_token(),
            graphed_temp=graphed_temp,
            graphed_depth=graphed_depth,
            condition_sum=condition_sum,
            audio_condition=audio_condition,
            text_condition=text_condition,
            temporal_audio_memory=temporal_audio,
            temporal_text_memory=temporal_text,
            condition_steps=(
                full_condition.shape[-1] // model.query2mem_scale
            ),
        )

        streaming_batch = (
            batch_size if self.cfg_coef == 1.0 else 2 * batch_size
        )
        state.exit_stack.enter_context(
            model.streaming(streaming_batch)
        )
        state.reset_callback = model.reset_streaming
        return state

    @torch.no_grad()
    def step(
        self,
        condition: torch.Tensor | tp.List[torch.Tensor] | None = None,
        ca_query_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self._streaming_state
        if state is None:
            raise RuntimeError("Streaming state is not initialized.")
        if not isinstance(state, _OfflineGLMGenState):
            raise TypeError(f"Unexpected generator state: {type(state)!r}")
        if state.offset >= state.condition_steps:
            raise RuntimeError(
                "Requested more gesture steps than the installed offline "
                f"condition provides ({state.condition_steps})."
            )

        model = self.glm_model
        batch_size = state.batch_size
        if state.offset == 0:
            state.cache[:, :, 0] = model.initial_token_id
        temporal_input = state.cache[:, :, :1]
        if self.check and (
            temporal_input == model.ungenerated_token_id
        ).any():
            raise RuntimeError(
                "Temporal cache contains an ungenerated token at "
                f"step {state.offset}."
            )
        if self.cfg_coef != 1.0:
            temporal_input = temporal_input.repeat(2, 1, 1)

        if (
            state.temporal_audio_memory is None
            or state.temporal_text_memory is None
        ):
            raise RuntimeError("Projected offline memory is missing.")
        transformer_out, temporal_logits = state.graphed_temp(
            temporal_input,
            state.temporal_audio_memory,
            state.temporal_text_memory,
            state.condition_sum,
        )

        if self.cfg_coef != 1.0:
            temporal_logits, null_logits = temporal_logits.chunk(2, dim=0)
            temporal_logits = null_logits + self.cfg_coef * (
                temporal_logits - null_logits
            )
        temporal_logits[..., model.pad_token_id] = float("-inf")
        upper_q0 = sample_token(
            temporal_logits.float(),
            use_sampling=self.use_sampling,
            temp=self.temp_temporal,
            top_k=self.top_k_temp,
            top_p=self.top_p_temp,
        )[:, 0, 0]

        memory_start = state.offset * model.query2mem_scale
        memory_end = memory_start + model.query2mem_scale
        if state.audio_condition is None or state.text_condition is None:
            raise RuntimeError("Offline depth conditioning is missing.")
        current_audio = state.audio_condition[:, memory_start:memory_end]
        current_text = state.text_condition[:, memory_start:memory_end]

        if ca_query_padding_mask is None:
            # The released depformer helper expects a tensor in practice.
            # An all-false mask preserves "attend normally" without changing
            # the original generator implementation.
            depth_padding_mask = torch.zeros(
                batch_size,
                model.n_q - 1,
                1,
                device=upper_q0.device,
                dtype=torch.bool,
            )
        else:
            depth_padding_mask = ca_query_padding_mask[:, 1:]
        cfg_stop_mask = (
            depth_padding_mask[0, :, 0].cpu().tolist()
            if self.cfg_coef != 1.0
            and depth_padding_mask is not None
            else None
        )
        depth_tokens = state.graphed_depth(
            upper_q0,
            transformer_out,
            current_audio,
            current_text,
            state.condition_sum,
            depth_padding_mask,
            cfg_stop_mask,
            self.bp_dist,
        )

        state.offset += 1
        state.cache[:, 0, 0] = upper_q0
        state.cache[:, 1:, 0] = depth_tokens
        return state.cache.clone()
