"""Parameter-free masked-frame future-gesture teachers for MIBURI.

These variants keep the released MIBURI embeddings, temporal transformer,
classifier, and kinematic/depth transformer. No future encoder or fusion
parameters are added. Future gesture is exposed by:

1. physically replacing the supervised target frame and its lookahead guard
   with the existing per-codebook PAD/MASK embedding;
2. physically replacing gesture history older than MIBURI's original
   25-token temporal window;
3. running the existing temporal gesture self-attention bidirectionally; and
4. computing losses only at the physically masked target frame.

The trainer supplies one target time per sample. With the released codec,
``frame_chunk_size=2`` means ten 25-FPS motion frames (400 ms) correspond to
five temporal gesture tokens. Thus a target at ``t`` masks gesture tokens
``g[t:t+5]`` and first exposes ``g[t+5]``.

``GTemporalDepthModel3FutureGesture`` retains causal audio/text
cross-attention. Its trainer also removes audio/text after the selected target
for the entire sample, preventing a future gesture position from relaying
future condition information through a later transformer layer.

``GTemporalDepthModel3FutureGestureFullCondition`` instead exposes complete
paired audio/text memories. Both are offline diagnostic teachers: they cannot
perform standalone autoregressive generation, but the trainer can perform an
explicitly labeled oracle infill evaluation when ground-truth future gesture
is supplied by an evaluation dataset.
"""

import torch
from torch import nn

from .gesture_lm import GTemporalDepthModel3
from .gesture_lm_offline import _as_static_memory_attention


def build_masked_future_gesture_inputs(
    input_codes: torch.Tensor,
    target_codes: torch.Tensor,
    target_times: torch.Tensor,
    horizon_tokens: int,
    past_context_tokens: int,
    mask_token_id: int,
) -> torch.Tensor:
    """Build leak-free temporal inputs for one target time per sample.

    Gesture history before ``g[t-past_context_tokens]`` and the guard
    ``g[t:t+horizon_tokens]`` are physically absent. The surviving past is
    therefore identical to raw MIBURI's local causal gesture window, while
    the first true future gesture is ``g[t+horizon_tokens]``. True target
    codes are restored after that point so training-time lower/face part
    dropping does not silently remove the privileged future suffix.
    """

    if input_codes.shape != target_codes.shape or input_codes.dim() != 3:
        raise ValueError(
            "Expected matching [B,K,T] input/target codes, got "
            f"{tuple(input_codes.shape)} and {tuple(target_codes.shape)}."
        )
    if horizon_tokens <= 0:
        raise ValueError("horizon_tokens must be positive.")
    if past_context_tokens <= 0:
        raise ValueError("past_context_tokens must be positive.")
    batch, _, steps = input_codes.shape
    if target_times.shape != (batch,):
        raise ValueError(
            f"Expected target_times shape {(batch,)}, got "
            f"{tuple(target_times.shape)}."
        )

    temporal_codes = input_codes.clone()
    for batch_index, target_time_tensor in enumerate(target_times):
        target_time = int(target_time_tensor.item())
        future_start = target_time + horizon_tokens
        if target_time < 0 or future_start >= steps:
            raise ValueError(
                f"Target {target_time} with horizon {horizon_tokens} has no "
                f"future gesture in a {steps}-token sequence."
            )
        temporal_codes[
            batch_index,
            :,
            future_start:,
        ] = target_codes[
            batch_index,
            :,
            future_start:,
        ]
        temporal_codes[
            batch_index,
            :,
            target_time:future_start,
        ] = mask_token_id
        past_start = max(0, target_time - past_context_tokens)
        temporal_codes[
            batch_index,
            :,
            :past_start,
        ] = mask_token_id
    return temporal_codes


def truncate_condition_codes_after_targets(
    condition_codes: torch.Tensor,
    target_times: torch.Tensor,
    condition_steps_per_gesture: int,
    null_token_id: int,
) -> torch.Tensor:
    """Remove condition information strictly after each selected target."""

    if condition_codes.dim() != 3:
        raise ValueError(
            "Expected condition codes [B,K,S], got "
            f"{tuple(condition_codes.shape)}."
        )
    if condition_steps_per_gesture <= 0:
        raise ValueError("condition_steps_per_gesture must be positive.")
    if target_times.shape != (condition_codes.shape[0],):
        raise ValueError(
            "target_times must contain one time for every condition sample."
        )

    truncated = condition_codes.clone()
    condition_length = condition_codes.shape[-1]
    for batch_index, target_time_tensor in enumerate(target_times):
        target_time = int(target_time_tensor.item())
        keep_steps = min(
            condition_length,
            (target_time + 1) * condition_steps_per_gesture,
        )
        truncated[batch_index, :, keep_steps:] = null_token_id
    return truncated


class GTemporalDepthModel3FutureGesture(GTemporalDepthModel3):
    """Masked-frame future gesture with causal paired audio/text."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.causal:
            raise ValueError(
                "Construct the base model causally so the kinematic "
                "transformer and temporal cross-attention retain released "
                "MIBURI behavior."
            )

        # Reuse the existing self-attention weights with a full attention
        # mask. This changes no modules and adds no learned parameters.
        for layer in self.temporal_transformer.layers:
            layer.self_attn.causal = False
        self.temporal_gesture_causal = False
        self.temporal_condition_mode = "causal_audio_text"

    def build_temporal_key_padding_mask(
        self,
        temporal_input_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Block all physically absent frames as keys in every layer."""

        source_padding = (
            temporal_input_codes == self.pad_token_id
        ).all(dim=1)
        # Once an old-history prefix has been removed, raw causal MIBURI
        # would no longer see BOS either. For early targets, BOS remains
        # available exactly as in the raw causal window.
        bos_padding = source_padding[:, :1]
        return torch.cat([bos_padding, source_padding], dim=1)

    def forward_oracle_temporal_targets(
        self,
        temporal_input_codes: torch.Tensor,
        audio_codes: torch.Tensor,
        text_codes: torch.Tensor,
        sum_condition: torch.Tensor,
        target_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate only the selected masked temporal queries.

        This is the temporal half of oracle future-gesture evaluation. The
        caller is responsible for constructing a different physically masked
        gesture sequence for every target and, for the causal-condition
        teacher, removing audio/text after that target.

        Returns:
            ``transformer_out`` shaped ``[B, 1, D]`` and upper-q0 logits
            shaped ``[B, 1, 1, card + 1]``. Keeping singleton time axes makes
            the result directly compatible with the released kinematic
            autoregressive rollout helpers.
        """

        if temporal_input_codes.dim() != 3:
            raise ValueError(
                "Expected temporal_input_codes [B,K,T], got "
                f"{tuple(temporal_input_codes.shape)}."
            )
        batch, codebooks, steps = temporal_input_codes.shape
        if codebooks != self.n_q:
            raise ValueError(
                f"Expected {self.n_q} gesture codebooks, got {codebooks}."
            )
        if target_times.shape != (batch,):
            raise ValueError(
                f"Expected target_times shape {(batch,)}, got "
                f"{tuple(target_times.shape)}."
            )
        if ((target_times < 0) | (target_times >= steps)).any():
            raise ValueError(
                "Oracle target times must lie inside the gesture sequence."
            )
        if sum_condition.shape != (batch,):
            raise ValueError(
                f"Expected one speaker id per oracle target, got "
                f"{tuple(sum_condition.shape)}."
            )

        initial = self._get_initial_token().expand(
            batch,
            codebooks,
            -1,
        )
        temporal_sequence = torch.cat(
            [initial, temporal_input_codes],
            dim=-1,
        )
        key_padding_mask = self.build_temporal_key_padding_mask(
            temporal_input_codes
        )
        audio_condition, text_condition = self.process_conditions(
            audio_codes,
            text_codes,
        )
        transformer_out, logits = self.forward_temporal(
            temporal_sequence,
            audio_condition=audio_condition.squeeze(1),
            text_condition=text_condition.squeeze(1),
            sum_condition=sum_condition[:, None].expand(
                -1,
                temporal_sequence.shape[-1],
            ),
            key_padding_mask=key_padding_mask,
        )
        batch_indices = torch.arange(
            batch,
            device=target_times.device,
        )
        selected_out = transformer_out[
            batch_indices,
            target_times,
        ].unsqueeze(1)
        selected_logits = logits[
            batch_indices,
            :,
            target_times,
        ].unsqueeze(2)
        return selected_out, selected_logits

    def forward(self, *args, temporal_include_last_input=True, **kwargs):
        # Include [BOS, g0, ..., g(T-1)] as self-attention inputs. The base
        # forward discards the extra final output and keeps T aligned targets.
        temporal_input_codes = kwargs.get("temporal_input_codes")
        if temporal_input_codes is None:
            raise RuntimeError(
                "Masked-frame future teachers require "
                "temporal_input_codes."
            )
        kwargs["temporal_key_padding_mask"] = (
            self.build_temporal_key_padding_mask(temporal_input_codes)
        )
        return super().forward(
            *args,
            temporal_include_last_input=temporal_include_last_input,
            **kwargs,
        )


class GTemporalDepthModel3FutureGestureFullCondition(
    GTemporalDepthModel3FutureGesture
):
    """Masked-frame future gesture plus complete paired audio/text."""

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
