"""Leakage-safe, sampled GlobalRegret future-content training controls.

Only the detached teacher's temporal speech memory is intervened on. Each
target gets its own recomputed gesture prefix; depth receives true current
speech and the student's within-step (possibly stochastic) gesture prefix.
"""
from __future__ import annotations

import random

import torch

from .gesture_lm_shared_regret import (
    _prepend_initial_and_shift,
    _process_conditions_squeezed,
    relaxed_temporal_cross_attention,
)


def eligible_context_positions(length: int) -> list[int]:
    """Enough observed groups for the entire remaining future; no empty future."""
    if length < 2:
        return []
    return list(range((length - 1) // 2, length - 1))


def sample_context_positions(length: int, count: int, seed: int) -> list[int]:
    if count < 1:
        raise ValueError("regret_context_targets must be positive")
    eligible = eligible_context_positions(length)
    return sorted(random.Random(seed).sample(eligible, min(count, len(eligible))))


def past_as_future(memory: torch.Tensor, step: int, scale: int = 1) -> torch.Tensor:
    """[B,S,D] memory: preserve <=t, copy [cut-future:cut] into [cut:end].

    The historical segment ends with the current group (inclusive). Nothing
    after that group is used, and no wrapping, padding or repetition is added.
    This works for raw tokens transposed to [B,S,K] and processed embeddings.
    """
    if memory.ndim != 3 or scale < 1 or memory.shape[1] % scale:
        raise ValueError("Expected [B,S,D] memory with complete aligned groups")
    cut = (step + 1) * scale
    future = memory.shape[1] - cut
    if step < 0 or future <= 0 or cut < future:
        raise ValueError("Target requires nonempty future and sufficient observed history")
    result = memory.clone()
    result[:, cut:] = memory[:, cut - future:cut]
    return result


@torch.no_grad()
def forward_future_context_teacher(
    model, *, input_codes, audio_codes, text_codes, sum_condition, positions,
    context="matched", include_depth_levels=True, depth_input_codes=None,
    ca_depth_padding_mask=None,
):
    """Return detached [B,K,N,V] logits at selected target positions.

    GlobalRegret past access and full clip end are retained in both controls.
    Recompute *all* earlier hidden states with the selected memory, rather
    than splicing a cache which may already contain true-future information.
    """
    if context not in ("matched", "past_as_future"):
        raise ValueError(f"Unknown future context: {context}")
    if model.temporal_transformer.is_streaming:
        raise ValueError("Future-content teacher cannot reuse a streaming cache")
    batch, _, length = input_codes.shape
    scale = model.query2mem_scale
    if audio_codes.shape[0] != batch or text_codes.shape[0] != batch:
        raise ValueError("Gesture and speech batch sizes differ")
    if audio_codes.shape[-1] != length * scale or text_codes.shape[-1] != length * scale:
        raise ValueError("Speech length must match the complete gesture clip")
    positions = list(positions)
    eligible = set(eligible_context_positions(length))
    if not positions or len(set(positions)) != len(positions) or any(t not in eligible for t in positions):
        raise ValueError("Select distinct targets with nonempty future and enough history")
    depth_codes = input_codes if depth_input_codes is None else depth_input_codes
    if depth_codes.shape != input_codes.shape:
        raise ValueError("Student and teacher depth prefixes must have identical shapes")
    audio, text = _process_conditions_squeezed(model, audio_codes, text_codes)
    sequence = _prepend_initial_and_shift(model, input_codes)
    outputs = []
    with relaxed_temporal_cross_attention(model.temporal_transformer):
        for step in positions:
            teacher_audio = past_as_future(audio, step, scale) if context == "past_as_future" else audio
            teacher_text = past_as_future(text, step, scale) if context == "past_as_future" else text
            hidden, temporal = model.forward_temporal(
                sequence[:, :, :step + 1], audio_condition=teacher_audio,
                text_condition=teacher_text,
                sum_condition=sum_condition[:, None].expand(-1, step + 1),
            )
            current = temporal[:, :, -1:]
            if include_depth_levels:
                prefix = depth_codes[:, :-1, step:step + 1]
                depth = model.forward_depth_training(
                    prefix, transformer_out=hidden[:, -1:],
                    # Always the true current audio/text group, never donor speech.
                    audio_condition=audio[:, step * scale:(step + 1) * scale],
                    text_condition=text[:, step * scale:(step + 1) * scale],
                    sum_condition=sum_condition[:, None],
                    depth_padding_mask=prefix == model.pad_token_id,
                    ca_query_padding_mask=(ca_depth_padding_mask[:, 1:, step:step + 1]
                                           if ca_depth_padding_mask is not None else None),
                )
                current = torch.cat((current, depth), dim=1)
            outputs.append(current)
    return torch.cat(outputs, dim=2)
