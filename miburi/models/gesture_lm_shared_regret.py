"""Shared-weight Global Regret distillation for ``GTemporalDepthModel3``.

Unlike ``gesture_lm_regret.py`` (a separately pretrained, frozen full-future
checkpoint distilled into a causal student), this module builds the
"teacher" view from the *same* weights as the causal student, differing only
in the temporal transformer's cross-attention mask -- exactly the dual-view
mechanism described in "Regret Pre-training: Bridging Prior and Posterior
Views for Enhanced Knowledge Grounding". No second model is constructed and
no checkpoint is loaded; the student's own audio/text cross-attention is
temporarily relaxed to bidirectional (GlobalRegret) for one extra,
gradient-free forward pass per training step.

``forward_teacher_view`` (GlobalRegret over speech/text) leaves gesture
self-attention untouched: training is fully teacher-forced, so ``g(<t)`` is
already identical, ordinary input for the student -- there is nothing
"privileged" to unlock there. The only future information that view exposes
is future speech/text conditioning, which lives in a different stream than
the gesture target being predicted, so (unlike the paper's own-stream
teacher) no target-position masking is required to avoid leaking the answer.

``forward_masked_target_teacher_view`` is the complementary, same-stream
mechanism: it relaxes gesture *self*-attention instead of cross-attention,
and does need target-position masking (mirroring the paper's Eq. 17)
because the privileged information now lives in the same stream as the
prediction target. It is deliberately sparse (one selected timestep per
sample) rather than dense, and cross-attention stays causal in that view so
its signal isn't conflated with the speech/text mechanism above.
"""

from __future__ import annotations

from contextlib import contextmanager
import typing as tp

import torch
import torch.nn.functional as F

from .gesture_lm_future_gesture import build_masked_future_gesture_inputs


def _prepend_initial_and_shift(model, codes: torch.Tensor) -> torch.Tensor:
    """Reproduce ``GTemporalDepthModel3.forward``'s temporal input shift.

    Kept as a tiny, read-only duplicate of that five-line snippet rather than
    a model-file change, since ``forward()`` does not expose the shifted
    sequence it hands to ``forward_temporal``.
    """

    B, K, _ = codes.shape
    initial = model._get_initial_token().expand(B, K, -1)
    sequence = torch.cat([initial, codes], dim=2)
    return sequence[:, :, :-1]


def _process_conditions_squeezed(model, audio_codes, text_codes):
    audio_condition, text_condition = model.process_conditions(
        audio_codes, text_codes,
    )
    return audio_condition.squeeze(1), text_condition.squeeze(1)


def _forward_depth_branch(
    model,
    *,
    teacher_transformer_out: torch.Tensor,
    input_codes: torch.Tensor,
    depth_input_codes: torch.Tensor | None,
    audio_condition: torch.Tensor,
    text_condition: torch.Tensor,
    sum_condition: torch.Tensor,
    ca_depth_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Shared depth-branch call for every teacher view in this module.

    ``depth_input_codes`` defaults to the plain ``input_codes`` -- the depth
    transformer's own teacher-forced prefix is left untouched by whatever
    privileged view produced ``teacher_transformer_out``; only the temporal
    hidden state it conditions on changes.
    """

    dep_inpseq = (
        depth_input_codes if depth_input_codes is not None else input_codes
    )
    if dep_inpseq.shape != input_codes.shape:
        raise ValueError(
            "depth_input_codes must match input_codes' shape, got "
            f"{tuple(dep_inpseq.shape)} and {tuple(input_codes.shape)}."
        )
    _, _, T = input_codes.shape
    depth_padding_mask = dep_inpseq == model.pad_token_id
    dep_sum_condition = sum_condition.unsqueeze(1).expand(-1, T)
    ca_query_padding_mask = (
        ca_depth_padding_mask[:, 1:, :]
        if ca_depth_padding_mask is not None
        else None
    )
    return model.forward_depth_training(
        dep_inpseq[:, :-1, :],
        transformer_out=teacher_transformer_out,
        audio_condition=audio_condition,
        text_condition=text_condition,
        sum_condition=dep_sum_condition,
        depth_padding_mask=depth_padding_mask[:, :-1, :],
        ca_query_padding_mask=ca_query_padding_mask,
    )


@contextmanager
def relaxed_temporal_cross_attention(temporal_transformer):
    """Temporarily make ``temporal_transformer``'s cross-attention bidirectional.

    Flips ``.causal = False`` on every cross-attention head (audio and text
    memories) of every layer, in place, on the *same* parameter-carrying
    modules the causal student uses -- no weight copy, no second model.
    Restoration is guaranteed via ``finally`` so a mid-pass exception (e.g.
    an OOM) can never leave the student's own causal path corrupted for the
    next training step.
    """

    toggled: list[tuple[object, bool]] = []
    try:
        for layer in temporal_transformer.layers:
            for cross_attn in layer.cross_attns:
                toggled.append((cross_attn, cross_attn.causal))
                cross_attn.causal = False
        yield
    finally:
        for cross_attn, original_causal in toggled:
            cross_attn.causal = original_causal


@contextmanager
def relaxed_temporal_self_attention(temporal_transformer):
    """Temporarily make ``temporal_transformer``'s gesture self-attention
    bidirectional, leaving cross-attention (audio/text) untouched.

    Mirrors ``relaxed_temporal_cross_attention`` exactly but flips the other
    attention module each decoder layer owns (``layer.self_attn`` instead of
    ``layer.cross_attns``). Used for the masked-target future-gesture
    teacher view, which isolates future *gesture* information from the
    future speech/text signal the cross-attention variant supplies.
    """

    toggled: list[tuple[object, bool]] = []
    try:
        for layer in temporal_transformer.layers:
            toggled.append((layer.self_attn, layer.self_attn.causal))
            layer.self_attn.causal = False
        yield
    finally:
        for self_attn, original_causal in toggled:
            self_attn.causal = original_causal


@torch.no_grad()
def forward_teacher_view(
    model,
    *,
    input_codes: torch.Tensor,
    audio_codes: torch.Tensor,
    text_codes: torch.Tensor,
    sum_condition: torch.Tensor,
    ca_depth_padding_mask: torch.Tensor | None = None,
    include_depth_levels: bool = True,
    depth_input_codes: torch.Tensor | None = None,
) -> tp.Tuple[torch.Tensor, torch.Tensor | None]:
    """Run the GlobalRegret teacher view of ``model`` for one training step.

    Reuses the exact same weights, the exact same teacher-forced
    ``input_codes``, and the exact same conditioning as the causal student's
    own forward pass; only the temporal transformer's cross-attention mask
    is relaxed to bidirectional for the duration of this call. Runs entirely
    under ``no_grad`` -- the paper's stop-gradient teacher view.

    ``depth_input_codes`` lets a caller feed the depth transformer's own
    input prefix a *different* tensor than the temporal transformer's input
    -- e.g. a trainer that also does stochastic-RVQ prefix regularization
    (``UpperFaceLowerGTDM3FrozenTemporalRVQTrainer``-style) needs the
    teacher's depth branch to see the exact same stochastic prefix the
    student's depth branch was conditioned on, not the canonical codes,
    otherwise the KL would confound "future context" with "which RVQ
    prefix variant was used". Defaults to ``input_codes`` when omitted,
    which reproduces this function's original (pre-RVQ) behavior exactly.

    Returns ``(teacher_temp_logits, teacher_depth_logits)`` with the same
    shapes ``GTemporalDepthModel3.forward`` produces for its temporal/depth
    halves (``teacher_depth_logits`` is ``None`` when
    ``include_depth_levels`` is False).
    """

    temporal_sequence = _prepend_initial_and_shift(model, input_codes)
    temporal_sum_condition = sum_condition.unsqueeze(1).expand(
        -1, temporal_sequence.shape[-1],
    )
    audio_condition, text_condition = _process_conditions_squeezed(
        model, audio_codes, text_codes,
    )

    with relaxed_temporal_cross_attention(model.temporal_transformer):
        teacher_transformer_out, teacher_temp_logits = model.forward_temporal(
            temporal_sequence,
            audio_condition=audio_condition,
            text_condition=text_condition,
            sum_condition=temporal_sum_condition,
        )

    if not include_depth_levels:
        return teacher_temp_logits, None

    teacher_depth_logits = _forward_depth_branch(
        model,
        teacher_transformer_out=teacher_transformer_out,
        input_codes=input_codes,
        depth_input_codes=depth_input_codes,
        audio_condition=audio_condition,
        text_condition=text_condition,
        sum_condition=sum_condition,
        ca_depth_padding_mask=ca_depth_padding_mask,
    )
    return teacher_temp_logits, teacher_depth_logits


@torch.no_grad()
def forward_masked_target_teacher_view(
    model,
    *,
    input_codes: torch.Tensor,
    target_codes: torch.Tensor,
    target_times: torch.Tensor,
    audio_codes: torch.Tensor,
    text_codes: torch.Tensor,
    sum_condition: torch.Tensor,
    horizon_tokens: int,
    past_context_tokens: int,
    mask_token_id: int,
    ca_depth_padding_mask: torch.Tensor | None = None,
    include_depth_levels: bool = True,
    depth_input_codes: torch.Tensor | None = None,
) -> tp.Tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse, masked-target, bidirectional-self-attention teacher view.

    One selected timestep per sample (``target_times``) has its input
    tokens hidden, across every codebook, for ``[t, t+horizon_tokens)``;
    every other position -- including the true future beyond that guard --
    stays the intact, ordinary ground-truth input. Gesture self-attention
    is relaxed to bidirectional for this call so the hidden position can
    actually see that surrounding context; cross-attention to audio/text
    stays causal, isolating future *gesture* information from the future
    speech/text signal ``forward_teacher_view`` already supplies elsewhere.

    With ``horizon_tokens=1`` this hides only the literal target token --
    the paper's own Eq. 17 masking, applied to the gesture stream. Unlike
    the reset-future trainers' privileged views, the exposed future here is
    still the intact per-clip codec encoding: this function makes no
    attempt to prevent the causal codec's own receptive field from leaking
    information about the guarded interval into tokens just beyond it.
    That's deliberate -- it's a cheap, maximally-permissive upper-bound
    check for whether future gesture information helps at all, before
    paying for a leak-safe (reset-encoded) version.

    Runs entirely under ``no_grad``, and is sparse over ``T`` (self-attention
    is still computed at every position internally, but only the selected
    ``target_times`` positions are meaningful and expected to be read out).
    """

    masked_codes = build_masked_future_gesture_inputs(
        input_codes,
        target_codes,
        target_times,
        horizon_tokens=horizon_tokens,
        past_context_tokens=past_context_tokens,
        mask_token_id=mask_token_id,
    )
    temporal_sequence = _prepend_initial_and_shift(model, masked_codes)
    temporal_sum_condition = sum_condition.unsqueeze(1).expand(
        -1, temporal_sequence.shape[-1],
    )
    audio_condition, text_condition = _process_conditions_squeezed(
        model, audio_codes, text_codes,
    )

    with relaxed_temporal_self_attention(model.temporal_transformer):
        teacher_transformer_out, teacher_temp_logits = model.forward_temporal(
            temporal_sequence,
            audio_condition=audio_condition,
            text_condition=text_condition,
            sum_condition=temporal_sum_condition,
        )

    if not include_depth_levels:
        return teacher_temp_logits, None

    teacher_depth_logits = _forward_depth_branch(
        model,
        teacher_transformer_out=teacher_transformer_out,
        input_codes=input_codes,
        depth_input_codes=depth_input_codes,
        audio_condition=audio_condition,
        text_condition=text_condition,
        sum_condition=sum_condition,
        ca_depth_padding_mask=ca_depth_padding_mask,
    )
    return teacher_temp_logits, teacher_depth_logits


def dense_regret_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL(stopgrad(teacher) || student), dense over every position.

    ``student_logits``/``teacher_logits``: ``[B, K, T, V]``. ``valid_mask``:
    ``[B, K, T]`` (any dtype castable to float; nonzero == valid). Averages
    per-codebook mean KL across ``K``, mirroring the released per-codebook CE
    convention (``compute_training_ce_objective``) so this loss sits at a
    comparable scale to the causal objective it augments, and so codebooks
    that are entirely dropped in a batch (e.g. face/lower-body dropout)
    don't get silently zero-weighted into the K-average instead of excluded.

    The teacher tensor is treated as already gradient-free (produced under
    ``forward_teacher_view``'s ``no_grad``); it is also explicitly detached
    here so this function is safe to call on its own.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher logits must have identical shapes, got "
            f"{tuple(student_logits.shape)} and "
            f"{tuple(teacher_logits.shape)}."
        )
    if valid_mask.shape != student_logits.shape[:3]:
        raise ValueError(
            "valid_mask must have shape [B,K,T] matching the logits' "
            f"leading dims, got {tuple(valid_mask.shape)} for logits "
            f"{tuple(student_logits.shape)}."
        )

    student_log_probs = F.log_softmax(
        student_logits.float() / temperature, dim=-1,
    )
    teacher_probs = F.softmax(
        teacher_logits.float().detach() / temperature, dim=-1,
    )
    kl_per_position = (
        teacher_probs
        * (teacher_probs.clamp_min(1e-12).log() - student_log_probs)
    ).sum(dim=-1)  # [B, K, T]

    valid = valid_mask.to(kl_per_position.dtype)
    per_codebook_count = valid.sum(dim=(0, 2))  # [K]
    per_codebook_sum = (kl_per_position * valid).sum(dim=(0, 2))  # [K]
    has_valid = per_codebook_count > 0
    if not bool(has_valid.any()):
        return student_logits.new_zeros(())

    per_codebook_mean = (
        per_codebook_sum[has_valid] / per_codebook_count[has_valid]
    )
    return per_codebook_mean.mean() * (temperature**2)
