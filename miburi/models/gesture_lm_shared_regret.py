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

``forward_dense_future_gesture_teacher_view`` is the complementary,
same-stream mechanism: it relaxes gesture *self*-attention instead of
cross-attention, and does need target-position masking (mirroring the
paper's Eq. 17) because the privileged information now lives in the same
stream as the prediction target -- implemented *literally* as the paper's
own mask: a per-query attention-bias exclusion of the ``horizon_tokens``
keys starting at that query's own target (``horizon_tokens=1``, the
default, matches Eq. 17 exactly; larger values widen the guard, the same
knob the old sparse mechanism's ``horizon_tokens`` offered, just applied
densely to every position), over otherwise completely unmodified tokens
(no token substitution). Because the exclusion is a property of the
query-key pair rather than the input tokens, every position gets a valid
teacher view in a single forward pass regardless of the window's width,
matching the density of the cross-attention mechanism above. Cross-attention
stays causal in this view so its signal isn't conflated with the speech/text
mechanism above.

Getting this required two small, additive changes outside this module:
``StreamingMultiheadCrossAttention.forward`` gained an optional
``extra_attn_bias`` parameter (ANDed into its internally-computed mask,
``miburi/modules/transformer.py``), and ``GTemporalDepthModel3.forward_temporal``
gained a ``self_attn_bias`` passthrough (``miburi/models/gesture_lm.py``).
Both default to ``None`` and are no-ops for every other existing caller.
"""

from __future__ import annotations

from contextlib import contextmanager
import typing as tp

import torch
import torch.nn.functional as F


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


def _prepend_initial_full(model, codes: torch.Tensor) -> torch.Tensor:
    """Like ``_prepend_initial_and_shift`` but keeps the final position.

    ``GTemporalDepthModel3.forward`` calls this its ``temporal_include_last_
    input=True`` path: the untruncated, length-``T+1`` sequence is required
    so every real target -- including the very last one, ``codes[:, :, -1]``
    -- has a corresponding key position for a target-exclusion bias to
    actually name. The plain shift (which truncates) silently drops that
    key, leaving the last position with nothing to exclude.
    """

    B, K, _ = codes.shape
    initial = model._get_initial_token().expand(B, K, -1)
    return torch.cat([initial, codes], dim=2)


def _shifted_key_padding_mask(
    valid_position_mask: torch.Tensor, seq_len: int,
) -> torch.Tensor:
    """``[B, seq_len]`` boolean key-padding mask (True == padding) for a
    ``_prepend_initial_*``-shifted sequence of the given length.

    Position 0 (the initial token) is always valid. Position ``m >= 1``
    maps to ``valid_position_mask[:, m-1]`` -- shifted by one, matching how
    ``_prepend_initial_and_shift``/``_prepend_initial_full`` build the
    sequence from ``codes``. Works for both the truncated (``seq_len=T``)
    and untruncated (``seq_len=T+1``) sequences.
    """

    B = valid_position_mask.shape[0]
    device = valid_position_mask.device
    shifted_valid = torch.ones(B, seq_len, dtype=torch.bool, device=device)
    body_len = seq_len - 1
    shifted_valid[:, 1:] = valid_position_mask[:, :body_len]
    return ~shifted_valid


def _build_target_exclusion_bias(
    seq_len: int, device, *, horizon_tokens: int = 1,
) -> torch.Tensor:
    """``[1, 1, seq_len, seq_len]`` boolean mask, True == allowed to attend,
    False for keys in ``[query+1, query+1+horizon_tokens)`` -- the
    shifted-sequence positions holding the query's own prediction target
    and, for ``horizon_tokens > 1``, the immediately following ones too.
    Keys at or before the query (``j <= i``, its genuine causal history)
    are never excluded, so no row can ever end up fully masked regardless
    of ``horizon_tokens``.

    With ``horizon_tokens=1`` (the default) this is exactly the paper's
    GlobalRegret mask (Eq. 17: ``M[i, i+1] = -inf``, everywhere else 0).
    Larger values widen the guard the same way the old sparse mechanism's
    ``horizon_tokens`` did, just applied to every query position in this
    one dense pass instead of one sampled target per sample. Either way
    this is a pure attention-level exclusion over otherwise intact,
    unmodified tokens: it inherits the module's leak-permissive caveat
    (see the module docstring) -- it does not protect against MIBURI's
    causal gesture codec smearing information about a hidden position
    into a visible one.
    """

    if horizon_tokens < 1:
        raise ValueError("horizon_tokens must be at least one.")
    positions = torch.arange(seq_len, device=device)
    delta = positions.view(1, -1) - positions.view(-1, 1)  # key - query
    exclude = (delta >= 1) & (delta <= horizon_tokens)
    allow = ~exclude
    return allow.view(1, 1, seq_len, seq_len)


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
def forward_dense_future_gesture_teacher_view(
    model,
    *,
    input_codes: torch.Tensor,
    audio_codes: torch.Tensor,
    text_codes: torch.Tensor,
    sum_condition: torch.Tensor,
    valid_position_mask: torch.Tensor,
    horizon_tokens: int = 1,
    ca_depth_padding_mask: torch.Tensor | None = None,
    include_depth_levels: bool = True,
    depth_input_codes: torch.Tensor | None = None,
) -> tp.Tuple[torch.Tensor, torch.Tensor | None]:
    """Dense, single-pass, bidirectional-self-attention teacher view.

    The paper's own GlobalRegret mechanism (Eq. 17), applied to MIBURI's
    gesture stream: every position gets a simultaneous, valid teacher view
    in one forward pass, via a per-query attention-bias exclusion (block
    the ``horizon_tokens`` keys starting at each query's own target, via
    ``_build_target_exclusion_bias``) rather than a single selected,
    token-substituted target. No token is ever replaced or corrupted here
    -- the true codes are always the input; only the attention pattern
    changes, which is what makes every position simultaneously valid in
    one pass. ``horizon_tokens=1`` (the default) hides only the literal
    target; larger values widen the guard for every position at once.

    Requires the untruncated sequence (``_prepend_initial_full``, MIBURI's
    ``temporal_include_last_input=True`` path) so the very last position
    also has a key to exclude -- the plain truncated shift silently drops
    it. Requires ``valid_position_mask`` since bidirectional attention can
    reach into within-buffer padding that causal attention never could.

    Leak-permissive by design: this does not protect against MIBURI's
    causal gesture codec smearing information about a hidden position
    into a visible later one. Runs entirely under ``no_grad``.
    """

    _, _, T = input_codes.shape
    full_sequence = _prepend_initial_full(model, input_codes)
    seq_len = full_sequence.shape[-1]
    temporal_sum_condition = sum_condition.unsqueeze(1).expand(-1, seq_len)
    audio_condition, text_condition = _process_conditions_squeezed(
        model, audio_codes, text_codes,
    )
    self_attn_bias = _build_target_exclusion_bias(
        seq_len, full_sequence.device, horizon_tokens=horizon_tokens,
    )
    key_padding_mask = _shifted_key_padding_mask(valid_position_mask, seq_len)

    with relaxed_temporal_self_attention(model.temporal_transformer):
        teacher_transformer_out, teacher_temp_logits = model.forward_temporal(
            full_sequence,
            audio_condition=audio_condition,
            text_condition=text_condition,
            sum_condition=temporal_sum_condition,
            key_padding_mask=key_padding_mask,
            self_attn_bias=self_attn_bias,
        )

    teacher_transformer_out = teacher_transformer_out[:, :T]
    teacher_temp_logits = teacher_temp_logits[:, :, :T]

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
