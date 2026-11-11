"""Evaluation-only controls for future speech, sharing one frozen generator.

Recompute the temporal prefix for each prediction. Reusing a matched-future
hidden-state cache in the mismatch condition would leak the correct future.
Only speech after the current group is substituted, jointly for audio/text.
"""

from __future__ import annotations

import torch

from .gesture_lm_condition_c import _condition_c_teacher_attention


CONDITIONS = ("causal", "matched_future", "mismatched_future")


def speech_views(audio, text, donor_audio, donor_text, *, step, clip_start, clip_end, scale=1):
    """Build [3,S,D] memories; donor represents this clip's aligned speech.

    Bounds are gesture-step coordinates, clip_end exclusive. All memories
    start at recording time zero; the depth stage gets the original current
    group separately. This function also works on unembedded [1,S,...] arrays.
    """
    if scale < 1 or not 0 <= clip_start <= step < clip_end:
        raise ValueError("Invalid speech splice bounds")
    if audio.shape[0] != 1 or text.shape[0] != 1:
        raise ValueError("One target recording is required")
    end = clip_end * scale
    cut = (step + 1) * scale
    donor_cut = (step + 1 - clip_start) * scale
    donor_end = (clip_end - clip_start) * scale
    result = []
    for original, donor in ((audio, donor_audio), (text, donor_text)):
        if original.shape[1] < end or donor.shape[0] != 1 or donor.shape[1] != donor_end:
            raise ValueError("Matched and mismatched speech must have identical lengths")
        if original.shape[2:] != donor.shape[2:]:
            raise ValueError("Target/donor speech feature shapes differ")
        views = original[:, :end].repeat(3, *([1] * (original.ndim - 1)))
        views[2:3, cut:end] = donor[:, donor_cut:donor_end]
        result.append(views)
    return tuple(result)


def view_attention_bias(query_steps, memory_groups, clip_steps, *, scale=1, batches=3, device="cpu"):
    """Restrict each view's upper boundary; Condition C supplies past bounds."""
    if min(query_steps, memory_groups, clip_steps, scale) < 1 or batches % 3:
        raise ValueError("Invalid comparison dimensions")
    query = torch.arange(query_steps, device=device)
    keys = torch.arange(memory_groups * scale, device=device) // scale
    future_end = (query // clip_steps + 1) * clip_steps - 1
    upper = torch.stack((query, future_end, future_end))
    allowed = keys[None, None, :] <= upper[:, :, None]
    return allowed[:, None].repeat(batches // 3, 1, 1, 1)


@torch.no_grad()
def temporal_prediction(model, histories, audio_views, text_views, speaker, clip_steps):
    """Return current hidden states/logits; histories contain g_<t only."""
    if model.training or model.temporal_transformer.is_streaming:
        raise ValueError("Use an eval-mode model without a temporal streaming cache")
    batch, codebooks, previous_steps = histories.shape
    if codebooks != model.n_q or batch % 3 or audio_views.shape[0] != batch:
        raise ValueError("Expected three aligned views per history block")
    scale = model.query2mem_scale
    if audio_views.shape[1] % scale or text_views.shape[:2] != audio_views.shape[:2]:
        raise ValueError("Audio/text memory clocks differ")
    sequence = torch.cat((model._get_initial_token().expand(batch, model.n_q, -1), histories), -1)
    speakers = torch.full((batch, previous_steps + 1), int(speaker), dtype=torch.long, device=histories.device)
    bias = view_attention_bias(previous_steps + 1, audio_views.shape[1] // scale, clip_steps,
                               scale=scale, batches=batch, device=histories.device)
    with _condition_c_teacher_attention(model):
        hidden, logits = model.forward_temporal(sequence, audio_views, text_views,
                                                sum_condition=speakers, cross_attn_bias=bias)
    return hidden[:, -1:], logits[:, :, -1:]


def depth_cross_mask(model, batch, *, drop_lower, device):
    mask = torch.zeros(batch, model.n_q - 1, 1, dtype=torch.bool, device=device)
    if drop_lower:
        mask[:, 7:15] = True  # global codebooks 8--15
    return mask


@torch.no_grad()
def teacher_forced_logits(model, hidden, temporal_logits, current_codes, audio_current,
                          text_current, speaker, *, drop_lower=True):
    """All codebook logits with the same canonical within-step gold prefix."""
    batch = hidden.shape[0]
    codes = current_codes.expand(batch, -1, -1)
    depth = model.forward_depth_training(
        codes[:, :-1], transformer_out=hidden, audio_condition=audio_current,
        text_condition=text_current,
        sum_condition=torch.full((batch, 1), int(speaker), dtype=torch.long, device=hidden.device),
        depth_padding_mask=codes[:, :-1] == model.pad_token_id,
        ca_query_padding_mask=depth_cross_mask(model, batch, drop_lower=drop_lower, device=hidden.device),
    )
    return torch.cat((temporal_logits, depth), dim=1).squeeze(2)


def choose_token(logits, *, pad_id, sampling, temperature, top_p, uniform):
    """Common uniform variate across conditions; correct marginal top-p law."""
    logits = logits.float().clone()
    logits[..., pad_id] = -torch.inf
    if not sampling:
        return logits.argmax(-1)
    if temperature <= 0 or not 0 < top_p <= 1 or not 0 <= uniform < 1:
        raise ValueError("Invalid sampling parameters")
    probabilities = (logits / temperature).softmax(-1)
    if top_p < 1:
        ordered, indices = probabilities.sort(descending=True, dim=-1)
        ordered = ordered.masked_fill(ordered.cumsum(-1) - ordered >= top_p, 0)
        probabilities = torch.zeros_like(probabilities).scatter(-1, indices, ordered)
    probabilities /= probabilities.sum(-1, keepdim=True)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Nonfinite token probabilities")
    return (probabilities.cumsum(-1) <= uniform).sum(-1).clamp_max(logits.shape[-1] - 1)


@torch.no_grad()
def generate_current_codes(model, hidden, temporal_logits, audio_current, text_current,
                           speaker, *, rng, sampling=True, cfg_scale=1.0, drop_lower=True):
    """Generate one 20-code vector per view; reset depth cache at each step.

    For CFG, inputs are conditional three views followed by three null-speech
    views. Speaker conditions and gesture history stay unchanged in the null
    branch. Token-prediction metrics use the unguided conditional logits.
    """
    conditional_batch = 3
    batch = hidden.shape[0]
    if batch != conditional_batch * (2 if cfg_scale != 1 else 1):
        raise ValueError("Incorrect generation CFG batch size")

    def sample(logits, top_p, apply_cfg=True):
        logits = logits.reshape(batch, -1)
        if cfg_scale != 1:
            conditional, null = logits.chunk(2)
            logits = null + cfg_scale * (conditional - null) if apply_cfg else conditional
        return choose_token(logits, pad_id=model.pad_token_id, sampling=sampling,
                            temperature=0.9, top_p=top_p,
                            uniform=torch.rand((), generator=rng).item())

    current = [sample(temporal_logits, 0.8)]
    cross_mask = depth_cross_mask(model, batch, drop_lower=drop_lower, device=hidden.device)
    speakers = torch.full((batch, 1), int(speaker), dtype=torch.long, device=hidden.device)
    distribution = model.bp_dist[1:] if model.body_part_emb is not None else None
    with model.depth_transformer.streaming(batch):
        for depth in range(model.n_q - 1):
            code = current[-1][:, None, None]
            if cfg_scale != 1:
                code = code.repeat(2, 1, 1)
            logits = model.forward_depth(
                depth, code, transformer_out=hidden, audio_condition=audio_current,
                text_condition=text_current, sum_condition=speakers,
                ca_query_padding_mask=cross_mask[:, depth:depth + 1], bp_dist=distribution,
            )
            current.append(sample(logits, 0.95, apply_cfg=not (drop_lower and 7 <= depth < 15)))
    return torch.stack(current, dim=1).unsqueeze(-1)
