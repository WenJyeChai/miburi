"""Focused checks for parameter-free masked-frame gesture teachers."""

import torch

from miburi.models.gesture_lm import GTemporalDepthModel3, GestureLMGen
from miburi.models.gesture_lm_future_gesture import (
    GTemporalDepthModel3FutureGesture,
    GTemporalDepthModel3FutureGestureFullCondition,
    build_masked_future_gesture_inputs,
    truncate_condition_codes_after_targets,
)
from miburi.models.gesture_lm_offline import StaticMemoryCrossAttention
from scripts.smoke_test_gesture_lm_offline import (
    _batch,
    _model_kwargs,
)


def _make_teacher(model_class=GTemporalDepthModel3FutureGesture):
    return model_class(**_model_kwargs())


def _masked_inputs(
    model,
    codes,
    target_times,
    horizon_tokens,
    past_context_tokens=25,
):
    return build_masked_future_gesture_inputs(
        codes,
        codes,
        target_times,
        horizon_tokens=horizon_tokens,
        past_context_tokens=past_context_tokens,
        mask_token_id=model.pad_token_id,
    )


def _temporal_logits(
    model,
    temporal_codes,
    audio,
    text,
    speaker,
):
    initial = model._get_initial_token().expand(
        temporal_codes.shape[0],
        model.n_q,
        -1,
    )
    temporal_input = torch.cat([initial, temporal_codes], dim=-1)
    audio_condition, text_condition = model.process_conditions(audio, text)
    temporal_speaker = speaker[:, None].expand(
        -1,
        temporal_input.shape[-1],
    )
    _, logits = model.forward_temporal(
        temporal_input,
        audio_condition.squeeze(1),
        text_condition.squeeze(1),
        temporal_speaker,
    )
    return logits[:, :, : temporal_codes.shape[-1]]


def test_target_and_guard_are_physically_absent():
    model = _make_teacher()
    codes, _, _, _ = _batch()
    target_times = torch.tensor([0, 1])
    horizon = 2
    masked = _masked_inputs(model, codes, target_times, horizon)

    for batch_index, target_time in enumerate(target_times.tolist()):
        assert (
            masked[
                batch_index,
                :,
                target_time:target_time + horizon,
            ]
            == model.pad_token_id
        ).all()
        torch.testing.assert_close(
            masked[batch_index, :, target_time + horizon:],
            codes[batch_index, :, target_time + horizon:],
        )


def test_history_older_than_raw_miburi_context_is_absent():
    model = _make_teacher()
    codes = torch.randint(model.card, (1, model.n_q, 12))
    target_times = torch.tensor([8])
    masked = _masked_inputs(
        model,
        codes,
        target_times,
        horizon_tokens=2,
        past_context_tokens=3,
    )
    # Query t keeps source gestures g[t-context:t] = g[5:8].
    assert (masked[:, :, :5] == model.pad_token_id).all()
    torch.testing.assert_close(masked[:, :, 5:8], codes[:, :, 5:8])
    assert (masked[:, :, 8:10] == model.pad_token_id).all()
    torch.testing.assert_close(masked[:, :, 10:], codes[:, :, 10:])
    key_padding = model.build_temporal_key_padding_mask(masked)
    # BOS + source g[0:5] are outside the raw past window.
    assert key_padding[:, :6].all()
    assert not key_padding[:, 6:9].any()
    # Source g[8:10] is the target/400-ms guard.
    assert key_padding[:, 9:11].all()
    assert not key_padding[:, 11:].any()


def test_current_and_guard_cannot_change_target_logits_but_future_can():
    torch.manual_seed(21)
    model = _make_teacher().eval()
    codes, audio, text, speaker = _batch()
    target_times = torch.zeros(codes.shape[0], dtype=torch.long)
    horizon = 2

    reference = _temporal_logits(
        model,
        _masked_inputs(model, codes, target_times, horizon),
        audio,
        text,
        speaker,
    )

    changed_hidden = codes.clone()
    changed_hidden[:, :, :horizon] = (
        changed_hidden[:, :, :horizon] + 1
    ) % model.card
    hidden_logits = _temporal_logits(
        model,
        _masked_inputs(
            model,
            changed_hidden,
            target_times,
            horizon,
        ),
        audio,
        text,
        speaker,
    )
    torch.testing.assert_close(
        reference[:, :, 0],
        hidden_logits[:, :, 0],
    )

    changed_future = codes.clone()
    changed_future[:, :, horizon] = (
        changed_future[:, :, horizon] + 1
    ) % model.card
    future_logits = _temporal_logits(
        model,
        _masked_inputs(
            model,
            changed_future,
            target_times,
            horizon,
        ),
        audio,
        text,
        speaker,
    )
    assert not torch.allclose(
        reference[:, :, 0],
        future_logits[:, :, 0],
    )


def test_condition_truncation_is_per_sample_and_target_aligned():
    condition = torch.arange(2 * 2 * 8).reshape(2, 2, 8)
    target_times = torch.tensor([1, 2])
    truncated = truncate_condition_codes_after_targets(
        condition,
        target_times,
        condition_steps_per_gesture=2,
        null_token_id=-1,
    )
    torch.testing.assert_close(truncated[0, :, :4], condition[0, :, :4])
    torch.testing.assert_close(truncated[1, :, :6], condition[1, :, :6])
    assert (truncated[0, :, 4:] == -1).all()
    assert (truncated[1, :, 6:] == -1).all()


def test_condition_variants_have_original_parameter_count_and_masks():
    torch.manual_seed(22)
    original = GTemporalDepthModel3(**_model_kwargs())
    causal = _make_teacher(GTemporalDepthModel3FutureGesture)
    full = _make_teacher(
        GTemporalDepthModel3FutureGestureFullCondition
    )
    expected = sum(parameter.numel() for parameter in original.parameters())
    assert sum(parameter.numel() for parameter in causal.parameters()) == expected
    assert sum(parameter.numel() for parameter in full.parameters()) == expected
    assert set(original.state_dict()) == set(causal.state_dict())
    assert set(original.state_dict()) == set(full.state_dict())

    for layer in causal.temporal_transformer.layers:
        assert not layer.self_attn.causal
        assert all(attention.causal for attention in layer.cross_attns)
    for layer in full.temporal_transformer.layers:
        assert not layer.self_attn.causal
        assert all(
            isinstance(attention, StaticMemoryCrossAttention)
            for attention in layer.cross_attns
        )
        assert all(not attention.causal for attention in layer.cross_attns)
    for model in (causal, full):
        assert all(
            layer.self_attn.causal
            for layer in model.depth_transformer.layers
        )


def test_causal_condition_cannot_relay_future_audio_text():
    torch.manual_seed(23)
    causal = _make_teacher().eval()
    full = _make_teacher(
        GTemporalDepthModel3FutureGestureFullCondition
    ).eval()
    full.load_state_dict(causal.state_dict())
    codes, audio, text, speaker = _batch()
    target_times = torch.zeros(codes.shape[0], dtype=torch.long)
    temporal_codes = _masked_inputs(
        causal,
        codes,
        target_times,
        horizon_tokens=2,
    )

    causal_audio = truncate_condition_codes_after_targets(
        audio,
        target_times,
        condition_steps_per_gesture=1,
        null_token_id=-1,
    )
    causal_text = truncate_condition_codes_after_targets(
        text,
        target_times,
        condition_steps_per_gesture=1,
        null_token_id=-1,
    )
    causal_reference = _temporal_logits(
        causal,
        temporal_codes,
        causal_audio,
        causal_text,
        speaker,
    )

    changed_audio = audio.clone()
    changed_text = text.clone()
    changed_audio[:, :, 1:] = (changed_audio[:, :, 1:] + 1) % 17
    changed_text[:, :, 1:] = (changed_text[:, :, 1:] + 1) % 19
    changed_causal_audio = truncate_condition_codes_after_targets(
        changed_audio,
        target_times,
        condition_steps_per_gesture=1,
        null_token_id=-1,
    )
    changed_causal_text = truncate_condition_codes_after_targets(
        changed_text,
        target_times,
        condition_steps_per_gesture=1,
        null_token_id=-1,
    )
    causal_changed = _temporal_logits(
        causal,
        temporal_codes,
        changed_causal_audio,
        changed_causal_text,
        speaker,
    )
    torch.testing.assert_close(
        causal_reference[:, :, 0],
        causal_changed[:, :, 0],
    )

    full_reference = _temporal_logits(
        full,
        temporal_codes,
        audio,
        text,
        speaker,
    )
    full_changed = _temporal_logits(
        full,
        temporal_codes,
        changed_audio,
        changed_text,
        speaker,
    )
    assert not torch.allclose(
        full_reference[:, :, 0],
        full_changed[:, :, 0],
    )


def test_oracle_temporal_targets_match_full_forward():
    torch.manual_seed(25)
    teacher = _make_teacher().eval()
    codes, audio, text, speaker = _batch()
    target_times = torch.tensor([0, 1])
    temporal_codes = _masked_inputs(
        teacher,
        codes,
        target_times,
        horizon_tokens=2,
    )
    causal_audio = truncate_condition_codes_after_targets(
        audio,
        target_times,
        condition_steps_per_gesture=1,
        null_token_id=-1,
    )
    causal_text = truncate_condition_codes_after_targets(
        text,
        target_times,
        condition_steps_per_gesture=1,
        null_token_id=-1,
    )
    selected_out, selected_logits = (
        teacher.forward_oracle_temporal_targets(
            temporal_codes,
            causal_audio,
            causal_text,
            speaker,
            target_times,
        )
    )
    assert selected_out.shape == (2, 1, 16)
    assert selected_logits.shape == (2, 1, 1, 17)

    full_logits = teacher(
        codes,
        audio_codes=causal_audio,
        text_codes=causal_text,
        sum_condition=speaker,
        temporal_input_codes=temporal_codes,
    )
    batch_indices = torch.arange(codes.shape[0])
    torch.testing.assert_close(
        selected_logits[:, 0, 0],
        full_logits[batch_indices, 0, target_times],
    )


def test_oracle_kinematic_rollout_uses_predicted_prefixes():
    torch.manual_seed(26)
    teacher = _make_teacher().eval()
    codes, audio, text, speaker = _batch()
    codes = codes[:1].expand(2, -1, -1)
    audio = audio[:1].expand(2, -1, -1)
    text = text[:1].expand(2, -1, -1)
    speaker = speaker[:1].expand(2)
    target_times = torch.tensor([0, 1])
    temporal_codes = _masked_inputs(
        teacher,
        codes,
        target_times,
        horizon_tokens=2,
    )
    causal_audio = truncate_condition_codes_after_targets(
        audio,
        target_times,
        condition_steps_per_gesture=1,
        null_token_id=-1,
    )
    causal_text = truncate_condition_codes_after_targets(
        text,
        target_times,
        condition_steps_per_gesture=1,
        null_token_id=-1,
    )
    temporal_out, q0_logits = (
        teacher.forward_oracle_temporal_targets(
            temporal_codes,
            causal_audio,
            causal_text,
            speaker,
            target_times,
        )
    )
    q0_logits[..., teacher.pad_token_id] = float("-inf")
    q0_tokens = q0_logits[:, 0, 0].argmax(dim=-1)

    rollout = GestureLMGen(
        teacher,
        use_sampling=False,
        cfg_coef=1.0,
    )
    current_audio = torch.stack(
        [audio[index, :, target] for index, target in enumerate(target_times)]
    ).unsqueeze(-1)
    current_text = torch.stack(
        [text[index, :, target] for index, target in enumerate(target_times)]
    ).unsqueeze(-1)
    depth_audio, depth_text = rollout.process_conditions(
        current_audio,
        current_text,
    )
    depth_tokens = rollout.depformer_step(
        q0_tokens,
        temporal_out,
        depth_audio,
        depth_text,
        speaker[:, None],
        torch.zeros(2, teacher.n_q - 1, 1, dtype=torch.bool),
        None,
        rollout.bp_dist,
    )
    assert depth_tokens.shape == (2, teacher.n_q - 1)
    assert (depth_tokens >= 0).all()
    assert (depth_tokens < teacher.card).all()


def test_base_checkpoint_compatibility_and_backward():
    torch.manual_seed(24)
    base = GTemporalDepthModel3(**_model_kwargs())
    teacher = _make_teacher().train()
    teacher.load_state_dict(base.state_dict())
    codes, audio, text, speaker = _batch()
    target_times = torch.zeros(codes.shape[0], dtype=torch.long)
    temporal_codes = _masked_inputs(
        teacher,
        codes,
        target_times,
        horizon_tokens=2,
    )
    logits = teacher(
        codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
        temporal_input_codes=temporal_codes,
    )
    assert logits.shape == (2, 20, 4, 17)
    logits[:, :, 0].square().mean().backward()
    assert teacher.temporal_classifier.weight.grad is not None
    assert all(
        classifier.weight.grad is not None
        for classifier in teacher.depformer_classifier
    )


if __name__ == "__main__":
    test_target_and_guard_are_physically_absent()
    test_history_older_than_raw_miburi_context_is_absent()
    test_current_and_guard_cannot_change_target_logits_but_future_can()
    test_condition_truncation_is_per_sample_and_target_aligned()
    test_condition_variants_have_original_parameter_count_and_masks()
    test_causal_condition_cannot_relay_future_audio_text()
    test_oracle_temporal_targets_match_full_forward()
    test_oracle_kinematic_rollout_uses_predicted_prefixes()
    test_base_checkpoint_compatibility_and_backward()
    print(
        "Masked-frame future-gesture smoke tests passed "
        "(25-token past/400ms guard/target exclusion/condition isolation/"
        "oracle target selection/parameter count)."
    )
