"""Focused checks for privileged future-gesture MIBURI teachers."""

import torch

from miburi.models.gesture_lm import GTemporalDepthModel3
from miburi.models.gesture_lm_future_gesture import (
    GTemporalDepthModel3FutureGesture,
    GTemporalDepthModel3FutureGestureFullCondition,
)
from miburi.models.gesture_lm_offline import StaticMemoryCrossAttention
from scripts.smoke_test_gesture_lm_offline import (
    _batch,
    _model_kwargs,
    _temporal_inputs,
)


def _make_teacher(model_class=GTemporalDepthModel3FutureGesture):
    return model_class(
        **_model_kwargs(),
        future_gesture_layers=2,
        future_gesture_heads=4,
        future_gesture_context=4,
        future_gesture_gate_init=0.2,
    )


def _temporal_logits(model, codes, audio, text, speaker):
    temporal_input = _temporal_inputs(model, codes)
    audio_condition, text_condition = model.process_conditions(audio, text)
    temporal_speaker = speaker[:, None].expand(-1, codes.shape[-1])
    _, logits = model.forward_temporal(
        temporal_input,
        audio_condition.squeeze(1),
        text_condition.squeeze(1),
        temporal_speaker,
        temporal_target_codes=codes,
    )
    return logits


def test_strict_future_context_excludes_current_target():
    torch.manual_seed(10)
    model = _make_teacher().eval()
    codes, _, _, _ = _batch()
    target_time = 1

    reference = model.encode_strict_future_gesture(codes)
    changed_current = codes.clone()
    changed_current[:, :, target_time] = (
        changed_current[:, :, target_time] + 1
    ) % model.card
    current_context = model.encode_strict_future_gesture(changed_current)
    torch.testing.assert_close(
        reference[:, target_time],
        current_context[:, target_time],
    )

    changed_future = codes.clone()
    changed_future[:, :, target_time + 1] = (
        changed_future[:, :, target_time + 1] + 1
    ) % model.card
    future_context = model.encode_strict_future_gesture(changed_future)
    assert not torch.allclose(
        reference[:, target_time],
        future_context[:, target_time],
    )
    assert torch.count_nonzero(reference[:, -1]) == 0


def test_temporal_q0_has_no_same_target_leakage():
    torch.manual_seed(11)
    model = _make_teacher().eval()
    codes, audio, text, speaker = _batch()
    target_time = 1
    reference = _temporal_logits(
        model,
        codes,
        audio,
        text,
        speaker,
    )

    changed_current = codes.clone()
    changed_current[:, :, target_time] = (
        changed_current[:, :, target_time] + 1
    ) % model.card
    current_logits = _temporal_logits(
        model,
        changed_current,
        audio,
        text,
        speaker,
    )
    torch.testing.assert_close(
        reference[:, :, target_time],
        current_logits[:, :, target_time],
    )

    changed_future = codes.clone()
    changed_future[:, :, target_time + 1] = (
        changed_future[:, :, target_time + 1] + 1
    ) % model.card
    future_logits = _temporal_logits(
        model,
        changed_future,
        audio,
        text,
        speaker,
    )
    assert not torch.allclose(
        reference[:, :, target_time],
        future_logits[:, :, target_time],
    )


def test_condition_variants_differ_only_by_temporal_condition_mask():
    torch.manual_seed(12)
    causal = _make_teacher(GTemporalDepthModel3FutureGesture)
    full = _make_teacher(
        GTemporalDepthModel3FutureGestureFullCondition
    )
    assert sum(p.numel() for p in causal.parameters()) == sum(
        p.numel() for p in full.parameters()
    )

    for layer in causal.temporal_transformer.layers:
        assert layer.self_attn.causal
        assert all(attention.causal for attention in layer.cross_attns)
    for layer in full.temporal_transformer.layers:
        assert layer.self_attn.causal
        assert all(
            isinstance(attention, StaticMemoryCrossAttention)
            for attention in layer.cross_attns
        )
        assert all(not attention.causal for attention in layer.cross_attns)
    for model in (causal, full):
        assert all(
            layer.self_attn.causal
            for layer in model.future_gesture_transformer.layers
        )
        assert all(
            layer.self_attn.causal
            for layer in model.depth_transformer.layers
        )


def test_causal_condition_hides_future_audio_and_text():
    torch.manual_seed(13)
    base = _make_teacher().eval()
    full = _make_teacher(
        GTemporalDepthModel3FutureGestureFullCondition
    ).eval()
    full.load_state_dict(base.state_dict())
    codes, audio, text, speaker = _batch()

    base_reference = _temporal_logits(
        base,
        codes,
        audio,
        text,
        speaker,
    )
    full_reference = _temporal_logits(
        full,
        codes,
        audio,
        text,
        speaker,
    )
    changed_audio = audio.clone()
    changed_audio[:, :, -1] = (changed_audio[:, :, -1] + 1) % 17
    changed_text = text.clone()
    changed_text[:, :, -1] = (changed_text[:, :, -1] + 1) % 19

    base_changed = _temporal_logits(
        base,
        codes,
        changed_audio,
        changed_text,
        speaker,
    )
    full_changed = _temporal_logits(
        full,
        codes,
        changed_audio,
        changed_text,
        speaker,
    )
    torch.testing.assert_close(
        base_reference[:, :, 0],
        base_changed[:, :, 0],
    )
    assert not torch.allclose(
        full_reference[:, :, 0],
        full_changed[:, :, 0],
    )


def test_base_checkpoint_warm_start_and_backward():
    torch.manual_seed(14)
    base = GTemporalDepthModel3(**_model_kwargs())
    teacher = _make_teacher().train()
    teacher.load_state_dict(base.state_dict())
    for key, value in base.state_dict().items():
        torch.testing.assert_close(teacher.state_dict()[key], value)

    codes, audio, text, speaker = _batch()
    logits = teacher(
        codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
    )
    assert logits.shape == (2, 20, 4, 17)
    logits.square().mean().backward()
    assert teacher.future_gesture_gate.grad is not None
    assert teacher.future_gesture_fusion.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in teacher.future_gesture_transformer.parameters()
    )


if __name__ == "__main__":
    test_strict_future_context_excludes_current_target()
    test_temporal_q0_has_no_same_target_leakage()
    test_condition_variants_differ_only_by_temporal_condition_mask()
    test_causal_condition_hides_future_audio_and_text()
    test_base_checkpoint_warm_start_and_backward()
    print(
        "Future-gesture teacher smoke tests passed "
        "(strict target exclusion/masks/warm-start/backward)."
    )
