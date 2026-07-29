"""Focused CPU checks for future-aware q0 forward-KL regret."""

import torch

from miburi.models.gesture_lm import GTemporalDepthModel3
from miburi.models.gesture_lm_future_gesture import (
    GTemporalDepthModel3FutureGestureFullCondition,
    build_masked_future_gesture_inputs,
)
from miburi.models.gesture_lm_regret import (
    forward_q0_regret_kl,
    is_temporal_q0_state_key,
    retain_temporal_q0_only,
)
from scripts.smoke_test_gesture_lm_offline import (
    _batch,
    _model_kwargs,
)


def test_forward_kl_direction_and_stop_gradient():
    student_logits = torch.tensor(
        [[1.0, -0.5, 0.25]],
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [[-0.25, 1.25, 0.5]],
        requires_grad=True,
    )
    loss = forward_q0_regret_kl(
        student_logits,
        teacher_logits,
        temperature=1.0,
    )
    assert loss.item() > 0
    loss.backward()
    assert student_logits.grad is not None
    assert student_logits.grad.abs().sum() > 0
    assert teacher_logits.grad is None

    identical = forward_q0_regret_kl(
        teacher_logits.detach(),
        teacher_logits.detach(),
    )
    torch.testing.assert_close(
        identical,
        torch.zeros_like(identical),
        atol=1e-6,
        rtol=0,
    )


def test_temporal_only_teacher_retains_oracle_forward():
    torch.manual_seed(31)
    teacher = GTemporalDepthModel3FutureGestureFullCondition(
        **_model_kwargs()
    ).eval()
    full_parameter_count = sum(
        parameter.numel()
        for parameter in teacher.parameters()
    )
    retain_temporal_q0_only(teacher)
    temporal_parameter_count = sum(
        parameter.numel()
        for parameter in teacher.parameters()
    )
    assert temporal_parameter_count < full_parameter_count
    assert all(
        is_temporal_q0_state_key(name)
        for name in teacher.state_dict()
    )

    codes, audio, text, speaker = _batch()
    target_times = torch.tensor([0, 1])
    temporal_codes = build_masked_future_gesture_inputs(
        codes,
        codes,
        target_times,
        horizon_tokens=2,
        past_context_tokens=25,
        mask_token_id=teacher.pad_token_id,
    )
    temporal_out, logits = teacher.forward_oracle_temporal_targets(
        temporal_codes,
        audio,
        text,
        speaker,
        target_times,
    )
    assert temporal_out.shape == (2, 1, 16)
    assert logits.shape == (2, 1, 1, 17)


def test_regret_updates_only_causal_student():
    torch.manual_seed(32)
    student = GTemporalDepthModel3(**_model_kwargs()).train()
    teacher = GTemporalDepthModel3FutureGestureFullCondition(
        **_model_kwargs()
    )
    teacher.load_state_dict(student.state_dict())
    retain_temporal_q0_only(teacher)
    teacher.requires_grad_(False)
    teacher.eval()

    codes, audio, text, speaker = _batch()
    target_times = torch.tensor([0, 1])
    temporal_codes = build_masked_future_gesture_inputs(
        codes,
        codes,
        target_times,
        horizon_tokens=2,
        past_context_tokens=25,
        mask_token_id=teacher.pad_token_id,
    )
    with torch.no_grad():
        _, teacher_logits = teacher.forward_oracle_temporal_targets(
            temporal_codes,
            audio,
            text,
            speaker,
            target_times,
        )

    student_logits = student(
        codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
    )
    batch_indices = torch.arange(codes.shape[0])
    selected_student = student_logits[
        batch_indices,
        0,
        target_times,
        :student.card,
    ]
    selected_teacher = teacher_logits[
        :,
        0,
        0,
        :teacher.card,
    ]
    regret = forward_q0_regret_kl(
        selected_student,
        selected_teacher,
    ) / student.n_q
    regret.backward()

    assert student.temporal_classifier.weight.grad is not None
    assert (
        student.temporal_classifier.weight.grad.abs().sum()
        > 0
    )
    assert any(
        parameter.grad is not None
        and parameter.grad.abs().sum() > 0
        for parameter in student.temporal_transformer.parameters()
    )
    assert all(
        parameter.grad is None
        or parameter.grad.abs().sum() == 0
        for parameter in student.depth_transformer.parameters()
    )
    assert all(
        parameter.grad is None
        or parameter.grad.abs().sum() == 0
        for parameter in student.depformer_classifier.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in teacher.parameters()
    )


if __name__ == "__main__":
    test_forward_kl_direction_and_stop_gradient()
    test_temporal_only_teacher_retains_oracle_forward()
    test_regret_updates_only_causal_student()
    print(
        "Regret smoke tests passed (forward KL direction/stop-gradient/"
        "temporal-only teacher/temporal-only student gradients)."
    )
