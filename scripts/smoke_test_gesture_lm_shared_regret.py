"""Focused CPU checks for shared-weight GlobalRegret distillation."""

import torch

from miburi.models.gesture_lm import GTemporalDepthModel3
from miburi.models.gesture_lm_shared_regret import (
    dense_regret_kl,
    forward_teacher_view,
    relaxed_temporal_cross_attention,
)
from scripts.smoke_test_gesture_lm_offline import _batch, _model_kwargs


def _make_model():
    torch.manual_seed(7)
    return GTemporalDepthModel3(**_model_kwargs())


def test_context_manager_toggles_and_restores_causal():
    model = _make_model()
    cross_attns = [
        cross_attn
        for layer in model.temporal_transformer.layers
        for cross_attn in layer.cross_attns
    ]
    assert len(cross_attns) > 0
    assert all(cross_attn.causal for cross_attn in cross_attns)

    with relaxed_temporal_cross_attention(model.temporal_transformer):
        assert all(not cross_attn.causal for cross_attn in cross_attns)

    assert all(cross_attn.causal for cross_attn in cross_attns)

    try:
        with relaxed_temporal_cross_attention(model.temporal_transformer):
            assert all(not cross_attn.causal for cross_attn in cross_attns)
            raise RuntimeError("simulated failure mid-teacher-pass")
    except RuntimeError:
        pass
    assert all(cross_attn.causal for cross_attn in cross_attns), (
        "causal flags must be restored even when the teacher pass raises"
    )

    for layer in model.depth_transformer.layers:
        assert layer.self_attn.causal, (
            "shared-regret must never touch the depth transformer's mask"
        )


def test_teacher_view_is_gradient_free_and_differs_from_student():
    model = _make_model().train()
    codes, audio, text, speaker = _batch()

    teacher_temp_logits, teacher_depth_logits = forward_teacher_view(
        model,
        input_codes=codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
        include_depth_levels=True,
    )
    assert not teacher_temp_logits.requires_grad
    assert not teacher_depth_logits.requires_grad
    assert teacher_temp_logits.grad_fn is None
    assert teacher_depth_logits.grad_fn is None

    assert all(cross_attn.causal for cross_attn in [
        ca for layer in model.temporal_transformer.layers
        for ca in layer.cross_attns
    ]), "student's causal mask must be restored after the teacher call"

    student_logits = model(
        codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
    )
    student_temp_logits = student_logits[:, :1]
    # With random weights and a short (4-token) sequence, a bidirectional
    # teacher view should not coincide with the causal student's q0 logits.
    assert not torch.allclose(
        teacher_temp_logits, student_temp_logits, atol=1e-6,
    )


def test_dense_regret_kl_direction_stopgrad_and_missing_codebooks():
    logits_a = torch.randn(2, 3, 4, 5, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    valid_mask = torch.ones(2, 3, 4, dtype=torch.bool)

    identical = dense_regret_kl(
        logits_a.detach(), logits_a.detach(), valid_mask,
    )
    torch.testing.assert_close(
        identical, torch.zeros_like(identical), atol=1e-6, rtol=0,
    )

    student = torch.randn(2, 3, 4, 5, requires_grad=True)
    teacher = torch.randn(2, 3, 4, 5, requires_grad=True)
    loss = dense_regret_kl(student, teacher, valid_mask)
    assert loss.item() > 0
    loss.backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None, "teacher must never receive gradient"

    # Codebook index 1 has no valid positions at all; it must be excluded
    # from the K-average rather than pulling it toward zero.
    partial_mask = valid_mask.clone()
    partial_mask[:, 1, :] = False
    with_gap = dense_regret_kl(logits_a, logits_b, partial_mask)
    full_mask_loss = dense_regret_kl(logits_a, logits_b, valid_mask)
    assert with_gap.item() != full_mask_loss.item()
    assert torch.isfinite(with_gap)


def test_regret_backward_touches_temporal_and_depth_student_paths():
    model = _make_model().train()
    codes, audio, text, speaker = _batch()

    teacher_temp_logits, teacher_depth_logits = forward_teacher_view(
        model,
        input_codes=codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
        include_depth_levels=True,
    )
    teacher_logits = torch.cat(
        [teacher_temp_logits, teacher_depth_logits], dim=1,
    )

    student_logits = model(
        codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
    )
    real_vocab = model.card
    valid_mask = torch.ones(
        student_logits.shape[0],
        student_logits.shape[1],
        student_logits.shape[2],
        dtype=torch.bool,
    )
    loss = dense_regret_kl(
        student_logits[..., :real_vocab],
        teacher_logits[..., :real_vocab],
        valid_mask,
    )
    loss.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.temporal_transformer.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.depth_transformer.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.depformer_classifier.parameters()
    )


def test_teacher_view_depth_input_codes_override_is_isolated_to_depth():
    """A distinct depth_input_codes only changes the depth branch.

    Regression test for the SharedRegretRVQ composition: the teacher's
    depth branch must be able to see a different prefix (e.g. a stochastic
    RVQ sample) than the temporal branch, without that override leaking
    into (or being ignored by) the temporal logits.
    """

    model = _make_model()
    codes, audio, text, speaker = _batch()

    default_temp_logits, default_depth_logits = forward_teacher_view(
        model,
        input_codes=codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
        include_depth_levels=True,
    )
    explicit_same_temp_logits, explicit_same_depth_logits = forward_teacher_view(
        model,
        input_codes=codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
        include_depth_levels=True,
        depth_input_codes=codes,
    )
    torch.testing.assert_close(
        default_temp_logits, explicit_same_temp_logits,
    )
    torch.testing.assert_close(
        default_depth_logits, explicit_same_depth_logits,
    )

    card = model.card
    perturbed_codes = torch.remainder(codes + 1, card)
    perturbed_temp_logits, perturbed_depth_logits = forward_teacher_view(
        model,
        input_codes=codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
        include_depth_levels=True,
        depth_input_codes=perturbed_codes,
    )
    # Temporal branch never sees depth_input_codes at all.
    torch.testing.assert_close(default_temp_logits, perturbed_temp_logits)
    # Depth branch's input prefix changed, so its logits must differ.
    assert not torch.allclose(
        default_depth_logits, perturbed_depth_logits, atol=1e-6,
    )


def test_teacher_view_rejects_mismatched_depth_input_codes_shape():
    model = _make_model()
    codes, audio, text, speaker = _batch()
    wrong_shape = codes[:, :, :-1]
    try:
        forward_teacher_view(
            model,
            input_codes=codes,
            audio_codes=audio,
            text_codes=text,
            sum_condition=speaker,
            include_depth_levels=True,
            depth_input_codes=wrong_shape,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for mismatched depth_input_codes shape"
        )


if __name__ == "__main__":
    test_context_manager_toggles_and_restores_causal()
    test_teacher_view_is_gradient_free_and_differs_from_student()
    test_dense_regret_kl_direction_stopgrad_and_missing_codebooks()
    test_regret_backward_touches_temporal_and_depth_student_paths()
    test_teacher_view_depth_input_codes_override_is_isolated_to_depth()
    test_teacher_view_rejects_mismatched_depth_input_codes_shape()
    print(
        "Shared-regret smoke tests passed (mask toggle/restore, "
        "gradient-free teacher view, dense KL direction/stop-gradient, "
        "temporal+depth student gradients, depth_input_codes override "
        "isolation, shape validation)."
    )
