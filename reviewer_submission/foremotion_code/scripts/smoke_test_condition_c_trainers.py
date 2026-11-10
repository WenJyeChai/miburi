"""CPU checks for Condition C trainer routing and stochastic-RVQ objectives.

These checks bypass trainer construction, datasets and pretrained downloads.
Run with pytest or ``python -m scripts.smoke_test_condition_c_trainers``.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from miburi.models.gesture_lm import GestureLMGen
from miburi.models.gesture_lm_offline import GestureLMOfflineGen
from scripts.smoke_test_gesture_lm_offline import _batch, _model_kwargs
from scripts.trainers import (
    UpperFaceLowerGTDM3ConditionCRegretRVQTrainer,
    UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer,
)
from scripts.trainers.uflgtdm3_shared_regret_rvq_trainer import (
    StochasticRVQTrainingMixin,
    UpperFaceLowerGTDM3SharedRegretRVQTrainer,
)
from scripts.trainers.uflgtdm3_shared_regret_trainer import (
    UpperFaceLowerGTDM3SharedRegretTrainer,
)
from scripts.trainers.uflgtdm3_trainer import UpperFaceLowerGTDM3Trainer


class _Tracker:
    def __init__(self):
        self.values = {}

    def update_meter(self, name, split, value):
        self.values[(split, name)] = float(value)


def _trainer(cls, card=16):
    trainer = cls.__new__(cls)
    trainer.args = SimpleNamespace(
        face_loss_weight=2.5,
        kinematic_rvq_start_epoch=5,
        kinematic_rvq_ramp_epochs=10,
        kinematic_rvq_sample_probability=0.15,
        kinematic_rvq_soft_target_weight=0.5,
        kinematic_rvq_topk=8,
        kinematic_rvq_temperature=0.5,
        kinematic_rvq_distance_chunk_size=4096,
    )
    trainer.tracker = _Tracker()
    trainer.modelout_ignore_index = card
    trainer.train_length = 10
    trainer._last_train_depth_input_codes = None
    return trainer


def _rvq_batch(card=16):
    codes, _, _, _ = _batch(card)
    valid = torch.ones_like(codes, dtype=torch.bool)
    valid[1, :, -1] = False
    targets = codes.masked_fill(~valid, card)
    inputs = targets.clone()
    # Input-only body dropout must survive in the depth prefix, while its
    # target is still supervised. Dataset padding suppresses both.
    inputs[0, 8:16, 0] = card
    stochastic = (codes + 1) % card
    indices = torch.stack([stochastic, (stochastic + 1) % card], dim=-1)
    probabilities = torch.empty_like(indices, dtype=torch.float)
    probabilities[..., 0] = 0.7
    probabilities[..., 1] = 0.3
    results = [
        SimpleNamespace(
            codes=stochastic[:, start:end],
            topk_indices=indices[:, start:end],
            topk_probabilities=probabilities[:, start:end],
        )
        for start, end in ((0, 8), (8, 16), (16, 20))
    ]
    return inputs, targets, valid, stochastic, {
        "rvq_results": results,
        "soft_weight": 0.5,
    }


def test_teacher_only_has_no_regret_training_or_validation_branch():
    cls = UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer
    assert issubclass(cls, StochasticRVQTrainingMixin)
    assert not issubclass(cls, UpperFaceLowerGTDM3SharedRegretTrainer)
    trainer = _trainer(cls)
    # Any accidental teacher/student comparison must fail, even though no
    # regret configuration or model object is supplied to this trainer.
    trainer._compute_regret = Mock(side_effect=AssertionError("unexpected KL"))
    trainer._forward_regret_teacher_view = Mock(
        side_effect=AssertionError("unexpected extra teacher forward")
    )
    trainer._compute_dense_future_gesture_regret = Mock(
        side_effect=AssertionError("unexpected future-motion teacher")
    )
    _, targets, valid, _, _ = _rvq_batch()
    logits = torch.randn(2, 20, 4, 17)
    assert trainer.get_additional_training_loss(
        logits, targets, valid, epoch=100, iteration=0
    ) is None
    trainer.record_validation_diagnostics(
        logits, targets, valid, epoch=100, iteration=0
    )
    for split in ("train", "val"):
        assert (split, "temporal_q0_ce") in trainer.tracker.values
    assert ("val", "kinematic_hard_ce") in trainer.tracker.values
    assert not any("regret" in name for _, name in trainer.tracker.values)
    trainer._compute_regret.assert_not_called()
    trainer._forward_regret_teacher_view.assert_not_called()
    trainer._compute_dense_future_gesture_regret.assert_not_called()
    assert trainer.get_generation_class() is GestureLMOfflineGen


def test_causal_student_routes_only_teacher_view_to_condition_c():
    trainer = _trainer(UpperFaceLowerGTDM3ConditionCRegretRVQTrainer)
    model = object()
    trainer._student_model = Mock(return_value=model)
    trainer.regret_include_depth_levels = True
    codes, audio, text, speaker = _batch()
    prefix = (codes + 1) % 16
    mask = torch.zeros_like(codes, dtype=torch.bool)
    expected = (torch.randn(2, 1, 4, 17), torch.randn(2, 19, 4, 17))
    with patch(
        "scripts.trainers.uflgtdm3_condition_c_trainer.forward_condition_c_teacher_view",
        return_value=expected,
    ) as forward:
        result = trainer._forward_regret_teacher_view(
            split="train", input_codes=codes, audio_codes=audio,
            text_codes=text, sum_condition=speaker,
            ca_depth_padding_mask=mask, depth_input_codes=prefix,
            sample_ids=["first", "second"],
        )
    assert result is expected
    forward.assert_called_once_with(
        model, input_codes=codes, audio_codes=audio, text_codes=text,
        sum_condition=speaker, ca_depth_padding_mask=mask,
        include_depth_levels=True, depth_input_codes=prefix,
    )
    assert trainer.get_generation_class() is GestureLMGen


def test_condition_c_rejects_incompatible_modes_before_any_model_loading():
    cases = [
        (UpperFaceLowerGTDM3ConditionCRegretRVQTrainer,
         {"dense_future_gesture_weight": 0.1}, "future speech only"),
        (UpperFaceLowerGTDM3ConditionCRegretRVQTrainer,
         {"regret_teacher_ckpt": "unexpected.safetensors"}, "regret_teacher_ckpt"),
        (UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer,
         {"regret_weight": 1.0}, "no student or KL"),
        (UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer,
         {"regret_initial_weight": 0.1}, "no student or KL"),
    ]
    with patch.object(
        UpperFaceLowerGTDM3Trainer, "__init__",
        side_effect=AssertionError("model loading should never be reached"),
    ):
        for cls, settings, message in cases:
            try:
                cls(SimpleNamespace(**settings))
            except ValueError as exc:
                assert message in str(exc)
            else:
                raise AssertionError(f"{cls.__name__} accepted {settings}")


def test_teacher_model_factory_uses_condition_c_with_all_codecs():
    trainer = _trainer(UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer)
    trainer.args = SimpleNamespace(
        gestureformer_heads=2, gestureformer_layers=4,
        gestureformer_depformer_heads=1, gestureformer_depformer_layers=2,
        num_temp_classifiers=1, param_dtype="float32", vad_guidance=True,
        vad_use_face_logits=True, textaudio_emb_freeze=False,
    )
    trainer.global_rank = 0
    trainer.local_rank = 0
    trainer.codec_difference = 1
    codec_layers = _model_kwargs()["gesture_codec_layers"]
    trainer.upper_gesture_codec, trainer.lower_gesture_codec, trainer.face_gesture_codec = [
        SimpleNamespace(quantizer=SimpleNamespace(vq=SimpleNamespace(layers=layers)))
        for layers in (codec_layers[:8], codec_layers[8:16], codec_layers[16:])
    ]
    pretrained = SimpleNamespace(
        text_emb=torch.nn.Embedding(19, 16),
        emb=torch.nn.ModuleList([torch.nn.Embedding(17, 16) for _ in range(8)]),
    )
    expected_model = object()
    module = "scripts.trainers.uflgtdm3_offline_trainer"
    with patch(f"{module}.loaders.CheckpointInfo.from_hf_repo") as checkpoint, \
         patch(f"{module}.loaders.get_gesturelm_kwargs", return_value={"n_q": 20}), \
         patch.object(type(trainer), "model_class", return_value=expected_model) as factory:
        checkpoint.return_value.get_moshi.return_value = pretrained
        assert trainer.get_model(trainer.args) is expected_model
    kwargs = factory.call_args.kwargs
    assert kwargs["num_temp_classifiers"] == 1
    assert kwargs["n_q"] == 20
    assert kwargs["dtype"] is torch.float32
    assert len(kwargs["gesture_codec_layers"]) == 20
    assert all(not p.requires_grad for p in kwargs["gesture_codec_layers"].parameters())
    assert all(p.requires_grad for p in codec_layers.parameters())


def test_rvq_prefixes_targets_and_masks_match_all_three_trainers():
    torch.manual_seed(41)
    inputs, targets, valid, stochastic, context = _rvq_batch()
    original_inputs, original_targets = inputs.clone(), targets.clone()
    outputs = []
    for cls in (
        UpperFaceLowerGTDM3SharedRegretRVQTrainer,
        UpperFaceLowerGTDM3ConditionCRegretRVQTrainer,
        UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer,
    ):
        trainer = _trainer(cls)
        prefix, training_targets, mask, loss_context = (
            trainer.prepare_kinematic_training_inputs(
                inputs, targets, valid,
                epoch=20, iteration=0, encoding_context=context,
            )
        )
        assert torch.equal(prefix[inputs == 16], inputs[inputs == 16])
        assert torch.equal(prefix[inputs != 16], stochastic[inputs != 16])
        assert torch.equal(training_targets[:, 0], targets[:, 0])
        assert torch.equal(training_targets[:, 1:][valid[:, 1:]],
                           stochastic[:, 1:][valid[:, 1:]])
        assert (training_targets[~valid] == 16).all()
        assert (training_targets[0, 8:16, 0] != 16).all()
        assert torch.equal(mask, valid)
        assert loss_context["soft_indices"].shape == (2, 20, 4, 2)
        outputs.append((prefix, training_targets, loss_context))
    for output in outputs[1:]:
        for key in (0, 1):
            torch.testing.assert_close(output[key], outputs[0][key])
        for key in ("soft_indices", "soft_probabilities"):
            torch.testing.assert_close(output[2][key], outputs[0][2][key])
    torch.testing.assert_close(inputs, original_inputs)
    torch.testing.assert_close(targets, original_targets)


def test_teacher_only_rvq_ce_preserves_q0_face_weight_and_padding_gradients():
    torch.manual_seed(42)
    inputs, targets, valid, _, context = _rvq_batch()
    raw_logits = torch.randn(2, 20, 4, 17)
    objectives, gradients = [], []
    for cls in (
        UpperFaceLowerGTDM3SharedRegretRVQTrainer,
        UpperFaceLowerGTDM3ConditionCRegretRVQTrainer,
        UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer,
    ):
        trainer = _trainer(cls)
        _, training_targets, mask, loss_context = (
            trainer.prepare_kinematic_training_inputs(
                inputs, targets, valid,
                epoch=20, iteration=0, encoding_context=context,
            )
        )
        logits = raw_logits.clone().requires_grad_()
        objective, *_ = trainer.compute_training_ce_objective(
            logits, training_targets, mask, loss_context=loss_context,
        )
        objective.backward()
        assert torch.isfinite(objective)
        assert (logits.grad[~valid] == 0).all()
        objectives.append(objective.detach())
        gradients.append(logits.grad)

        # Check the actual target distribution, beyond agreement between
        # classes: q0 uses hard labels, depth uses hard/soft mixtures, and
        # face heads carry the established objective weighting.
        probabilities = logits.detach().softmax(-1)
        for k in (0, 3, 11, 18):
            expected_target = F.one_hot(training_targets[:, k], 17).float()
            if k > 0:
                soft = torch.zeros_like(expected_target).scatter_add_(
                    -1, loss_context["soft_indices"][:, k],
                    loss_context["soft_probabilities"][:, k],
                )
                expected_target = 0.5 * expected_target + 0.5 * soft
            scale = (2.5 if k >= 16 else 1.0) / 20
            expected_gradient = (
                (probabilities[:, k] - expected_target)
                * valid[:, k, :, None]
                * scale / valid[:, k].sum()
            )
            torch.testing.assert_close(logits.grad[:, k], expected_gradient)
    for objective, gradient in zip(objectives[1:], gradients[1:]):
        torch.testing.assert_close(objective, objectives[0])
        torch.testing.assert_close(gradient, gradients[0])


def test_rvq_warmup_delegates_to_base_codec_encoding_without_extra_passes():
    trainer = _trainer(UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer)
    inputs = (torch.randn(2, 4, 3),) * 3
    expected = tuple(torch.full((2, k, 4), i) for i, k in enumerate((8, 8, 4)))
    codecs = [SimpleNamespace(encode=Mock(return_value=codes)) for codes in expected]
    trainer.upper_gesture_codec, trainer.lower_gesture_codec, trainer.face_gesture_codec = codecs
    with patch(
        "scripts.trainers.uflgtdm3_shared_regret_rvq_trainer.encode_stochastic_rvq",
        side_effect=AssertionError("RVQ encoder called during zero-weight warmup"),
    ):
        codes, context = trainer.encode_training_gesture_codes(
            inputs, epoch=0, iteration=0,
        )
    for actual, reference, codec, motion in zip(codes, expected, codecs, inputs):
        assert actual is reference
        codec.encode.assert_called_once_with(motion)
    assert context["soft_weight"] == 0.0
    assert context["rvq_results"] is None
    assert trainer._rvq_regularization_scale(5, 0) == 0.0
    assert trainer._rvq_regularization_scale(10, 0) == 0.5
    assert trainer._rvq_regularization_scale(15, 0) == 1.0


def test_teacher_only_supervised_forward_trains_temporal_and_depth_once():
    torch.manual_seed(43)
    trainer = _trainer(UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer)
    model = trainer.model_class(**_model_kwargs()).train()
    trainer.model = model
    codes, audio, text, speaker = _batch()
    mask = torch.ones_like(codes, dtype=torch.bool)
    depth_inputs = (codes + 1) % model.card
    temporal_calls, depth_calls = [], []
    temporal_hook = model.temporal_transformer.register_forward_hook(
        lambda *_: temporal_calls.append(1)
    )
    depth_hook = model.depth_transformer.register_forward_hook(
        lambda *_: depth_calls.append(1)
    )
    try:
        logits = model(
            codes, audio_codes=audio, text_codes=text,
            sum_condition=speaker, depth_input_codes=depth_inputs,
        )
        loss, *_ = trainer.compute_training_ce_objective(
            logits, codes, mask, loss_context={"soft_weight": 0.0},
        )
        assert trainer.get_additional_training_loss(
            logits, codes, mask, epoch=20, iteration=0,
        ) is None
        loss.backward()
    finally:
        temporal_hook.remove()
        depth_hook.remove()
    assert temporal_calls == [1]
    assert depth_calls == [1]
    for parameter in (
        model.temporal_classifier.weight,
        model.temp_condproj[0].weight,
        model.temp_condproj[1].weight,
        *(classifier.weight for classifier in model.depformer_classifier),
    ):
        assert parameter.requires_grad
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    tests = [(name, value) for name, value in globals().copy().items()
             if name.startswith("test_") and callable(value)]
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"Condition C trainer smoke tests passed ({len(tests)} tests).")
