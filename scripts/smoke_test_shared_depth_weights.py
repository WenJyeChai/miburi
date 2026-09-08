"""CPU checks for opt-in shared depth cores through the real trainer factories.

Only pretrained resources and model dimensions are substituted. Models, their
factories, Condition C teacher views, parser and cross-entropy are real; no
datasets, downloads, checkpoints or training loop are needed.

Run with ``python -B -m scripts.smoke_test_shared_depth_weights``.
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from miburi.models import loaders
from miburi.models.gesture_lm import GTemporalDepthModel3
from miburi.models.gesture_lm_condition_c import GTemporalDepthModel3ConditionC
from miburi.models.gesture_lm_offline import GTemporalDepthModel3Offline
from miburi.utils.compile import no_compile
from scripts.smoke_test_gesture_lm_offline import _batch, _model_kwargs
from scripts.trainers.uflgtdm3_condition_c_trainer import (
    UpperFaceLowerGTDM3ConditionCRegretRVQTrainer,
    UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer,
)
from scripts.trainers.uflgtdm3_offline_trainer import UpperFaceLowerGTDM3OfflineTrainer
from scripts.trainers.uflgtdm3_trainer import UpperFaceLowerGTDM3Trainer
from scripts.trainers.utils.config import parse_args


_MISSING = object()
_ROOT = Path(__file__).resolve().parents[1]
_FLAG = "gestureformer_depformer_weights_per_step"
_FACTORIES = (
    (UpperFaceLowerGTDM3Trainer, GTemporalDepthModel3),
    (UpperFaceLowerGTDM3OfflineTrainer, GTemporalDepthModel3Offline),
    (UpperFaceLowerGTDM3ConditionCRegretRVQTrainer, GTemporalDepthModel3),
    (UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer, GTemporalDepthModel3ConditionC),
)


def _factory(trainer_class, weights_per_step=_MISSING):
    """Exercise get_model while retaining the actual construction and weights."""
    kwargs = _model_kwargs()
    codec_layers = kwargs["gesture_codec_layers"]
    pretrained = SimpleNamespace(
        text_emb=nn.Embedding.from_pretrained(kwargs["text_procemb"]),
        emb=nn.ModuleList([
            nn.Embedding.from_pretrained(embedding)
            for embedding in kwargs["audio_procemb"]
        ]),
    )
    args = SimpleNamespace(
        gestureformer_heads=kwargs.pop("num_heads"),
        gestureformer_layers=kwargs.pop("num_layers"),
        gestureformer_depformer_heads=kwargs.pop("depformer_heads"),
        gestureformer_depformer_layers=kwargs.pop("depformer_layers"),
        num_temp_classifiers=kwargs.pop("num_temp_classifiers"),
        param_dtype="float32", ddp=False, vad_guidance=False,
        vad_use_face_logits=False, textaudio_emb_freeze=False,
    )
    if weights_per_step is not _MISSING:
        setattr(args, _FLAG, weights_per_step)
    trainer = trainer_class.__new__(trainer_class)
    trainer.args = args
    trainer.global_rank = trainer.local_rank = 0
    trainer.codec_difference = kwargs.pop("query2mem_scale")
    for key in ("text_procemb", "audio_procemb", "gesture_codec_layers",
                "body_parts", "bp_dist"):
        kwargs.pop(key)
    for name, layers in zip(
        ("upper_gesture_codec", "lower_gesture_codec", "face_gesture_codec"),
        (codec_layers[:8], codec_layers[8:16], codec_layers[16:]),
    ):
        setattr(trainer, name, SimpleNamespace(
            quantizer=SimpleNamespace(vq=SimpleNamespace(layers=layers)),
        ))
    # Returning the same dictionary catches accidental mutation of loader
    # defaults, which would otherwise contaminate subsequent model factories.
    original_kwargs = deepcopy(kwargs)
    production_defaults = deepcopy(loaders.get_gesturelm_kwargs())
    with patch.object(loaders.CheckpointInfo, "from_hf_repo") as checkpoint, \
         patch.object(loaders, "get_gesturelm_kwargs", return_value=kwargs):
        checkpoint.return_value.get_moshi.return_value = pretrained
        model = trainer.get_model(args)
        checkpoint.assert_called_once_with(loaders.DEFAULT_REPO)
        checkpoint.return_value.get_moshi.assert_called_once_with()
    assert kwargs == original_kwargs
    assert loaders.get_gesturelm_kwargs() == production_defaults
    assert all(parameter.requires_grad for parameter in codec_layers.parameters())
    assert all(not parameter.requires_grad
               for parameter in model.gesture_codec_layers.parameters())
    trainer.model = model
    return trainer, model


def _forward(model, batch):
    codes, audio, text, speaker = batch
    with no_compile():
        return model(codes, audio_codes=audio, text_codes=text,
                     sum_condition=speaker)


def _assert_nonzero_finite_grad(module):
    parameters = [parameter for parameter in module.parameters()
                  if parameter.requires_grad]
    assert parameters
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert sum(parameter.grad.abs().sum().item()
               for parameter in parameters) > 0


def _parse(config_path, *arguments):
    with patch("sys.argv", ["shared-depth-smoke", "-c", str(config_path),
                            *arguments]):
        return parse_args()


def test_configs_change_only_name_and_opt_in_and_cli_can_restore_baseline():
    for run in ("foremotion", "teacher"):
        baseline_path = _ROOT / "configs" / f"gtdm3_{run}_c_rvq_beatx_scott.yaml"
        shared_path = _ROOT / "configs" / f"gtdm3_{run}_c_shareddepth_rvq_beatx_scott.yaml"
        baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
        shared = yaml.safe_load(shared_path.read_text(encoding="utf-8"))
        assert _FLAG not in baseline
        assert shared.pop(_FLAG) is False
        assert shared.pop("project_name") != baseline.pop("project_name")
        assert shared == baseline
        assert getattr(_parse(baseline_path), _FLAG) is True
        assert getattr(_parse(shared_path), _FLAG) is False
        assert getattr(_parse(shared_path, f"--{_FLAG}", "True"), _FLAG) is True


def test_all_real_factories_route_shared_core_without_mutating_defaults():
    for trainer_class, model_class in _FACTORIES:
        _, model = _factory(trainer_class, False)
        assert type(model) is model_class
        assert model.depformer_multi_linear is True
        for layer in model.depth_transformer.layers:
            assert layer.weights_per_step == 0
            assert len(layer.self_attn.in_projs) == 1
            assert len(layer.self_attn.out_projs) == 1
            assert not isinstance(layer.gating, nn.ModuleList)
            assert not isinstance(layer.ca_gating, nn.ModuleList)
            for attention in layer.cross_attns:
                assert attention.weights_per_step == 0
                assert all(len(getattr(attention, name)) == 1
                           for name in ("in_projs", "key_projs", "val_projs", "out_projs"))
        # A subsequent default construction must return to the original core.
        _, subsequent = _factory(trainer_class)
        assert all(layer.weights_per_step == 19
                   for layer in subsequent.depth_transformer.layers)


def test_missing_flag_and_explicit_true_keep_existing_checkpoint_layout():
    torch.manual_seed(61)
    batch = _batch()
    for trainer_class, _ in _FACTORIES:
        _, legacy = _factory(trainer_class)
        _, explicit = _factory(trainer_class, True)
        result = explicit.load_state_dict(legacy.state_dict(), strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert all(layer.weights_per_step == 19
                   for layer in explicit.depth_transformer.layers)
        with torch.no_grad():
            torch.testing.assert_close(
                _forward(legacy.eval(), batch), _forward(explicit.eval(), batch),
            )


def test_shared_core_preserves_distinct_codebook_io_and_temporal_layout():
    for trainer_class in (UpperFaceLowerGTDM3ConditionCRegretRVQTrainer,
                          UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer):
        _, baseline = _factory(trainer_class, True)
        _, shared = _factory(trainer_class, False)
        assert shared.n_q == 20 and shared.num_temp_classifiers == 1
        assert shared.temporal_classifier.out_features == shared.card + 1
        for name in ("depformer_in", "depformer_gemb", "depformer_gproj",
                     "spk_depemb", "depformer_classifier"):
            modules = getattr(shared, name)
            assert len(modules) == 19
            weights = [module.weight for module in modules]
            assert len({weight.data_ptr() for weight in weights}) == 19
        non_core_layout = lambda model: {
            name: tuple(value.shape) for name, value in model.state_dict().items()
            if not name.startswith("depth_transformer.")
        }
        assert non_core_layout(shared) == non_core_layout(baseline)
        assert sum(parameter.numel() for parameter in shared.depth_transformer.parameters()) < sum(
            parameter.numel() for parameter in baseline.depth_transformer.parameters()
        )


def test_depth_ce_backpropagates_through_shared_core_for_student_and_teacher():
    torch.manual_seed(62)
    for trainer_class in (UpperFaceLowerGTDM3ConditionCRegretRVQTrainer,
                          UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer):
        _, model = _factory(trainer_class, False)
        batch = _batch()
        logits = _forward(model.train(), batch)
        assert logits.shape == (2, 20, 4, 17)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, 17),
                               batch[0][:, 1:].reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        # A depth-only objective must reach temporal hidden states, while the
        # separate q0 classifier has no gradient from those 19 predictions.
        q0_grad = model.temporal_classifier.weight.grad
        assert q0_grad is None or torch.count_nonzero(q0_grad).item() == 0
        _assert_nonzero_finite_grad(model.temporal_transformer.layers[0].self_attn)
        for layer in model.depth_transformer.layers:
            for module in (layer.self_attn, *layer.cross_attns,
                           layer.gating, layer.ca_gating):
                _assert_nonzero_finite_grad(module)
        for classifier in model.depformer_classifier:
            _assert_nonzero_finite_grad(classifier)


def test_detached_foremotion_teacher_matches_c_model_and_restores_student():
    torch.manual_seed(63)
    trainer, student = _factory(UpperFaceLowerGTDM3ConditionCRegretRVQTrainer, False)
    _, direct_teacher = _factory(UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer, False)
    direct_teacher.load_state_dict(student.state_dict(), strict=True)
    student.eval()
    direct_teacher.eval()
    trainer.regret_include_depth_levels = True
    batch = _batch()
    codes, audio, text, speaker = batch
    original_memories = [layer.cross_attns for layer in student.temporal_transformer.layers]
    original_parameters = tuple(id(parameter) for parameter in student.parameters())
    student_logits = _forward(student, batch)
    with no_compile():
        q0, depth = trainer._forward_regret_teacher_view(
            split="train", input_codes=codes, audio_codes=audio, text_codes=text,
            sum_condition=speaker, ca_depth_padding_mask=None,
            depth_input_codes=codes,
        )
    teacher_logits = torch.cat([q0, depth], dim=1)
    assert teacher_logits.shape == student_logits.shape
    assert not q0.requires_grad and not depth.requires_grad
    assert all(parameter.grad is None for parameter in student.parameters())
    assert tuple(id(parameter) for parameter in student.parameters()) == original_parameters
    assert all(layer.cross_attns is original for layer, original in zip(
        student.temporal_transformer.layers, original_memories,
    ))
    assert all(attention.causal for layer in student.temporal_transformer.layers
               for attention in layer.cross_attns)
    with torch.no_grad():
        torch.testing.assert_close(teacher_logits, _forward(direct_teacher, batch))
        torch.testing.assert_close(student_logits, _forward(student, batch))
    # Keep a causal forward graph alive across the temporary teacher views.
    # Student CE plus actual teacher-to-student KL must still backpropagate.
    ce = F.cross_entropy(student_logits.reshape(-1, 17), codes.reshape(-1))
    kl = F.kl_div(F.log_softmax(student_logits, dim=-1),
                  F.softmax(teacher_logits, dim=-1), reduction="none").sum(-1).mean()
    assert torch.isfinite(ce) and torch.isfinite(kl)
    (ce + kl).backward()
    _assert_nonzero_finite_grad(student.depth_transformer.layers[0].self_attn)
    _assert_nonzero_finite_grad(student.temporal_classifier)


if __name__ == "__main__":
    torch.set_num_threads(1)
    tests = [(name, value) for name, value in globals().copy().items()
             if name.startswith("test_") and callable(value)]
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"Shared depth weight smoke tests passed ({len(tests)} tests).")
