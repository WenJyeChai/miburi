"""CPU tests for leakage-safe past-as-future distillation and its matched control.

Run: NO_TORCH_COMPILE=1 python -B -m scripts.smoke_test_past_as_future_training
No dataset, checkpoint download, CUDA or optimizer construction is required.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import torch

from miburi.models.gesture_lm import GTemporalDepthModel3
from miburi.models.gesture_lm_past_future import (
    eligible_context_positions, forward_future_context_teacher,
    past_as_future, sample_context_positions,
)
from miburi.models.gesture_lm_shared_regret import forward_teacher_view
from scripts.smoke_test_gesture_lm_offline import _model_kwargs
from scripts.trainers.uflgtdm3_future_context_regret_rvq_trainer import (
    UpperFaceLowerGTDM3FutureContextRegretRVQTrainer as Trainer,
)


def fixture(scale=1):
    torch.manual_seed(7)
    kwargs = _model_kwargs()
    kwargs.update(num_layers=2, query2mem_scale=scale)
    model = GTemporalDepthModel3(**kwargs).eval()
    codes = torch.randint(16, (2, 20, 6))
    audio = torch.randint(17, (2, 2, 6 * scale))
    text = torch.randint(19, (2, 1, 6 * scale))
    return model, dict(input_codes=codes, audio_codes=audio, text_codes=text,
                       sum_condition=torch.tensor([1, 2]), positions=[2, 4])


class Tracker:
    def __init__(self):
        self.values = {}

    def update_meter(self, name, split, value):
        self.values[name, split] = float(value)


def trainer(model, mode):
    obj = Trainer.__new__(Trainer)
    obj.args = SimpleNamespace(ddp=False, regret_future_context=mode,
                               regret_context_targets=2, random_seed=2342,
                               regret_temperature=1.0, drop_lower_crossattn=True)
    obj.model = model
    obj.modelout_ignore_index = 16
    obj.regret_include_depth_levels = True
    obj.upper_gesture_codec = SimpleNamespace(num_codebooks=8)
    obj.lower_gesture_codec = SimpleNamespace(num_codebooks=8)
    obj._last_train_depth_input_codes = None
    obj._future_context_step = (3, 7)
    obj.tracker = Tracker()
    return obj


class PastAsFutureTests(unittest.TestCase):
    def test_joint_splice_preserves_prefix_alignment_and_length(self):
        for scale in (1, 2):
            memory = torch.arange(2 * 6 * scale * 3).reshape(2, 6 * scale, 3)
            before = memory.clone()
            result = past_as_future(memory, 2, scale)
            cut = 3 * scale
            torch.testing.assert_close(result[:, :cut], memory[:, :cut])
            torch.testing.assert_close(result[:, cut:], memory[:, :cut])
            torch.testing.assert_close(memory, before)
            self.assertEqual(result.shape, memory.shape)
        for t in (-1, 0, 1, 5, 6):
            with self.assertRaises(ValueError):
                past_as_future(torch.zeros(2, 6, 3), t)

    def test_eligibility_sampling_is_reproducible_and_does_not_consume_torch_rng(self):
        self.assertEqual(eligible_context_positions(6), [2, 3, 4])
        self.assertEqual(eligible_context_positions(125), list(range(62, 124)))
        self.assertEqual(eligible_context_positions(1), [])
        state = torch.get_rng_state().clone()
        a = sample_context_positions(125, 4, 2342)
        self.assertEqual(a, sample_context_positions(125, 4, 2342))
        self.assertNotEqual(a, sample_context_positions(125, 4, 3453))
        torch.testing.assert_close(state, torch.get_rng_state())
        self.assertEqual(sample_context_positions(2, 4, 1), [0])
        with self.assertRaises(ValueError):
            sample_context_positions(6, 0, 1)

    def test_matched_selected_teacher_equals_existing_global_teacher(self):
        for scale in (1, 2):
            model, args = fixture(scale)
            full_args = {k: v for k, v in args.items() if k != "positions"}
            q0, depth = forward_teacher_view(model, **full_args)
            expected = torch.cat((q0, depth), dim=1)[:, :, args["positions"]]
            result = forward_future_context_teacher(model, **args, context="matched")
            torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
            self.assertFalse(result.requires_grad)

    def test_true_future_cannot_leak_through_any_teacher_hidden_layer(self):
        model, args = fixture()
        args["positions"] = [2]
        original = {k: v.clone() if torch.is_tensor(v) else list(v) for k, v in args.items()}
        changed = dict(args)
        changed["audio_codes"] = args["audio_codes"].clone()
        changed["text_codes"] = args["text_codes"].clone()
        changed["audio_codes"][:, :, 3:] = (changed["audio_codes"][:, :, 3:] + 5) % 17
        changed["text_codes"][:, :, 3:] = (changed["text_codes"][:, :, 3:] + 5) % 19
        a = forward_future_context_teacher(model, **args, context="past_as_future")
        b = forward_future_context_teacher(model, **changed, context="past_as_future")
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        c = forward_future_context_teacher(model, **args, context="matched")
        d = forward_future_context_teacher(model, **changed, context="matched")
        self.assertGreater((c - d).abs().max().item(), 1e-5)
        for key in ("input_codes", "audio_codes", "text_codes", "sum_condition"):
            torch.testing.assert_close(args[key], original[key])

    def test_depth_uses_original_current_speech_and_stochastic_prefix(self):
        model, args = fixture(2)
        stochastic = (args["input_codes"] + 1) % 16
        stochastic[0, 8:16, 2] = model.pad_token_id
        with patch.object(model, "forward_depth_training", wraps=model.forward_depth_training) as depth:
            result = forward_future_context_teacher(
                model, **args, context="past_as_future", depth_input_codes=stochastic)
        audio, text = model.process_conditions(args["audio_codes"], args["text_codes"])
        for call, step in zip(depth.call_args_list, args["positions"]):
            torch.testing.assert_close(call.args[0], stochastic[:, :-1, step:step + 1])
            torch.testing.assert_close(call.kwargs["audio_condition"], audio.squeeze(1)[:, step * 2:(step + 1) * 2])
            torch.testing.assert_close(call.kwargs["text_condition"], text.squeeze(1)[:, step * 2:(step + 1) * 2])
        self.assertEqual(result.shape, (2, 20, 2, 17))

    def test_attention_policy_restored_after_exception(self):
        model, args = fixture()
        with patch.object(model, "forward_temporal", side_effect=RuntimeError("probe")):
            with self.assertRaisesRegex(RuntimeError, "probe"):
                forward_future_context_teacher(model, **args, context="past_as_future")
        self.assertTrue(all(c.causal for layer in model.temporal_transformer.layers for c in layer.cross_attns))
        self.assertTrue(all(layer.self_attn.causal for layer in model.temporal_transformer.layers))

    def test_trainer_pairs_targets_trims_padding_and_backpropagates_student(self):
        model, args = fixture()
        codes = args["input_codes"].clone()
        codes[1, :, 4:] = model.pad_token_id
        valid = codes != model.pad_token_id
        logits = model(codes, audio_codes=args["audio_codes"], text_codes=args["text_codes"],
                       sum_condition=args["sum_condition"])
        module = "scripts.trainers.uflgtdm3_future_context_regret_rvq_trainer"
        selections = []
        for mode in ("matched", "past_as_future"):
            obj = trainer(model, mode)
            with patch(module + ".forward_future_context_teacher", wraps=forward_future_context_teacher) as forward:
                loss = obj._compute_regret(
                    split="train", logits=logits, gesture_tokens=codes, pad_loss_mask=valid,
                    alpha=6, input_codes=codes, audio_codes=args["audio_codes"],
                    text_codes=args["text_codes"], sum_condition=args["sum_condition"],
                    sample_ids=["recording-a", "recording-b"])
            selections.append([(c.kwargs["input_codes"].shape[-1], c.kwargs["positions"]) for c in forward.call_args_list])
            self.assertEqual([x[0] for x in selections[-1]], [4, 6])
            self.assertTrue(torch.isfinite(loss))
            if mode == "past_as_future":
                loss.backward()
                self.assertGreater(sum(p.grad.abs().sum().item() for p in model.temporal_transformer.parameters() if p.grad is not None), 0)
                self.assertGreater(sum(p.grad.abs().sum().item() for p in model.depth_transformer.parameters() if p.grad is not None), 0)
        self.assertEqual(selections[0], selections[1])

    def test_noncontiguous_padding_is_rejected(self):
        model, args = fixture()
        obj = trainer(model, "past_as_future")
        valid = torch.ones_like(args["input_codes"], dtype=torch.bool)
        valid[0, :, 1] = False
        with self.assertRaisesRegex(ValueError, "contiguous"):
            obj._compute_regret(split="train", logits=torch.zeros(2, 20, 6, 17),
                                gesture_tokens=args["input_codes"], pad_loss_mask=valid,
                                alpha=6, **{k: v for k, v in args.items() if k != "positions"})

    def test_paired_configs_parse_and_differ_only_in_context_and_names(self):
        from scripts.trainers.utils import config
        root = Path(__file__).resolve().parents[1]
        configs = []
        for context in ("matched", "past_as_future"):
            path = root / "configs" / f"gtdm3_futurecontext_{context}_beatx_allspk.yaml"
            with patch.object(sys, "argv", ["check", "--config", str(path)]):
                parsed = config.parse_args()
            self.assertEqual(parsed.trainer, "UpperFaceLowerGTDM3FutureContextRegretRVQ")
            self.assertEqual(parsed.regret_future_context, context)
            configs.append(vars(parsed))
        differences = {key for key in configs[0] if configs[0][key] != configs[1][key]}
        self.assertLessEqual(differences, {"config", "project_name", "wandb_name", "regret_future_context", "name"})

    def test_scott_controls_preserve_reference_training_recipe(self):
        from scripts.trainers.utils import config
        root = Path(__file__).resolve().parents[1]
        controls = []
        for context in ("matched", "past_as_future"):
            path = root / "configs" / f"gtdm3_futurecontext_{context}_beatx_scott.yaml"
            with patch.object(sys, "argv", ["check", "--config", str(path)]):
                args = config.parse_args()
            self.assertEqual(args.trainer, "UpperFaceLowerGTDM3FutureContextRegretRVQ")
            self.assertEqual(args.regret_future_context, context)
            self.assertEqual(args.dataset_ratio, "scott_beatx_lowervalid")
            self.assertEqual(args.batch_size, 64)
            self.assertEqual(args.lr_policy, "cosine_delay")
            self.assertEqual(args.lr_cosine_start_epoch, 500)
            self.assertEqual(args.lr_cosine_end_epoch, -1)
            self.assertEqual(args.epochs, 10000)
            self.assertEqual(args.face_loss_weight, 2.5)
            self.assertEqual(args.regret_weight, 6)
            self.assertEqual(args.regret_initial_weight, 0.6)
            self.assertEqual(args.dense_future_gesture_weight, 0)
            self.assertEqual(args.kinematic_rvq_sample_probability, 0.15)
            self.assertEqual(args.kinematic_rvq_soft_target_weight, 0.5)
            self.assertEqual(args.wandb_mode, "offline")
            self.assertFalse(args.is_continue)
            self.assertIn("demoexp_release_uppercodec", args.upperbodycodec_ckpt)
            self.assertIn("demoexp_release_lowercodec", args.lowerbodycodec_ckpt)
            self.assertIn("demoexp_release_facecodec", args.facecodec_ckpt)
            controls.append(vars(args))
        differences = {key for key in controls[0] if controls[0][key] != controls[1][key]}
        self.assertLessEqual(differences, {"config", "project_name", "wandb_name", "wandb_tags", "regret_future_context", "name"})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
