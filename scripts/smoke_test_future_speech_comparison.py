"""CPU checks for future-content controls; no real data/checkpoints required.

Run: python -B -m scripts.smoke_test_future_speech_comparison
"""
from contextlib import contextmanager, redirect_stdout
import copy
from io import StringIO
import json
import math
import hashlib
import os
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

from miburi.models.gesture_lm import GTemporalDepthModel3
from miburi.models.gesture_lm_condition_c import forward_condition_c_teacher_view
from miburi.models.gesture_lm_future_control import (
    CONDITIONS, choose_token, generate_current_codes, speech_views, teacher_forced_logits,
    temporal_prediction, view_attention_bias,
)
from miburi.models import gesture_lm_future_control as controls
import miburi.utils.compile as compile_utils
from miburi.utils.compile import torch_compile_lazy
from scripts.smoke_test_gesture_lm_offline import _model_kwargs
from scripts import future_speech_comparison as experiment


def model_fixture(audio_streams=2):
    torch.manual_seed(42)
    kwargs = _model_kwargs()
    kwargs.update(num_layers=2, depformer_layers=2, context=3, memory_context=3,
                  audio_procemb=[torch.randn(17, 16) for _ in range(audio_streams)])
    return GTemporalDepthModel3(**kwargs).eval().requires_grad_(False)


def inputs(model, steps=7):
    torch.manual_seed(18)
    codes = torch.randint(model.card, (1, 20, steps))
    audio_codes = torch.randint(17, (1, len(model.audio_procemb), steps))
    text_codes = torch.randint(19, (1, 1, steps))
    audio, text = model.process_conditions(audio_codes, text_codes)
    return codes, audio_codes, text_codes, audio.squeeze(1), text.squeeze(1)


def current_logits(model, codes, audio, text, donor_audio, donor_text, step, clip_steps=7):
    start = step // clip_steps * clip_steps
    end = min(audio.shape[1], start + clip_steps)
    av, tv = speech_views(audio, text, donor_audio, donor_text, step=step, clip_start=start, clip_end=end)
    h, q = temporal_prediction(model, codes[:, :, :step].expand(3, -1, -1), av, tv, 1, clip_steps)
    return teacher_forced_logits(model, h, q, codes[:, :, step:step + 1],
                                 audio[:, step:step + 1].repeat(3, 1, 1),
                                 text[:, step:step + 1].repeat(3, 1, 1), 1)


class FutureSpeechChecks(unittest.TestCase):
    def test_runner_bypasses_lazy_compile_and_restores_policy_on_failure(self):
        cli = experiment.parse_args(["--config", "unused.yaml", "--checkpoint", "unused.safetensors",
                                     "--eval-hdf5", "unused.hdf5", "--mode", "tokens", "--device", "cpu"])
        with patch.dict(os.environ, {"NO_TORCH_COMPILE": ""}):
            @torch_compile_lazy
            def probe(value):
                return value.square() + 1

        previous = compile_utils._compile_disabled

        def check_before_loading_assets(settings):
            self.assertTrue(compile_utils._compile_disabled)
            torch.testing.assert_close(probe(torch.tensor([2.0])), torch.tensor([5.0]))
            raise RuntimeError("intentional eager probe stop")

        with patch.object(torch, "compile", side_effect=AssertionError("Compiler must not be invoked")) as compiler, \
             patch.object(experiment.codec_audit, "parse_lm_config", side_effect=check_before_loading_assets), \
             redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "intentional eager probe stop"):
            experiment.run(cli)
        compiler.assert_not_called()
        self.assertEqual(compile_utils._compile_disabled, previous)

    def test_training_provenance_accepts_global_without_relabelling_the_experiment(self):
        for trainer in ("UpperFaceLowerGTDM3ConditionCRegretRVQ", "UpperFaceLowerGTDM3SharedRegretRVQ",
                        "UpperFaceLowerGTDM3SharedRegret"):
            with self.subTest(trainer=trainer):
                config = SimpleNamespace(trainer=trainer)
                protocol = experiment.checkpoint_protocol(config)
                self.assertEqual(config.trainer, trainer)
                self.assertEqual(protocol["training_trainer"], trainer)
                self.assertEqual(protocol["evaluation_attention"], "condition_c_identical_rolling_past_plus_controlled_future")
                is_global = "SharedRegret" in trainer
                self.assertEqual(protocol["interpretation_note"] is not None, is_global)
                with TemporaryDirectory() as directory:
                    output = Path(directory)
                    experiment.write_report(output, [], [], protocol)
                    report = (output / "report.md").read_text(encoding="utf-8")
                    self.assertIn(trainer, report)
                    if is_global:
                        self.assertIn("do not reproduce its original GlobalRegret teacher mask", report)
        for trainer in ("UpperFaceLowerGTDM3Offline", "UpperFaceLowerGTDM3ConditionCTeacherRVQ", "unknown"):
            with self.subTest(unsupported=trainer), self.assertRaisesRegex(ValueError, "Unsupported trainer"):
                experiment.checkpoint_protocol(SimpleNamespace(trainer=trainer))

    def test_legacy_speaker_map_uses_original_training_cache_and_preserves_saved_indices(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "train.hdf5"
            with h5py.File(training, "w") as database:
                for name, values in (("chunk_ids", ["1_wayne_0_1_1_C0", "2_scott_0_1_1_C0"]),
                                     ("chunk_relpaths", ["wayne", "scott"]),
                                     ("chunk_splits", ["train", "train"]),
                                     ("chunk_speaker_ids", ["wayne", "scott"])):
                    database.create_dataset(name, data=np.array(values, dtype="S"))
                database.create_dataset("chunk_is_sitting", data=np.zeros(2, dtype=bool))
            checksum = hashlib.sha256(training.read_bytes()).hexdigest()
            config = root / "config.yaml"
            original = {"dataset": "unified", "dataset_ratio": "scott_beatx_lowervalid",
                        "beatx_cache_path": str(training), "embody3d_cache_path": None}
            config.write_text(json.dumps(original), encoding="utf-8")
            cli = SimpleNamespace(config=config, checkpoint=root / "last_1000.safetensors", speaker_map=None,
                                  eval_hdf5=root / "evaluation_cache_must_not_be_used.hdf5")
            with redirect_stdout(StringIO()):
                mapping, source = experiment.load_speaker_map(cli)
            self.assertEqual(mapping, {"wayne": 0, "scott": 1})
            self.assertEqual(source["kind"], "reconstructed_from_original_training_cache")
            self.assertEqual(source["training_dataset_ratio"], "scott_beatx_lowervalid")
            self.assertEqual(hashlib.sha256(training.read_bytes()).hexdigest(), checksum)
            saved = root / "speaker_id_to_index.json"
            self.assertFalse(saved.exists())
            self.assertFalse(list(root.glob("*.chunk_index*")))
            saved.write_text('{"scott": 4}', encoding="utf-8")
            mapping, source = experiment.load_speaker_map(cli)
            self.assertEqual(mapping, {"scott": 4})
            self.assertEqual(source["kind"], "saved_json")
            cli.speaker_map = root / "explicit_missing.json"
            with self.assertRaisesRegex(FileNotFoundError, "Speaker map not found"):
                experiment.load_speaker_map(cli)
            cli.speaker_map = None
            saved.unlink()
            original["beatx_cache_path"] = str(root / "missing_training.hdf5")
            config.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "original training cache"):
                experiment.load_speaker_map(cli)

    def test_joint_splice_preserves_past_current_and_exact_future_length(self):
        audio = torch.arange(14).reshape(1, 14, 1)
        text = audio + 100
        donor_audio = torch.arange(6).reshape(1, 6, 1) + 1000
        donor_text = donor_audio + 100
        original = audio.clone()
        av, tv = speech_views(audio, text, donor_audio, donor_text, step=4, clip_start=3, clip_end=6, scale=2)
        self.assertEqual(av.shape, (3, 12, 1))
        torch.testing.assert_close(av[:, :10], audio[:, :10].expand(3, -1, -1))
        torch.testing.assert_close(tv[:, :10], text[:, :10].expand(3, -1, -1))
        torch.testing.assert_close(av[2, 10:], donor_audio[0, 4:])
        torch.testing.assert_close(tv[2, 10:], donor_text[0, 4:])
        torch.testing.assert_close(audio, original)
        donor_audio[:, :4] = -999
        av_again, _ = speech_views(audio, text, donor_audio, donor_text, step=4, clip_start=3, clip_end=6, scale=2)
        torch.testing.assert_close(av, av_again)

    def test_future_upper_bound_respects_each_clip_and_cfg_roles(self):
        mask = view_attention_bias(7, 8, 4, scale=2, batches=6)
        self.assertEqual(mask.shape, (6, 1, 7, 16))
        self.assertTrue(mask[0, 0, 2, :6].all())
        self.assertFalse(mask[0, 0, 2, 6:].any())
        self.assertTrue(mask[1, 0, 2, :8].all())
        self.assertFalse(mask[1, 0, 2, 8:].any())
        self.assertTrue(mask[1, 0, 4].all())
        torch.testing.assert_close(mask[:3], mask[3:])

    def test_causal_and_matched_logits_match_existing_implementations(self):
        model = model_fixture()
        codes, raw_a, raw_t, a, t = inputs(model)
        actual = current_logits(model, codes, a, t, a.clone(), t.clone(), step=3)
        cross_mask = torch.zeros_like(codes, dtype=torch.bool)
        cross_mask[:, 8:16] = True
        with torch.no_grad():
            causal = model(codes, raw_a, raw_t, sum_condition=torch.tensor([1]), ca_depth_padding_mask=cross_mask)
            q, d = forward_condition_c_teacher_view(model, input_codes=codes, audio_codes=raw_a,
                                                     text_codes=raw_t, sum_condition=torch.tensor([1]),
                                                     ca_depth_padding_mask=cross_mask)
        matched = torch.cat((q, d), 1)
        torch.testing.assert_close(actual[0], causal[0, :, 3], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(actual[1], matched[0, :, 3], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(actual[1], actual[2])

    def test_correct_future_cannot_leak_into_mismatched_hidden_states(self):
        model = model_fixture()
        codes, _, _, a, t = inputs(model)
        donor_a, donor_t = torch.randn_like(a), torch.randn_like(t)
        before = current_logits(model, codes, a, t, donor_a, donor_t, step=3)
        changed_a, changed_t = a.clone(), t.clone()
        changed_a[:, 4:] += 10
        changed_t[:, 4:] -= 10
        after = current_logits(model, codes, changed_a, changed_t, donor_a, donor_t, step=3)
        torch.testing.assert_close(before[0], after[0])
        torch.testing.assert_close(before[2], after[2])
        self.assertGreater(float((before[1] - after[1]).abs().max()), 1e-4)

    def test_no_future_and_identity_donor_controls(self):
        model = model_fixture()
        codes, _, _, a, t = inputs(model)
        no_future = current_logits(model, codes, a, t, torch.randn_like(a), torch.randn_like(t), step=6)
        torch.testing.assert_close(no_future[1], no_future[2])
        for cfg in (1.0, 1.5):
            av, tv = speech_views(a, t, a, t, step=3, clip_start=0, clip_end=7)
            histories = codes[:, :, :3].expand(3, -1, -1)
            ac, tc = a[:, 3:4].repeat(3, 1, 1), t[:, 3:4].repeat(3, 1, 1)
            if cfg != 1:
                histories = histories.repeat(2, 1, 1)
                av, tv = torch.cat((av, torch.zeros_like(av))), torch.cat((tv, torch.zeros_like(tv)))
                ac, tc = torch.cat((ac, torch.zeros_like(ac))), torch.cat((tc, torch.zeros_like(tc)))
            h, q = temporal_prediction(model, histories, av, tv, 1, 7)
            for sampling in (False, True):
                generated = generate_current_codes(model, h, q, ac, tc, 1, rng=torch.Generator().manual_seed(9),
                                                     sampling=sampling, cfg_scale=cfg)
                torch.testing.assert_close(generated[1], generated[2])
                self.assertFalse((generated == model.pad_token_id).any())

    def test_weights_modules_gradients_and_streaming_states_are_preserved(self):
        model = model_fixture()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        modules = [layer.cross_attns for layer in model.temporal_transformer.layers]
        codes, _, _, a, t = inputs(model)
        result = current_logits(model, codes, a, t, torch.randn_like(a), torch.randn_like(t), 3)
        self.assertFalse(result.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for old, layer in zip(modules, model.temporal_transformer.layers):
            self.assertIs(old, layer.cross_attns)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key])
        self.assertFalse(model.temporal_transformer.is_streaming)
        self.assertFalse(model.depth_transformer.is_streaming)

    def test_common_sampler_never_selects_padding_or_filtered_zero_mass(self):
        logits = torch.tensor([[-100.0, 1.0, 2.0, 100.0]]).repeat(3, 1)
        for uniform in (0.0, 0.25, 0.999999):
            sampled = choose_token(logits, pad_id=3, sampling=True, temperature=0.9, top_p=0.01, uniform=uniform)
            torch.testing.assert_close(sampled, torch.full((3,), 2))

    def test_cfg_respects_existing_lower_body_guidance_mask(self):
        model = model_fixture()
        codes, _, _, a, t = inputs(model)
        av, tv = speech_views(a, t, a, t, step=3, clip_start=0, clip_end=7)
        history = codes[:, :, :3].expand(3, -1, -1).repeat(2, 1, 1)
        h, q = temporal_prediction(model, history, torch.cat((av, torch.zeros_like(av))),
                                    torch.cat((tv, torch.zeros_like(tv))), 1, 7)
        ac, tc = a[:, 3:4].repeat(3, 1, 1), t[:, 3:4].repeat(3, 1, 1)
        ac, tc = torch.cat((ac, torch.zeros_like(ac))), torch.cat((tc, torch.zeros_like(tc)))
        captured, raw_depth = [], []
        choose = controls.choose_token
        forward = model.forward_depth

        def capture(logits, **kwargs):
            captured.append(logits.clone())
            return choose(logits, **kwargs)

        def capture_depth(*args, **kwargs):
            result = forward(*args, **kwargs)
            raw_depth.append(result.reshape(6, -1))
            return result

        with patch.object(controls, "choose_token", side_effect=capture), patch.object(model, "forward_depth", side_effect=capture_depth):
            generate_current_codes(model, h, q, ac, tc, 1, rng=torch.Generator().manual_seed(1),
                                   sampling=False, cfg_scale=1.5, drop_lower=True)
        for index, raw in enumerate(raw_depth):
            conditional, null = raw.chunk(2)
            expected = conditional if 7 <= index < 15 else null + 1.5 * (conditional - null)
            torch.testing.assert_close(captured[index + 1], expected)

    def test_donor_plan_is_reproducible_same_speaker_and_different_utterance(self):
        infos = []
        for i, speaker in enumerate(("scott", "scott", "wayne", "wayne")):
            steps = 7 + i
            infos.append({"recording_id": f"recording_{i}", "speaker": speaker, "steps": steps,
                          "audio": torch.full((8, steps), i), "text": torch.full((1, steps), i)})
        plan = experiment.make_donor_plan(infos, infos, 4, 2342)
        self.assertEqual(plan, experiment.make_donor_plan(infos, infos, 4, 2342))
        lookup = {i["recording_id"]: i for i in infos}
        for row in plan:
            self.assertNotEqual(row["recording_id"], row["donor_recording_id"])
            self.assertEqual(row["speaker"], lookup[row["donor_recording_id"]]["speaker"])
            self.assertEqual(row["clip_end_step"] - row["clip_start_step"], row["donor_end_step"] - row["donor_start_step"])
        with self.assertRaises(ValueError):
            experiment.make_donor_plan(infos[:1], infos[:1], 4, 2342)
        identical_future = copy.deepcopy(infos[:2])
        identical_future[1]["audio"] = identical_future[0]["audio"].clone()
        identical_future[1]["text"] = identical_future[0]["text"].clone()
        identical_future[1]["steps"] = identical_future[0]["steps"]
        identical_future[1]["audio"][:, 0] += 1
        with self.assertRaises(ValueError):
            experiment.make_donor_plan(identical_future, identical_future[:1], 7, 2342)

    def test_ce_accuracy_denominators_and_paired_recording_bootstrap(self):
        logits = torch.zeros(3, 20, 17)
        valid = torch.ones(20, dtype=torch.bool)
        valid[16:] = False
        values = experiment.count_tokens(logits, torch.zeros(20, dtype=torch.long), valid)
        self.assertAlmostEqual(values[0, :, 0].sum(), math.log(17) * 16, places=5)
        self.assertEqual(values[0, :, 1].sum(), 16)
        self.assertEqual(values[0, :, 3].sum(), 16)
        rows = []
        for recording in ("one", "two"):
            totals = np.zeros((3, 20, 4))
            for view, (ce, accuracy) in enumerate(((2, 0.4), (1, 0.8), (3, 0.3))):
                totals[view, :, 0] = ce * 10
                totals[view, :, 1] = accuracy * 10
                totals[view, :, 3] = 10
            rows += experiment.token_rows(totals, {"recording_id": recording, "speaker": "scott"})
        table = experiment.aggregate_tokens(rows)
        matched = next(r for r in table if r["condition"] == "matched_future" and r["scope"] == "all")
        self.assertEqual(matched["ce_nats"], 1)
        self.assertEqual(matched["accuracy"], 0.8)
        diff = experiment.paired_differences(rows, 100, 2)
        ce = next(r for r in diff if r["scope"] == "all" and r["contrast"] == "matched_future - causal" and r["metric"] == "ce_nats")
        self.assertEqual((ce["difference"], ce["ci95_low"], ce["ci95_high"]), (-1, -1, -1))

    def test_strict_checkpoint_loader_uses_saved_embeddings_and_config(self):
        from safetensors.torch import save_file
        from miburi.models import loaders
        original = model_fixture(audio_streams=8)
        kwargs = _model_kwargs()
        for name in ("num_heads", "num_layers", "depformer_heads", "depformer_layers", "query2mem_scale",
                     "num_temp_classifiers", "text_procemb", "audio_procemb", "gesture_codec_layers", "body_parts", "bp_dist"):
            kwargs.pop(name, None)
        kwargs.update(context=3, memory_context=3)
        config = SimpleNamespace(gestureformer_heads=4, gestureformer_layers=2,
                                 gestureformer_depformer_heads=4, gestureformer_depformer_layers=2,
                                 gestureformer_depformer_weights_per_step=True, num_temp_classifiers=1,
                                 vad_guidance=False, vad_use_face_logits=False, textaudio_emb_freeze=False,
                                 param_dtype="float32")
        codecs = {}
        for part, lo, hi in (("upper", 0, 8), ("lower", 8, 16), ("face", 16, 20)):
            codecs[part] = SimpleNamespace(quantizer=SimpleNamespace(vq=SimpleNamespace(layers=original.gesture_codec_layers[lo:hi])))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last_0.safetensors"
            save_file({k: v.detach().clone().contiguous() for k, v in original.state_dict().items()}, str(checkpoint))
            with patch.object(loaders, "get_gesturelm_kwargs", return_value=kwargs):
                restored = experiment.load_generator(config, codecs, checkpoint, "cpu")
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])
            config.param_dtype = "bfloat16"
            with patch.object(loaders, "get_gesturelm_kwargs", return_value=kwargs):
                reduced_precision = experiment.load_generator(config, codecs, checkpoint, "cpu")
            self.assertTrue(all(p.dtype == torch.bfloat16 for p in reduced_precision.parameters()))
            config.param_dtype = "float32"
            with torch.no_grad():
                codecs["upper"].quantizer.vq.layers[0]._codebook.embedding.add_(1)
            with patch.object(loaders, "get_gesturelm_kwargs", return_value=kwargs), self.assertRaisesRegex(ValueError, "codec does not match"):
                experiment.load_generator(config, codecs, checkpoint, "cpu")

    def test_metric_networks_are_shared_but_condition_accumulators_are_independent(self):
        class Metric:
            def __init__(self, config):
                self.eval_copy = nn.Linear(2, 2)
                self.smplx = nn.Linear(2, 2)
                for name in ("alignmenter", "gt_alignmenter", "l1_calculator", "gt_l1_calculator"):
                    setattr(self, name, SimpleNamespace(history=[]))

            def reset(self):
                self.dataset_len = 0
                self.pred_latents = []

        module = types.ModuleType("_future_comparison_metric_utils.metrics")
        module.GestureMetrics = Metric
        with patch.dict(sys.modules, {module.__name__: module}):
            metrics = experiment.metric_instances(SimpleNamespace())
        self.assertIs(metrics["causal"].eval_copy, metrics["matched_future"].eval_copy)
        self.assertIs(metrics["causal"].smplx, metrics["matched_future"].smplx)
        for network in ("eval_copy", "smplx"):
            self.assertFalse(getattr(metrics["causal"], network).training)
            self.assertTrue(all(not p.requires_grad for p in getattr(metrics["causal"], network).parameters()))
        metrics["causal"].l1_calculator.history.append(1)
        metrics["causal"].pred_latents.append(2)
        self.assertEqual(metrics["matched_future"].l1_calculator.history, [])
        self.assertEqual(metrics["matched_future"].pred_latents, [])

    def test_full_runner_exports_token_and_generation_tables_on_identical_test_records(self):
        class Codec:
            def __init__(self, part, count, features):
                self.part, self.count, self.features = part, count, features

            def encode(self, value):
                return torch.arange(self.count, device=value.device)[None, :, None].expand(value.shape[0], -1, value.shape[1] // 2).clone()

            @contextmanager
            def streaming(self, batch):
                yield self

            def decode(self, codes):
                result = torch.zeros(codes.shape[0], codes.shape[-1] * 2, self.features, device=codes.device)
                joints = {"upper": 43, "lower": 9, "face": 1}[self.part]
                result[..., :joints * 6] = torch.tensor([1, 0, 0, 0, 1, 0], device=codes.device).repeat(joints)
                if self.part == "face":
                    result[..., 6:] = (codes[:, 0].float() / 16).repeat_interleave(2, dim=-1)[:, :, None]
                return result

        class Metric:
            def reset(self):
                self.dataset_len = 0

            def update(self, values):
                assert compile_utils._compile_disabled
                assert values["tar_beta"].shape == (300,)
                assert values["rec_pose"].shape == values["tar_pose"].shape == (8, 165)
                assert values["rec_exps"].shape == (8, 100)
                self.dataset_len += 1

            def compute_metrics(self):
                return {"fgd": 0.25, "beat_alignment": 0.7, "gt_beat_alignment": 0.75,
                        "facial_l2": 0.01, "l1_diversity": 1.0, "gt_l1_diversity": 1.2}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "eval.hdf5"
            ids = ["2_scott_0_1_1", "2_scott_0_2_2", "2_scott_0_9_9"]
            splits = ["test", "test", "train"]
            with h5py.File(path, "w") as database:
                database.attrs["motion_fps"] = 25
                for name, values in (("chunk_ids", ids), ("chunk_relpaths", ids),
                                     ("chunk_splits", splits), ("chunk_speaker_ids", ["scott"] * 3)):
                    database.create_dataset(name, data=np.array(values, dtype="S"))
                database.create_dataset("chunk_is_sitting", data=np.zeros(3, dtype=bool))
                for index, (fid, split) in enumerate(zip(ids, splits)):
                    group = database.create_group(fid)
                    group.attrs.update(file_id=fid, split=split, chunk_startsec=0.0, chunk_endsec=8 / 25,
                                       speaker_id="scott", is_sitting=False)
                    for name, shape in (("motion", (8, 55, 3)), ("transl", (8, 3)), ("contacts", (8, 4)),
                                        ("expressions", (8, 100)), ("betas", (300,))):
                        group.create_dataset(name, data=np.zeros(shape, dtype=np.float32))
                    group.create_dataset("pose_valid", data=np.ones(8, dtype=bool))
                    group.create_dataset("moshi_audiotokens", data=np.full((1, 8, 4), index + 1, dtype=np.int64))
                    group.create_dataset("moshi_texttokens", data=np.full((1, 1, 4), index + 1, dtype=np.int64))
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "speaker_id_to_index.json").write_text('{"scott": 1}', encoding="utf-8")
            for filename in ("model.safetensors", "upper.safetensors", "lower.safetensors", "face.safetensors"):
                (root / filename).write_bytes(b"synthetic fixture only")
            cli = experiment.parse_args([
                "--config", str(experiment.ROOT / "configs/gtdm3_foremotion_c_rvq_beatx_scott.yaml"),
                "--checkpoint", str(root / "model.safetensors"), "--eval-hdf5", str(path),
                "--upper-codec", str(root / "upper.safetensors"), "--lower-codec", str(root / "lower.safetensors"),
                "--face-codec", str(root / "face.safetensors"), "--output-dir", str(root / "out"),
                "--generation-seeds", "0", "5", "--clip-steps", "2", "--bootstrap", "20", "--sampler", "greedy", "--save-motion",
            ])
            codecs = {"upper": Codec("upper", 8, 258), "lower": Codec("lower", 8, 61), "face": Codec("face", 4, 106)}
            metrics = {name: Metric() for name in CONDITIONS}
            with patch.object(torch.cuda, "set_device"), \
                 patch.object(experiment.codec_audit, "load_codecs", return_value=codecs), \
                 patch.object(experiment, "load_generator", return_value=model_fixture(audio_streams=8)), \
                 patch.object(experiment, "metric_instances", return_value=metrics), redirect_stdout(StringIO()):
                output = experiment.run(cli)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["recordings"], 2)
            self.assertEqual(metadata["split"], "test")
            self.assertEqual(metadata["execution_mode"], "eager_pytorch_no_repo_torch_compile")
            import csv
            with (output / "token_summary.csv").open() as handle:
                summary = list(csv.DictReader(handle))
            self.assertTrue(all(int(r["tokens"]) == 80 for r in summary if r["scope"] == "all"))
            with (output / "generation_metrics.csv").open() as handle:
                generation = list(csv.DictReader(handle))
            self.assertEqual(len(generation), 6)
            self.assertTrue(all(int(r["recordings"]) == 2 for r in generation))
            self.assertTrue(all(float(r["expression_mse"]) >= 0 for r in generation))
            self.assertTrue((output / "generated_motion/0/2_scott_0_1_1.npz").is_file())
            self.assertTrue((output / "paired_token_differences.csv").is_file())
            with (output / "token_intervention_coverage.csv").open() as handle:
                coverage = list(csv.DictReader(handle))
            self.assertTrue(all(r["scored_positions"] == r["scored_positions_with_changed_future"] for r in coverage))
            self.assertTrue((output / "side_by_side.csv").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
