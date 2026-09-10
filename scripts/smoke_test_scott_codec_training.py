"""CPU checks for the opt-in Scott codec training workflow.

Run ``python -B -m scripts.smoke_test_scott_codec_training`` from the repository.
Checks use synthetic motion and temporary outputs; they do not launch training,
contact W&B, modify released checkpoints, or require a GPU.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]
PARTS = ("upper", "lower", "face")
CONFIGS = {part: ROOT / f"configs/mimi_scott_scratch_{part}.yaml" for part in PARTS}


def _close(actual, expected, atol=1e-6):
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.as_tensor(expected, dtype=torch.float64), atol=atol, rtol=0,
    )


def _parse(part):
    from scripts.trainers.utils.config import parse_args

    with patch.object(sys, "argv", ["smoke_scott_codec", "--config", str(CONFIGS[part])]):
        return parse_args()


def test_scott_training_splits():
    from scripts.smoke_test_codec_reconstruction_audit import _make_hdf5
    from scripts.trainers.dataloaders.unified_dataset import UNIFIEDDataset

    with TemporaryDirectory(prefix="scott-codec-split-") as directory:
        folder = Path(directory)
        path = folder / "tiny_beatx.h5"
        ids = _make_hdf5(path, frames=250, invalid_val=False)
        for part in PARTS:
            args = _parse(part)
            args.beatx_cache_path = str(path)
            args.embody3d_cache_path = None
            args.index_cache_dir = str(folder / "index")
            train = UNIFIEDDataset(args, "train", only_motion=True, dataset_ratio=args.dataset_ratio)
            val = UNIFIEDDataset(args, "val", only_motion=True, dataset_ratio=args.dataset_ratio,
                                 varying_frame_length=False)
            try:
                assert [ref.chunk_id for ref in train._chunk_refs] == [ids[0]]
                assert [ref.chunk_id for ref in val._chunk_refs] == [ids[1]]
                assert train[0]["split"] == "train" and val[0]["split"] == "val"
                assert train[0]["audio_tokens"] is None
                assert ids[2] not in train.chunk_id_df["filechunk_id"].tolist()
            finally:
                train.close()
                val.close()


def test_fresh_models_match_release_checkpoint_shapes():
    from safetensors import safe_open
    from miburi.models import loaders
    from scripts.trainers.basecausalcodec_trainer import BaseCausalCodecTrainer

    for part in PARTS:
        args = _parse(part)
        assert args.is_train and not args.is_continue and not args.continue_ckpt
        assert args.dataset_ratio == "scott_beatx_lowervalid"
        assert args.pose_length == 250 and args.motion_fps == 25 and args.frame_chunk_size == 2
        assert args.param_dtype == args.optim_dtype == args.latent_dtype == "float32"
        factory = getattr(loaders, f"get_{part}gesturecodec_kwargs")
        # The legacy constructor pops entries from nested VQ kwargs. Isolate
        # that global dictionary without substituting any actual architecture.
        with patch.object(loaders, f"get_{part}gesturecodec_kwargs", return_value=deepcopy(factory())):
            model = BaseCausalCodecTrainer.get_model(args)
        model.init_weights()
        expected = dict(upper=(258, 8), lower=(61, 8), face=(106, 4))[part]
        channels = model.channels[0] if isinstance(model.channels, list) else model.channels
        assert channels == expected[0]
        assert model.num_codebooks == expected[1] and model.cardinality == 2048
        assert model.frame_rate == 12.5
        assert all(parameter.requires_grad for parameter in model.parameters())
        assert not any(layer._codebook.initialized for layer in model.quantizer.vq.layers)
        checkpoint_dir = ROOT / f"experiments/demoexp_release_{part}codec"
        checkpoints = list(checkpoint_dir.glob("*.safetensors"))
        assert checkpoints, f"Missing local release checkpoint in {checkpoint_dir}"
        checkpoint = checkpoints[0]
        with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
            checkpoint_shapes = {}
            for name in weights.keys():
                clean = name
                while clean.startswith(("module.", "m.")):
                    clean = clean.split(".", 1)[1]
                checkpoint_shapes[clean] = tuple(weights.get_slice(name).get_shape())
        assert {name: tuple(value.shape) for name, value in model.state_dict().items()} == checkpoint_shapes
        del model


def test_scratch_reconstruction_trains_backbone_and_rvq():
    from miburi.models import loaders
    from scripts.trainers.basecausalcodec_trainer import BaseCausalCodecTrainer

    torch.manual_seed(31)
    args = _parse("face")
    args.transformer_layers, args.transformer_heads, args.convblock_layers = 1, 2, 1
    kwargs = deepcopy(loaders.get_facegesturecodec_kwargs())
    kwargs.update(latent_dim=32, n_filters=16, dim_feedforward=64, dropout=0)
    kwargs["vq_args"].update(dimension=8, n_q=2, bins=16, input_dimension=32,
                             output_dimension=32, q_dropout=False, no_quantization_rate=0)
    with patch.object(loaders, "get_facegesturecodec_kwargs", return_value=kwargs):
        model = BaseCausalCodecTrainer.get_model(args)
    model.init_weights()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    values = torch.randn(2, 8, 106)
    initial = model.quantizer.vq.layers[0]._codebook.embedding_sum.clone()
    # Residual codebooks initialize progressively; two training passes permit
    # both stages of this tiny CPU model to initialize from encoder latents.
    for _ in range(2):
        optimizer.zero_grad()
        decoded, quantized, _, _ = model(values)
        loss = (decoded - values).square().mean() + quantized.penalty
        assert bool(torch.isfinite(loss))
        loss.backward()
        for module in (model.seanet_encoder, model.seanet_decoder):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients) > 0
        optimizer.step()
    assert all(layer._codebook.initialized for layer in model.quantizer.vq.layers)
    assert not torch.equal(initial, model.quantizer.vq.layers[0]._codebook.embedding_sum)


def test_launcher_commands_are_scoped_and_fresh():
    from scripts import train_scott_codecs as launcher

    for part in (*PARTS, "all"):
        args = launcher.parse_args(["--part", part, "--dry_run", "--wandb", "False"])
        commands = launcher.build_commands(args, run_id="smoke")
        selected = PARTS if part == "all" else (part,)
        assert len(commands) == len(selected)
        for selected_part, command in zip(selected, commands):
            assert command[0] == sys.executable
            assert Path(command[1]).resolve() == ROOT / "scripts/train.py"
            def value(flag):
                return command[command.index(flag) + 1]
            assert Path(value("--config")).resolve() == CONFIGS[selected_part]
            assert value("--is_continue") == "False"
            assert value("--dataset_ratio") == "scott_beatx_lowervalid"
            assert value("--codec_standard_eval") == "True"
            assert "smoke" in value("--notes")
            assert "--continue_ckpt" not in command
            assert "--test_ckpt" not in command
    # Resume, alternate architecture and pretrained weights are deliberately
    # unavailable through this from-scratch launcher.
    import contextlib
    import io
    for forbidden in (
        ["--is_continue", "True"], ["--continue_ckpt", "old.safetensors"],
        ["--config", "other.yaml"], ["--test_ckpt", "old.safetensors"],
    ):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                launcher.parse_args(forbidden)
            except SystemExit as error:
                assert error.code != 0
            else:
                raise AssertionError(f"Fresh-only launcher accepted {forbidden[0]}")
    args = launcher.parse_args(["--part", "face", "--batch_size", "7", "--epochs", "45",
                                "--out_path", "temporary codec output", "--dry_run"])
    command = launcher.build_commands(args, run_id="smoke-two")[0]
    assert command[command.index("--batch_size") + 1] == "7"
    assert command[command.index("--epochs") + 1] == "45"
    assert command[command.index("--out_path") + 1].endswith(("/", "\\"))


def test_launcher_dry_run_and_failure_stop():
    import contextlib
    import io
    import subprocess
    from scripts import train_scott_codecs as launcher

    with contextlib.redirect_stdout(io.StringIO()), \
         patch.object(launcher, "preflight") as preflight, \
         patch.object(launcher.subprocess, "run") as child:
        assert launcher.main(["--part", "all", "--dry_run"]) == 0
        preflight.assert_not_called()
        child.assert_not_called()
    failure = subprocess.CalledProcessError(7, ["synthetic codec child"])
    with contextlib.redirect_stdout(io.StringIO()), \
         patch.object(launcher, "preflight") as preflight, \
         patch.object(launcher.subprocess, "run", side_effect=[None, failure]) as child:
        try:
            launcher.main(["--part", "all"])
        except subprocess.CalledProcessError as error:
            assert error.returncode == 7
        else:
            raise AssertionError("Failed codec child did not stop the launch sequence")
        preflight.assert_called_once()
        assert child.call_count == 2
        assert all(call.kwargs["check"] is True for call in child.call_args_list)


def test_final_validation_is_opt_in():
    import ast

    # Evaluate the real scheduling expression without importing/running the GPU
    # training entry point. A final non-periodic checkpoint must accompany the
    # new reconstruction validation; legacy runs retain their original cadence.
    tree = ast.parse((ROOT / "scripts/train.py").read_text(encoding="utf-8"))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "validation_due"
                           for target in node.targets)]
    assert len(assignments) == 1
    code = compile(ast.Expression(assignments[0].value), "train_validation_schedule", "eval")
    for enabled in (False, True):
        args = SimpleNamespace(codec_standard_eval=enabled, epochs=45, test_period=10)
        observed = [epoch for epoch in range(46)
                    if eval(code, {"args": args, "epoch": epoch})]
        assert observed == [0, 10, 20, 30, 40] + ([45] if enabled else [])


def test_standard_metric_units_pooling_and_shape_guards():
    from scripts.codec_reconstruction_metrics import merge_metric_sums
    from scripts.trainers.utils.codec_reconstruction_validation import (
        standard_feature_statistics, finalize_standard_metrics,
    )

    target = torch.zeros(1, 4, 61)
    decoded = target.clone()
    decoded[..., :54], decoded[..., 54:57], decoded[..., 57:] = .5, 2., .25
    values = finalize_standard_metrics(standard_feature_statistics("lower", target, decoded), "lower")
    _close(values["rotation6d_mse"], .25)
    _close(values["translation_velocity_mse"], 4)
    _close(values["contact_mse"], .0625)
    _close(values["rotation6d_mae"], .5)
    _close(values["rotation6d_rmse"], .5)
    _close(values["rotation6d_psnr_db"], 10 * math.log10(16))
    _close(values["contact_psnr_db"], 10 * math.log10(16))
    assert "translation_velocity_psnr_db" not in values

    # Pool coordinates across differently sized clips before taking the root.
    pooled = {}
    for frames, error in ((1, 1.), (3, 3.)):
        target = torch.zeros(frames, 106)
        decoded = target.clone()
        decoded[..., 6:] = error
        merge_metric_sums(pooled, standard_feature_statistics("face", target, decoded))
    values = finalize_standard_metrics(pooled, "face")
    _close(values["expression_mse"], 7)
    _close(values["expression_mae"], 2.5)
    _close(values["expression_rmse"], math.sqrt(7))
    assert "expression_psnr_db" not in values
    assert math.isinf(values["rotation6d_psnr_db"])
    empty = finalize_standard_metrics(standard_feature_statistics(
        "face", target, decoded, valid_mask=torch.zeros(3, dtype=torch.bool)), "face")
    assert all(value is None for value in empty.values())
    invalid = [("face", torch.zeros(2, 105), torch.zeros(2, 105), None),
               ("upper", torch.zeros(2, 258), torch.zeros(3, 258), None),
               ("lower", torch.zeros(2, 61), torch.zeros(2, 61), torch.ones(3)),
               ("face", torch.zeros(2, 106), torch.full((2, 106), float("nan")), None)]
    for part, target, decoded, valid_mask in invalid:
        try:
            standard_feature_statistics(part, target, decoded, valid_mask=valid_mask)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed/nonfinite standard metric inputs were accepted")


def test_best_checkpoint_is_separate_and_updates_only_on_improvement():
    import json
    from safetensors.torch import load_file
    from scripts.trainers.utils.codec_reconstruction_validation import save_best_reconstruction

    with TemporaryDirectory(prefix="scott-codec-best-") as directory:
        folder = Path(directory)
        old_checkpoint = folder / "last_10.safetensors"
        old_checkpoint.write_bytes(b"existing periodic checkpoint sentinel")
        trainer = SimpleNamespace(
            model=torch.nn.Linear(2, 2), global_rank=0, checkpoint_path=str(folder),
            args=SimpleNamespace(is_continue=False, motion_fps=25),
        )
        metrics = {"expression_mse": 1., "rotation6d_psnr_db": math.inf}
        assert save_best_reconstruction(trainer, "face", 10, metrics)
        best_path = folder / "best_reconstruction.safetensors"
        initial_bytes = best_path.read_bytes()
        with torch.no_grad():
            trainer.model.weight.add_(1)
        for value in (1., 2.):
            assert not save_best_reconstruction(trainer, "face", 20, {"expression_mse": value})
            assert best_path.read_bytes() == initial_bytes
        assert save_best_reconstruction(trainer, "face", 30, {"expression_mse": .5})
        metadata = json.loads((folder / "best_reconstruction.json").read_text(encoding="utf-8"))
        assert metadata["epoch"] == 30 and metadata["value"] == .5
        assert metadata["metric"] == "expression_mse" and metadata["split"] == "val"
        _close(load_file(str(best_path))["weight"], trainer.model.weight)
        assert old_checkpoint.read_bytes() == b"existing periodic checkpoint sentinel"
        assert not list(folder.glob("*.pending.*"))
        for value in (None, math.nan, math.inf):
            try:
                save_best_reconstruction(trainer, "face", 40, {"expression_mse": value})
            except ValueError:
                pass
            else:
                raise AssertionError("Nonfinite metric selected a best checkpoint")


def test_validation_restores_training_state_even_on_failure():
    from scripts.trainers.utils.codec_reconstruction_validation import frozen_validation_codec

    class Quantizer:
        num_codebooks, total_codebooks = 1, 4
        def set_num_codebooks(self, count):
            self.num_codebooks = count

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[1].eval()
    model[0].bias.requires_grad_(False)
    model.quantizer = Quantizer()
    original_modes = [module.training for module in model.modules()]
    original_gradients = [parameter.requires_grad for parameter in model.parameters()]
    try:
        with frozen_validation_codec(model):
            assert not any(module.training for module in model.modules())
            assert not any(parameter.requires_grad for parameter in model.parameters())
            assert model.quantizer.num_codebooks == 4
            raise RuntimeError("synthetic validation failure")
    except RuntimeError as error:
        assert "synthetic validation failure" in str(error)
    assert [module.training for module in model.modules()] == original_modes
    assert [parameter.requires_grad for parameter in model.parameters()] == original_gradients
    assert model.quantizer.num_codebooks == 1


def test_complete_validation_pipeline_preserves_model_and_selects_best():
    import h5py
    from scripts.codec_reconstruction_audit import _fingerprint
    from scripts.smoke_test_codec_reconstruction_audit import _IdentityFaceCodec, _make_hdf5
    from scripts.trainers.dataloaders.unified_dataset import UNIFIEDDataset
    from scripts.trainers.utils.codec_reconstruction_validation import run_standard_codec_validation
    from scripts.trainers.utils.tools import EpochTracker

    class CountingDataset(UNIFIEDDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            self.visited.append(sample["filechunk_id"])
            return sample

    class ErrorCodec(_IdentityFaceCodec):
        error_scale = 1.0
        def decode(self, codes):
            decoded = super().decode(codes)
            decoded[..., 6:] += self.error_scale * self.last_values[..., 6:]
            return decoded

    with TemporaryDirectory(prefix="scott-codec-validation-") as directory:
        folder = Path(directory)
        data_path = folder / "tiny_beatx.h5"
        ids = _make_hdf5(data_path, frames=8, invalid_val=True, extra_scott_train=True)
        with h5py.File(data_path, "a") as file:
            file["chunk_splits"][2] = "val"
            file[ids[2]].attrs["split"] = "val"
        args = _parse("face")
        args.pose_length, args.codec_eval_fgd = 8, False
        args.beatx_cache_path, args.embody3d_cache_path = str(data_path), None
        args.index_cache_dir = str(folder / "index")
        dataset = CountingDataset(args, "val", only_motion=True, dataset_ratio=args.dataset_ratio,
                                  varying_frame_length=True)
        dataset.visited = []
        # The runner reads complete items directly; the dataset's collate
        # variation flag cannot crop either validation clip.
        model = ErrorCodec().train()
        trainer = SimpleNamespace(
            model=model, val_data=dataset, args=args, global_rank=0,
            checkpoint_path=str(folder / "run"), tracker=EpochTracker(["rec_6d"], [False]),
            wandb_logger=SimpleNamespace(log_validation=Mock()), writer=SimpleNamespace(add_scalar=Mock()),
        )
        before = _fingerprint(model)
        original_gradients = [p.requires_grad for p in model.parameters()]
        try:
            with patch("scripts.trainers.utils.codec_reconstruction_validation._BEATXReconstructionFGD") as fgd:
                metrics = run_standard_codec_validation(trainer, "face", 10)
                fgd.assert_not_called()
            assert dataset.visited == [ids[1], ids[2]]
            assert model.entries == 2 and model._streaming_state is None
            assert metrics["clip_count"] == 2 and metrics["frame_count"] == 16
            # The two eight-frame expression targets are 1 and 2. Every frame,
            # including the first clip's flagged frame 3, contributes once.
            _close(metrics["expression_mse"], 2.5)
            _close(metrics["expression_mae"], 1.5)
            _close(metrics["expression_rmse"], math.sqrt(2.5))
            assert _fingerprint(model) == before
            assert model.training and [p.requires_grad for p in model.parameters()] == original_gradients
            assert trainer._codec_fgd_status == "disabled"
            trainer.wandb_logger.log_validation.assert_called_once()
            assert all(math.isfinite(float(call.args[1])) for call in trainer.writer.add_scalar.call_args_list)
            best_path = Path(trainer.checkpoint_path) / "best_reconstruction.safetensors"
            saved = best_path.read_bytes()
            dataset.visited.clear()
            model.error_scale = 2.0
            worse = run_standard_codec_validation(trainer, "face", 20)
            _close(worse["expression_mse"], 10)
            assert best_path.read_bytes() == saved
            assert trainer._codec_best_reconstruction["epoch"] == 10
            assert dataset.visited == [ids[1], ids[2]] and model.entries == 4
            assert _fingerprint(model) == before and model.training
        finally:
            dataset.close()


def test_optional_fgd_preserves_reconstructed_jaw_and_uses_real_moments():
    import warnings
    from scripts.trainers.utils import rotation_conversions as rc
    from scripts.trainers.utils.codec_reconstruction_validation import _BEATXReconstructionFGD
    from scripts.trainers.utils.metrics import FIDCalculator, match_motion_fps

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []
        def map2latent(self, values):
            self.seen.append(values.detach().clone())
            return values[..., :240].reshape(1, 4, 8, 240).mean(2)

    evaluator = _BEATXReconstructionFGD.__new__(_BEATXReconstructionFGD)
    evaluator.device, evaluator.source_fps = torch.device("cpu"), 25
    evaluator._resample = match_motion_fps
    evaluator._distance = FIDCalculator.calculate_frechet_distance
    evaluator.encoder = Encoder()
    evaluator.target, evaluator.predicted, evaluator.trimmed_frames = [], [], 0
    pose = torch.zeros(40, 55, 3)
    decoded = torch.zeros(1, 40, 106)
    jaw = torch.zeros(40, 1, 3)
    jaw[..., 0] = .2
    decoded[0, :, :6] = rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(jaw)).reshape(40, 6)
    mask = torch.zeros(55, dtype=torch.bool)
    mask[22] = True
    evaluator.add({"motion": pose, "dataset_name": "BEATX"}, "face", decoded,
                  SimpleNamespace(face_mask=mask))
    target, predicted = evaluator.encoder.seen
    assert target.shape == predicted.shape == (1, 32, 330)
    assert evaluator.trimmed_frames == 15  # Native 40 frames -> 47 at 30 fps -> 32.
    assert not torch.allclose(target[..., 132:138], predicted[..., 132:138])
    keep = torch.ones(330, dtype=torch.bool)
    keep[132:138] = False
    _close(target[..., keep], predicted[..., keep])
    assert not bool(pose.any()), "FGD must not modify the ground-truth source motion"
    # Constant synthetic motion intentionally has singular covariance.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Matrix is singular.*")
        score = evaluator.compute()
    assert math.isfinite(score) and score > 0
    with patch.object(evaluator, "_distance", side_effect=ValueError("synthetic invalid moments")):
        try:
            evaluator.compute()
        except ValueError as error:
            assert "synthetic invalid moments" in str(error)
        else:
            raise AssertionError("Invalid FGD moments became a fabricated numeric score")


def main():
    torch.set_num_threads(2)
    checks = [test_scott_training_splits, test_fresh_models_match_release_checkpoint_shapes,
              test_scratch_reconstruction_trains_backbone_and_rvq,
              test_launcher_commands_are_scoped_and_fresh,
              test_launcher_dry_run_and_failure_stop,
              test_final_validation_is_opt_in,
              test_standard_metric_units_pooling_and_shape_guards,
              test_best_checkpoint_is_separate_and_updates_only_on_improvement,
              test_validation_restores_training_state_even_on_failure,
              test_complete_validation_pipeline_preserves_model_and_selects_best,
              test_optional_fgd_preserves_reconstructed_jaw_and_uses_real_moments]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"Passed {len(checks)} Scott codec training checks.")


if __name__ == "__main__":
    main()
