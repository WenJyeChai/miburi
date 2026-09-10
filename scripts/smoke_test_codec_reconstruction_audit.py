"""Analytic CPU checks for the codec reconstruction notebook's helpers.

Run ``python -B -m scripts.smoke_test_codec_reconstruction_audit``.
The optional ``--release-codecs`` check loads the current three local codec
checkpoints; the default suite uses only generated data and tiny toy modules.
These checks validate metric arithmetic and evaluation plumbing, not the
reconstruction quality of real Scott motion.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def _close(actual, expected, *, atol=1e-6):
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.as_tensor(expected, dtype=torch.float64),
        atol=atol,
        rtol=0,
    )


def _make_hdf5(path: Path, frames: int = 8, invalid_val: bool = True,
               extra_scott_train: bool = False):
    """Use the real UNIFIEDDataset chunk schema and speaker ID convention."""
    ids = ["2_scott_0_1_1_C0", "2_scott_0_2_1_C0", "9_miranda_0_1_1_C0"]
    splits = ["train", "val", "train"]
    speakers = ["scott", "scott", "miranda"]
    if extra_scott_train:
        ids[2], speakers[2] = "2_scott_0_3_1_C0", "scott"
    with h5py.File(path, "w") as file:
        text_dtype = h5py.string_dtype("utf-8")
        for key, values in {
            "chunk_ids": ids,
            "chunk_splits": splits,
            "chunk_relpaths": [f"{name}.npz" for name in ids],
            "chunk_speaker_ids": speakers,
        }.items():
            file.create_dataset(key, data=np.asarray(values, dtype=object), dtype=text_dtype)
        file.create_dataset("chunk_is_sitting", data=np.zeros(len(ids), dtype=bool))
        for index, (name, split) in enumerate(zip(ids, splits)):
            group = file.create_group(name)
            group.attrs["file_id"] = name.rsplit("_C", 1)[0]
            group.attrs["split"] = split
            group.attrs["chunk_startsec"] = 0.0
            group.attrs["chunk_endsec"] = frames / 25
            group.create_dataset("motion", data=np.zeros((frames, 55, 3), np.float32))
            group.create_dataset("transl", data=np.zeros((frames, 3), np.float32))
            group.create_dataset("betas", data=np.zeros(300, np.float32))
            group.create_dataset("expressions", data=np.full((frames, 100), index, np.float32))
            group.create_dataset("contacts", data=np.zeros((frames, 4), np.float32))
            valid = np.ones(frames, dtype=bool)
            if split == "val" and invalid_val:
                valid[3] = False
            group.create_dataset("pose_valid", data=valid)
    return ids


class _StatefulToyCodec(nn.Module):
    """Fails if evaluation enables gradients, training, or leaks clip state."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("codebook", torch.arange(4, dtype=torch.float32))
        self.quantizer = SimpleNamespace(num_codebooks=1, cardinality=4)
        self.channels = 1
        self.frame_chunk_size = 2
        self.num_codebooks = 1
        self.cardinality = 4
        self._streaming_state = None
        self.entries = 0
        self.offset = 0

    @contextmanager
    def streaming(self, batch_size):
        assert self._streaming_state is None
        self._streaming_state = batch_size
        self.offset = 0
        self.entries += 1
        try:
            yield self
        finally:
            self._streaming_state = None
            self.offset = 0

    def encode(self, values):
        assert not self.training, "codec must be in evaluation mode"
        assert not torch.is_grad_enabled(), "codec evaluation must disable gradients"
        assert self._streaming_state is not None, "streaming mode required"
        # Repeat each code into a two-frame reconstruction; stateful output makes
        # accidental reuse of one clip's streaming context observable.
        codes = (values[:, ::2, 0].long() + self.offset).unsqueeze(1)
        self.offset += 1
        return codes

    def decode(self, codes):
        assert not self.training and not torch.is_grad_enabled()
        return codes.transpose(1, 2).float().repeat_interleave(2, dim=1)


class _IdentityFaceCodec(_StatefulToyCodec):
    """Preserves motion while assigning independently known token histograms."""

    def __init__(self, mutate=False):
        super().__init__()
        self.channels = 106
        self.num_codebooks = 4
        self.frame_rate = 12.5
        self.latent_dtype = torch.float32
        self.mutate = mutate

    def encode(self, values):
        assert not self.training and not torch.is_grad_enabled()
        if self.mutate:
            self.codebook.add_(1)
        self.last_values = values.clone()
        sequence = [0, 0, 1, 1] if values[0, 0, -1] == 0 else [1, 1, 2, 3]
        code = sequence[self.offset]
        self.offset += 1
        return torch.full((values.shape[0], self.num_codebooks, 1), code, dtype=torch.long)

    def decode(self, codes):
        assert not self.training and not torch.is_grad_enabled()
        return self.last_values.clone()


class _FlaggedFrameErrorFaceCodec(_IdentityFaceCodec):
    def decode(self, codes):
        decoded = super().decode(codes)
        if self.offset == 2:
            decoded[:, 1, 6:] += 8  # Global frame 3, expression coordinates only.
        return decoded


class _FlaggedFrameErrorLowerCodec(_IdentityFaceCodec):
    def __init__(self):
        super().__init__()
        self.channels = 61
        self.num_codebooks = 8

    def decode(self, codes):
        decoded = super().decode(codes)
        if self.offset == 2:
            decoded[:, 1, 54:57] += 8  # Frame 3, lower root velocity only.
        return decoded


@contextmanager
def _audit_case(*, invalid_val=False, extra_scott_train=False, splits=("train", "val"),
                parts=("face",)):
    from scripts import codec_reconstruction_audit as audit
    import yaml

    with TemporaryDirectory(prefix="codec-audit-pipeline-") as directory:
        folder = Path(directory)
        data_path = folder / "tiny_beatx.h5"
        ids = _make_hdf5(data_path, invalid_val=invalid_val, extra_scott_train=extra_scott_train)
        config = yaml.safe_load((ROOT / audit.DEFAULT_CONFIG).read_text(encoding="utf-8"))
        config["pose_length"] = 8
        config_path = folder / "test_config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        dummy_checkpoint = folder / "toy.safetensors"
        dummy_checkpoint.write_bytes(b"toy checkpoint; loader is replaced in this test")
        settings = audit.AuditSettings(
            lm_config=str(config_path), repository_root=str(ROOT), device="cpu", splits=splits,
            parts=parts, path_overrides={"beatx_cache_path": str(data_path),
                "embody3d_cache_path": None, "facecodec_ckpt": str(dummy_checkpoint),
                "lowerbodycodec_ckpt": str(dummy_checkpoint)},
        )
        yield audit, settings, data_path, ids


def _run_toy(audit, settings, model=None):
    model = model or _IdentityFaceCodec()
    model.eval().requires_grad_(False)
    assert len(settings.parts) == 1
    with patch.object(audit, "load_codecs", return_value={settings.parts[0]: model}):
        return audit.run_audit(settings)


def _summary(result, split, scope, metric="expression_rmse"):
    rows = [row for row in result["summary_records"] if row["split"] == split
            and row["metric_scope"] == scope and row["metric"] == metric]
    assert len(rows) == 1, rows
    return rows[0]


def test_unified_dataset_fixture():
    from scripts.trainers.dataloaders.unified_dataset import UNIFIEDDataset

    with TemporaryDirectory(prefix="codec-audit-smoke-") as directory:
        folder = Path(directory)
        path = folder / "tiny_beatx.h5"
        ids = _make_hdf5(path)
        args = SimpleNamespace(
            beatx_cache_path=str(path), embody3d_cache_path=None,
            index_cache_dir=str(folder / "index"), motion_fps=25,
            frame_chunk_size=2, pose_length=8, body_part="full",
        )
        for split, expected in (("train", ids[0]), ("val", ids[1])):
            dataset = UNIFIEDDataset(
                args, split, only_motion=True,
                dataset_ratio="scott_beatx_lowervalid", varying_frame_length=False,
            )
            try:
                assert len(dataset) == 1, "speaker/split filtering must select only Scott"
                item = dataset[0]
                assert item["filechunk_id"] == expected
                assert item["motion_face"].shape == (8, 1, 3)
                assert item["expressions"].shape == (8, 100)
                assert item["audio_tokens"] is None
            finally:
                for handle in dataset._h5_handles.values():
                    handle.close()


def test_rotation_geodesic():
    from scripts.codec_reconstruction_metrics import finalize_metrics, reconstruction_metrics

    identity = torch.eye(3, dtype=torch.float64)
    rotations = torch.stack((
        identity,
        torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
        torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]),
    )).unsqueeze(0)
    stats = reconstruction_metrics(
        target_rotations=identity.expand(1, 3, 3, 3),
        reconstructed_rotations=rotations,
        rotation_regions={"identity": [0], "quarter_turn": [1], "half_turn": [2]},
    )
    values = finalize_metrics(stats)
    _close(values["identity_rotation_geodesic_deg"], 0)
    _close(values["quarter_turn_rotation_geodesic_deg"], 90)
    _close(values["half_turn_rotation_geodesic_deg"], 180)


def test_position_and_true_velocity_error():
    from scripts.codec_reconstruction_metrics import (
        finalize_metrics, geometry_metrics, reconstruction_metrics,
    )

    target = torch.arange(5, dtype=torch.float64)[:, None].expand(-1, 3) / 25
    predicted = target + torch.tensor([.003, .004, 0.], dtype=torch.float64)
    values = finalize_metrics(reconstruction_metrics(
        target_translation=target, reconstructed_translation=predicted, fps=25,
    ))
    _close(values["translation_rmse_m"], .005 / math.sqrt(3))
    _close(values["translation_velocity_rmse_m_s"], 0)
    vertices = target[:, None].expand(-1, 2, -1)
    predicted_vertices = predicted[:, None].expand(-1, 2, -1)
    geometry = finalize_metrics(geometry_metrics(
        target_joints=vertices, reconstructed_joints=predicted_vertices,
        target_face_vertices=vertices, reconstructed_face_vertices=predicted_vertices,
        fps=25,
    ))
    _close(geometry["all_mpjpe_mm"], 5)
    _close(geometry["face_vertices_mean_vertex_l2_mm"], 5)
    _close(geometry["face_vertices_coordinate_rmse_mm"], 5 / math.sqrt(3))
    _close(geometry["face_vertices_velocity_coordinate_rmse_mm_s"], 0)


def test_masked_derivatives_and_frame_weighted_pooling():
    from scripts.codec_reconstruction_metrics import (
        finalize_metrics, merge_metric_sums, reconstruction_metrics,
    )

    # A discontinuous position offset on the far side of a missing frame must
    # not be interpreted as a velocity spike across that gap.
    target = torch.zeros(5, 3, dtype=torch.float64)
    predicted = torch.tensor([[0.] * 3, [0.] * 3, [float("nan")] * 3,
                              [10.] * 3, [10.] * 3], dtype=torch.float64)
    stats = reconstruction_metrics(
        target_translation=target, reconstructed_translation=predicted,
        valid_mask=[True, True, False, True, True], warmup_frames=1, fps=25,
    )
    assert stats["translation_rmse_m"]["count"] == 9
    assert stats["translation_velocity_rmse_m_s"]["count"] == 3
    _close(finalize_metrics(stats)["translation_velocity_rmse_m_s"], 0)

    # One frame with error 1 and three frames with error 3 pool to sqrt(7),
    # not the unweighted mean of per-clip RMSEs (2).
    totals = {}
    for frames, error in ((1, 1.), (3, 3.)):
        metric = reconstruction_metrics(
            target_expressions=torch.zeros(frames, 2),
            reconstructed_expressions=torch.full((frames, 2), error),
        )
        merge_metric_sums(totals, metric)
    assert totals["expression_rmse"]["count"] == 8
    _close(finalize_metrics(totals)["expression_rmse"], math.sqrt(7))
    empty = reconstruction_metrics(
        target_translation=torch.zeros(1, 3),
        reconstructed_translation=torch.zeros(1, 3), valid_mask=[False],
    )
    assert all(value is None for value in finalize_metrics(empty).values())


def test_frozen_streaming_resets_and_input_guards():
    from scripts.codec_reconstruction_audit import _fingerprint, reconstruct_codec

    model = _StatefulToyCodec()
    values = torch.zeros(1, 8, 1)
    try:
        reconstruct_codec(model, values)
    except ValueError as error:
        assert "frozen" in str(error)
    else:
        raise AssertionError("Training-mode codec was accepted")
    model.eval().requires_grad_(False)
    before = _fingerprint(model)
    first, first_codes = reconstruct_codec(model, values)
    second, second_codes = reconstruct_codec(model, values)
    _close(first_codes, [[[0, 1, 2, 3]]])
    _close(first, second)
    _close(first_codes, second_codes)
    assert model.entries == 2 and model._streaming_state is None
    assert _fingerprint(model) == before
    for invalid in (values[:, :7], torch.zeros(1, 8, 2)):
        try:
            reconstruct_codec(model, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid frame/feature shape was silently accepted")


def test_full_audit_usage_and_codebook_mutation_guard():
    with _audit_case() as (audit, settings, data_path, _):
        model = _IdentityFaceCodec().eval().requires_grad_(False)
        with patch.object(audit, "load_codecs", return_value={"face": model}):
            result = audit.run_audit(settings)
        assert result["metadata"]["state_unchanged"]
        assert len(result["metadata"]["sample_manifest"]) == 2
        assert all(row["value"] == 0 for row in result["summary_records"])
        assert len(result["codebook_records"]) == 8
        for row in result["codebook_records"]:
            assert row["token_count"] == 4
            assert row["usage_scope"] == "all_evaluated_frames"
            if row["split"] == "train":
                assert row["active_codes"] == 2
                _close(row["entropy_nats"], math.log(2))
                _close(row["perplexity"], 2)
                _close(row["unseen_train_mass"], 0)
            else:
                assert row["active_codes"] == 3
                _close(row["entropy_nats"], 1.5 * math.log(2))
                _close(row["perplexity"], math.sqrt(8))
                _close(row["unseen_train_mass"], .5)
        # The audit must not write an index cache beside the source HDF5.
        assert not list(data_path.parent.glob("*.chunk_index*"))
        mutating = _IdentityFaceCodec(mutate=True).eval().requires_grad_(False)
        with patch.object(audit, "load_codecs", return_value={"face": mutating}):
            try:
                audit.run_audit(settings)
            except RuntimeError as error:
                assert "parameters/buffers changed" in str(error)
            else:
                raise AssertionError("Codebook buffer mutation went undetected")


def test_finite_flags_change_scoring_without_changing_reconstruction():
    with _audit_case(invalid_val=True) as (audit, settings, data_path, ids):
        model = _FlaggedFrameErrorFaceCodec()
        result = _run_toy(audit, settings, model)
        assert model.entries == 2, "Metric scopes must not reconstruct a clip repeatedly"
        assert not result["skipped_records"]
        _close(_summary(result, "val", "all_frames")["value"], math.sqrt(8))
        valid = _summary(result, "val", "valid_frames")
        _close(valid["value"], 0)
        assert valid["count"] == 700
        assert not [row for row in result["summary_records"]
                    if row["split"] == "val" and row["metric_scope"] == "fully_valid_clips"]
        train_full = _summary(result, "train", "fully_valid_clips")
        assert train_full["count"] == 800
        quality = {row["split"]: row for row in result["quality_records"]}
        assert quality["val"]["filechunk_id"] == ids[1]
        assert quality["val"]["status"] == "evaluated"
        assert quality["val"]["valid_frames"] == 7
        assert quality["val"]["flagged_frames"] == 1
        _close(quality["val"]["valid_fraction"], 7 / 8)
        assert quality["train"]["fully_valid"] and not quality["val"]["fully_valid"]
        assert len(result["codebook_records"]) == 8
        assert all(row["token_count"] == 4 for row in result["codebook_records"])

        # Warmup affects denominators but never reclassifies a raw flagged clip
        # as fully valid, even if every flagged frame is inside the warmup.
        with h5py.File(data_path, "a") as file:
            file[ids[1]]["pose_valid"][:] = [False, True, True, True, True, True, True, True]
        settings.warmup_frames = 2
        warmed = _run_toy(audit, settings)
        assert _summary(warmed, "val", "valid_frames")["count"] == 600
        assert _summary(warmed, "val", "all_frames")["count"] == 600
        assert not [row for row in warmed["summary_records"]
                    if row["split"] == "val" and row["metric_scope"] == "fully_valid_clips"]
        assert all(row["token_count"] == 4 for row in warmed["codebook_records"])


def test_all_flagged_frames_report_empty_metrics_and_quality():
    with _audit_case(splits=("val",)) as (audit, settings, data_path, ids):
        with h5py.File(data_path, "a") as file:
            file[ids[1]]["pose_valid"][:] = False
        model = _FlaggedFrameErrorFaceCodec()
        result = _run_toy(audit, settings, model)
        assert model.entries == 1
        valid = _summary(result, "val", "valid_frames")
        assert valid["count"] == 0 and valid["value"] is None
        _close(_summary(result, "val", "all_frames")["value"], math.sqrt(8))
        assert not [row for row in result["summary_records"]
                    if row["metric_scope"] == "fully_valid_clips"]
        quality = result["quality_summary_records"][0]
        assert quality["evaluated_clips"] == 1 and quality["skipped_clips"] == 0
        assert quality["valid_frames"] == 0 and quality["flagged_frames"] == 8
        assert quality["valid_fraction"] == 0
        assert all(row["token_count"] == 4 and row["unseen_train_mass"] is None
                   for row in result["codebook_records"])


def test_direct_velocity_target_stencil_excludes_flagged_neighbors():
    with _audit_case(invalid_val=True, splits=("val",), parts=("lower",)) as (
        audit, settings, data_path, ids,
    ):
        result = _run_toy(audit, settings, _FlaggedFrameErrorLowerCodec())
        metric = "translation_velocity_direct_rmse_m_s"
        # The velocity target at an interior t uses translations t-1 and t+1;
        # validity additionally requires t. Flag 3 excludes direct targets 2,3,4.
        all_frames = _summary(result, "val", "all_frames", metric)
        valid = _summary(result, "val", "valid_frames", metric)
        assert all_frames["count"] == 8 * 3
        assert valid["count"] == 5 * 3
        _close(all_frames["value"], math.sqrt(8))
        _close(valid["value"], 0)
        assert _summary(result, "val", "valid_frames", "translation_velocity_rmse_m_s")["count"] == 5 * 3

        settings.warmup_frames = 2
        warmed = _run_toy(audit, settings, _FlaggedFrameErrorLowerCodec())
        assert _summary(warmed, "val", "valid_frames", metric)["count"] == 3 * 3
        assert _summary(warmed, "val", "all_frames", metric)["count"] == 5 * 3
        with h5py.File(data_path, "a") as file:
            file[ids[1]]["pose_valid"][:] = True
        all_valid_warmed = _run_toy(audit, settings, _FlaggedFrameErrorLowerCodec())
        for scope in ("all_frames", "valid_frames", "fully_valid_clips"):
            row = _summary(all_valid_warmed, "val", scope, metric)
            assert row["count"] == 5 * 3
            _close(row["value"], _summary(all_valid_warmed, "val", "all_frames", metric)["value"])
        # Endpoint derivatives require both endpoint and its single neighbor.
        settings.warmup_frames = 0
        with h5py.File(data_path, "a") as file:
            file[ids[1]]["pose_valid"][:] = [False, True, True, True, True, True, True, False]
        endpoints = _run_toy(audit, settings, _FlaggedFrameErrorLowerCodec())
        assert _summary(endpoints, "val", "valid_frames", metric)["count"] == 4 * 3


def test_nonfinite_clips_are_logged_without_replacement():
    with _audit_case(extra_scott_train=True, splits=("train",)) as (audit, settings, data_path, ids):
        with h5py.File(data_path, "a") as file:
            file[ids[0]]["motion"][3, 22, 0] = float("nan")
        model = _IdentityFaceCodec()
        result = _run_toy(audit, settings, model)
        assert model.entries == 1
        skipped = result["skipped_records"]
        assert len(skipped) == 1 and skipped[0]["filechunk_id"] == ids[0]
        assert skipped[0]["status"] == "skipped" and "motion" in skipped[0]["nonfinite_fields"]
        assert skipped[0]["reason"]
        assert {row["filechunk_id"] for row in result["sample_records"]} == {ids[2]}
        quality = result["quality_summary_records"][0]
        assert quality["available_clips"] == quality["selected_clips"] == 2
        assert quality["evaluated_clips"] == quality["skipped_clips"] == 1
        assert quality["selected_frames"] == 16 and quality["evaluated_frames"] == 8
        assert quality["valid_frames"] == 8 and quality["flagged_frames"] == 0

        # Every selected clip nonfinite: return a useful quality report with no
        # fabricated metrics. A sample cap counts selected clips, not successes.
        with h5py.File(data_path, "a") as file:
            file[ids[2]]["expressions"][2, 0] = float("inf")
        model = _IdentityFaceCodec()
        all_skipped = _run_toy(audit, settings, model)
        assert model.entries == 0
        assert not all_skipped["summary_records"] and not all_skipped["sample_records"]
        assert not all_skipped["codebook_records"] and len(all_skipped["skipped_records"]) == 2
        quality = all_skipped["quality_summary_records"][0]
        assert quality["selected_clips"] == quality["skipped_clips"] == 2
        assert quality["evaluated_frames"] == quality["valid_frames"] == quality["flagged_frames"] == 0
        assert quality["valid_fraction"] is None
        settings.max_samples_per_split = 1
        capped = _run_toy(audit, settings)
        assert len(capped["quality_records"]) == len(capped["skipped_records"]) == 1
        assert capped["quality_summary_records"][0]["selected_clips"] == 1
        assert capped["quality_summary_records"][0]["available_clips"] == 2
        with h5py.File(data_path, "a") as file:
            file[ids[0]]["motion"][:] = 0
            file[ids[2]]["expressions"][:] = 2
        baseline_cap = _run_toy(audit, settings)
        selected_id = baseline_cap["quality_records"][0]["filechunk_id"]
        with h5py.File(data_path, "a") as file:
            file[selected_id]["motion"][0, 0, 0] = float("nan")
        model = _IdentityFaceCodec()
        capped_bad = _run_toy(audit, settings, model)
        assert model.entries == 0, "The other valid clip must not replace a capped nonfinite selection"
        assert capped_bad["skipped_records"][0]["filechunk_id"] == selected_id
        assert len(capped_bad["quality_records"]) == 1


def test_malformed_inputs_and_model_errors_stay_fatal():
    for problem in ("missing", "shape", "read", "mask_values", "model"):
        with _audit_case(splits=("val",)) as (audit, settings, data_path, ids):
            with h5py.File(data_path, "a") as file:
                if problem == "missing":
                    del file[ids[1]]["expressions"]
                elif problem == "shape":
                    del file[ids[1]]["expressions"]
                    file[ids[1]].create_dataset("expressions", data=np.zeros((8, 99), np.float32))
                    file[ids[1]]["motion"][0, 0, 0] = float("nan")
                elif problem == "read":
                    del file[ids[1]]
                elif problem == "mask_values":
                    del file[ids[1]]["pose_valid"]
                    file[ids[1]].create_dataset("pose_valid", data=np.full(8, 2, np.int32))
            model = _IdentityFaceCodec()
            if problem == "model":
                model.encode = lambda inputs: (_ for _ in ()).throw(RuntimeError("sentinel model failure"))
            try:
                _run_toy(audit, settings, model)
            except (KeyError, ValueError, RuntimeError) as error:
                if problem == "model":
                    assert "sentinel model failure" in str(error)
            else:
                raise AssertionError(f"{problem} error was silently skipped")


def test_release_codec_reconstruction():
    from scripts.codec_reconstruction_audit import (
        AuditSettings, PARTS, _data_modules, _fingerprint,
        load_codecs, parse_lm_config, prepare_codec_inputs, reconstruct_codec,
    )

    args = parse_lm_config(AuditSettings(repository_root=str(ROOT), device="cpu"))
    with TemporaryDirectory(prefix="codec-audit-release-") as directory:
        data_path = Path(directory) / "tiny_beatx.h5"
        _make_hdf5(data_path, invalid_val=False)
        args.beatx_cache_path = str(data_path)
        args.embody3d_cache_path = None
        args.pose_length = 8
        dataset_module, _, _ = _data_modules(ROOT)
        dataset = dataset_module.UNIFIEDDataset(
            args, "train", only_motion=True,
            dataset_ratio="scott_beatx_lowervalid", varying_frame_length=False,
        )
        try:
            prepared = prepare_codec_inputs(dataset[0], args, repository_root=ROOT)
            for part in PARTS:
                model = load_codecs(args, device="cpu", parts=(part,))[part]
                assert not model.training and all(not p.requires_grad for p in model.parameters())
                before = _fingerprint(model)
                values = prepared["inputs"][part]
                decoded, codes = reconstruct_codec(model, values)
                repeated, repeated_codes = reconstruct_codec(model, values)
                assert decoded.shape == values.shape
                assert codes.shape == (1, 4 if part == "face" else 8, 4)
                assert bool(torch.isfinite(decoded).all())
                _close(decoded, repeated)
                _close(codes, repeated_codes)
                assert _fingerprint(model) == before
                assert model._streaming_state is None
                print(f"  {part}: real checkpoint, {tuple(decoded.shape)}, frozen and repeatable")
                del model
        finally:
            dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-codecs", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checks = [
        test_rotation_geodesic,
        test_position_and_true_velocity_error,
        test_masked_derivatives_and_frame_weighted_pooling,
        test_unified_dataset_fixture,
        test_frozen_streaming_resets_and_input_guards,
        test_full_audit_usage_and_codebook_mutation_guard,
        test_finite_flags_change_scoring_without_changing_reconstruction,
        test_all_flagged_frames_report_empty_metrics_and_quality,
        test_direct_velocity_target_stencil_excludes_flagged_neighbors,
        test_nonfinite_clips_are_logged_without_replacement,
        test_malformed_inputs_and_model_errors_stay_fatal,
    ]
    if args.release_codecs:
        checks.append(test_release_codec_reconstruction)
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"Passed {len(checks)} codec reconstruction audit checks.")


if __name__ == "__main__":
    main()
