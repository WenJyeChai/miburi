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


def _make_hdf5(path: Path, frames: int = 8, invalid_val: bool = True):
    """Use the real UNIFIEDDataset chunk schema and speaker ID convention."""
    ids = ["2_scott_0_1_1_C0", "2_scott_0_2_1_C0", "9_miranda_0_1_1_C0"]
    splits = ["train", "val", "train"]
    speakers = ["scott", "scott", "miranda"]
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
    from scripts import codec_reconstruction_audit as audit
    import yaml

    with TemporaryDirectory(prefix="codec-audit-pipeline-") as directory:
        folder = Path(directory)
        data_path = folder / "tiny_beatx.h5"
        _make_hdf5(data_path, invalid_val=False)
        config = yaml.safe_load((ROOT / audit.DEFAULT_CONFIG).read_text(encoding="utf-8"))
        config["pose_length"] = 8
        config_path = folder / "test_config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        dummy_checkpoint = folder / "toy.safetensors"
        dummy_checkpoint.write_bytes(b"toy checkpoint; loader is replaced in this test")
        settings = audit.AuditSettings(
            lm_config=str(config_path), repository_root=str(ROOT), device="cpu",
            parts=("face",), path_overrides={"beatx_cache_path": str(data_path),
                "embody3d_cache_path": None, "facecodec_ckpt": str(dummy_checkpoint)},
        )
        model = _IdentityFaceCodec().eval().requires_grad_(False)
        with patch.object(audit, "load_codecs", return_value={"face": model}):
            result = audit.run_audit(settings)
        assert result["metadata"]["state_unchanged"]
        assert len(result["metadata"]["sample_manifest"]) == 2
        assert all(row["value"] == 0 for row in result["summary_records"])
        assert len(result["codebook_records"]) == 8
        for row in result["codebook_records"]:
            assert row["token_count"] == 4
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
        assert not list(folder.glob("*.chunk_index*"))
        mutating = _IdentityFaceCodec(mutate=True).eval().requires_grad_(False)
        with patch.object(audit, "load_codecs", return_value={"face": mutating}):
            try:
                audit.run_audit(settings)
            except RuntimeError as error:
                assert "parameters/buffers changed" in str(error)
            else:
                raise AssertionError("Codebook buffer mutation went undetected")
        with h5py.File(data_path, "a") as file:
            file["2_scott_0_2_1_C0"]["pose_valid"][3] = False
        with patch.object(audit, "load_codecs", return_value={"face": model}):
            try:
                audit.run_audit(settings)
            except ValueError as error:
                assert "refuses replacement" in str(error)
            else:
                raise AssertionError("Invalid pose data was accepted or substituted")


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
    ]
    if args.release_codecs:
        checks.append(test_release_codec_reconstruction)
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"Passed {len(checks)} codec reconstruction audit checks.")


if __name__ == "__main__":
    main()
