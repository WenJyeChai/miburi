"""Focused CPU checks for raw reset-suffix future-teacher construction.

These tests use a deterministic lightweight codec double so they isolate the
information boundary itself.  A remote checkpoint smoke run should additionally
exercise the same helpers with the three released gesture codecs.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from miburi.models.gesture_lm_reset_future import (
    build_reset_future_teacher_inputs,
    build_reset_future_teacher_inputs_from_codes,
    encode_reset_suffix,
)
from trainers.dataloaders.utils.reset_future_cache import (
    MANIFEST_NAME,
    METADATA_NAME,
    RESET_FUTURE_CACHE_SCHEMA,
    RESET_FUTURE_CACHE_VERSION,
    ResetFutureManifestCache,
    build_reset_future_cache_signature,
    reset_future_cache_signatures_match,
)


class _DeterministicQuantizer:
    def __init__(self, num_codebooks: int, cardinality: int = 31):
        self.num_codebooks = num_codebooks
        self.cardinality = cardinality

    def encode(self, latent):
        base = torch.round(latent.sum(dim=1) * 100).long()
        return torch.stack(
            [
                (base + codebook_index * 7) % self.cardinality
                for codebook_index in range(self.num_codebooks)
            ],
            dim=1,
        )


class _DeterministicFreshCodec:
    def __init__(self, num_codebooks: int, frame_size: int = 2):
        self.num_codebooks = num_codebooks
        self.frame_size = frame_size
        self.quantizer = _DeterministicQuantizer(num_codebooks)
        self.training = True
        self.active_state = None

    def eval(self):
        self.training = False
        return self

    def get_streaming_state(self):
        return {"": self.active_state}

    def encode_to_latent(self, motion, quantize=False):
        assert not quantize
        batch, frames, features = motion.shape
        assert frames % self.frame_size == 0
        return (
            motion.reshape(
                batch,
                frames // self.frame_size,
                self.frame_size,
                features,
            )
            .mean(dim=2)
            .transpose(1, 2)
        )

    @staticmethod
    def _cast_for_module(latent, module):
        del module
        return latent


def _motion(batch=2, frames=16, features=8):
    values = torch.arange(
        batch * frames * features,
        dtype=torch.float32,
    )
    return values.reshape(batch, frames, features) / 1000


def test_hidden_interval_and_past_invariance():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    motion = _motion(batch=1)
    suffix_start = 8
    reference = encode_reset_suffix(
        codec,
        motion,
        suffix_start_frame=suffix_start,
    )

    variants = []
    zeroed = motion.clone()
    zeroed[:, :suffix_start] = 0
    variants.append(zeroed)
    randomized = motion.clone()
    randomized[:, :suffix_start] = torch.randn_like(
        randomized[:, :suffix_start]
    )
    variants.append(randomized)
    replaced = motion.clone()
    replaced[:, :suffix_start] = _motion(batch=1)[
        :,
        :suffix_start,
    ].flip(1)
    variants.append(replaced)

    for variant in variants:
        encoded = encode_reset_suffix(
            codec,
            variant,
            suffix_start_frame=suffix_start,
        )
        torch.testing.assert_close(
            encoded.prequantized_latent,
            reference.prequantized_latent,
        )
        torch.testing.assert_close(encoded.codes, reference.codes)


def test_future_sensitivity_and_cache_contamination():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    first = _motion(batch=1)
    second = first.clone()
    second[:, 10:] += 0.25

    first_a = encode_reset_suffix(
        codec,
        first,
        suffix_start_frame=8,
    )
    second_encoded = encode_reset_suffix(
        codec,
        second,
        suffix_start_frame=8,
    )
    first_b = encode_reset_suffix(
        codec,
        first,
        suffix_start_frame=8,
    )

    assert not torch.equal(first_a.codes, second_encoded.codes)
    assert not torch.equal(
        first_a.prequantized_latent,
        second_encoded.prequantized_latent,
    )
    torch.testing.assert_close(first_a.codes, first_b.codes)
    torch.testing.assert_close(
        first_a.prequantized_latent,
        first_b.prequantized_latent,
    )


def test_batched_reset_matches_individual_resets():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    motion = _motion(batch=3, frames=12)
    batched = encode_reset_suffix(
        codec,
        motion,
        suffix_start_frame=0,
        maximum_suffix_frames=8,
    )
    assert batched.codes.shape == (3, 2, 4)
    for sample_index in range(motion.shape[0]):
        individual = encode_reset_suffix(
            codec,
            motion[sample_index:sample_index + 1],
            suffix_start_frame=0,
            maximum_suffix_frames=8,
        )
        torch.testing.assert_close(
            batched.codes[sample_index:sample_index + 1],
            individual.codes,
        )
        torch.testing.assert_close(
            batched.prequantized_latent[
                sample_index:sample_index + 1
            ],
            individual.prequantized_latent,
        )


def test_right_padded_reset_prefix_matches_individual_suffix():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    long_window = _motion(batch=1, frames=12)[0]
    short_window = _motion(batch=1, frames=6)[0]
    padded = torch.nn.utils.rnn.pad_sequence(
        [long_window, short_window],
        batch_first=True,
    )
    batched = encode_reset_suffix(
        codec,
        padded,
        suffix_start_frame=0,
    )
    short = encode_reset_suffix(
        codec,
        short_window.unsqueeze(0),
        suffix_start_frame=0,
    )
    torch.testing.assert_close(
        batched.codes[1:2, :, :short.codes.shape[-1]],
        short.codes,
    )


def test_active_streaming_state_is_rejected():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    codec.active_state = object()
    try:
        encode_reset_suffix(
            codec,
            _motion(batch=1),
            suffix_start_frame=8,
        )
    except RuntimeError as exc:
        assert "active streaming state" in str(exc)
    else:
        raise AssertionError("A contaminated codec state was accepted.")


def test_lower_boundary_velocity_is_internal_to_suffix():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    motion = _motion(batch=1)
    changed = motion.clone()
    changed[:, 8, 1:4] += 1000
    reference = encode_reset_suffix(
        codec,
        motion,
        suffix_start_frame=8,
        zero_first_frame_feature_slice=(1, 4),
    )
    encoded = encode_reset_suffix(
        codec,
        changed,
        suffix_start_frame=8,
        zero_first_frame_feature_slice=(1, 4),
    )
    torch.testing.assert_close(
        encoded.prequantized_latent,
        reference.prequantized_latent,
    )
    torch.testing.assert_close(encoded.codes, reference.codes)


def test_all_body_parts_are_reset_and_boundaries_align():
    codecs = (
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(1),
    )
    motions = (
        _motion(features=8),
        _motion(features=8) + 0.1,
        _motion(features=8) + 0.2,
    )
    batch = motions[0].shape[0]
    steps = motions[0].shape[1] // codecs[0].frame_size
    codebooks = sum(codec.num_codebooks for codec in codecs)
    intact = torch.arange(
        batch * codebooks * steps,
        dtype=torch.long,
    ).reshape(batch, codebooks, steps) % 31
    targets = torch.tensor([1, 2])

    temporal, valid, boundaries = build_reset_future_teacher_inputs(
        intact,
        intact,
        targets,
        codec_motion_inputs=motions,
        gesture_codecs=codecs,
        horizon_tokens=2,
        past_context_tokens=3,
        mask_token_id=31,
        motion_fps=25,
        reset_prefix_drop_tokens=0,
        lower_velocity_feature_slice=(1, 4),
    )

    assert temporal.shape == intact.shape
    assert valid.shape == (batch, steps)
    assert len(boundaries) == batch
    for batch_index, boundary in enumerate(boundaries):
        assert boundary.target_frame == boundary.target_token * 2
        assert boundary.suffix_start_token == boundary.target_token + 2
        assert boundary.suffix_start_frame == (
            boundary.suffix_start_token * 2
        )
        assert boundary.offset_ms == 160.0
        start = boundary.suffix_start_token
        assert (temporal[batch_index, :, start:] != 31).any()
        # Every canonical part must differ from the intact future somewhere;
        # otherwise one body-part stream was accidentally left untouched.
        offsets = (0, 2, 4, 5)
        for part_start, part_end in zip(offsets[:-1], offsets[1:]):
            assert not torch.equal(
                temporal[
                    batch_index,
                    part_start:part_end,
                    start:,
                ],
                intact[
                    batch_index,
                    part_start:part_end,
                    start:,
                ],
            )


def test_reset_prefix_drop_masks_first_future_token():
    codecs = (
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(1),
    )
    motions = (
        _motion(batch=1, features=8),
        _motion(batch=1, features=8) + 0.1,
        _motion(batch=1, features=8) + 0.2,
    )
    intact = torch.zeros(1, 5, 8, dtype=torch.long)
    target = torch.tensor([1])
    temporal, valid, boundaries = build_reset_future_teacher_inputs(
        intact,
        intact,
        target,
        codec_motion_inputs=motions,
        gesture_codecs=codecs,
        horizon_tokens=2,
        past_context_tokens=3,
        mask_token_id=31,
        motion_fps=25,
        reset_prefix_drop_tokens=1,
        lower_velocity_feature_slice=(1, 4),
    )
    start = boundaries[0].suffix_start_token
    assert (temporal[:, :, start] == 31).all()
    assert not valid[:, start].any()
    assert valid[:, start + 1:].any()


def test_cached_variable_suffixes_match_online_construction():
    codecs = (
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(1),
    )
    motions = (
        _motion(batch=2, features=8),
        _motion(batch=2, features=8) + 0.1,
        _motion(batch=2, features=8) + 0.2,
    )
    intact = torch.zeros(2, 5, 8, dtype=torch.long)
    targets = torch.tensor([1, 2])
    online, online_valid, _ = build_reset_future_teacher_inputs(
        intact,
        intact,
        targets,
        codec_motion_inputs=motions,
        gesture_codecs=codecs,
        horizon_tokens=2,
        past_context_tokens=3,
        mask_token_id=31,
        motion_fps=25,
        lower_velocity_feature_slice=(1, 4),
    )

    reset_windows = []
    for sample_index, target in enumerate(targets.tolist()):
        start_frame = (target + 2) * 2
        parts = []
        for part_index, (codec, motion) in enumerate(
            zip(codecs, motions)
        ):
            parts.append(
                encode_reset_suffix(
                    codec,
                    motion[sample_index:sample_index + 1],
                    suffix_start_frame=start_frame,
                    zero_first_frame_feature_slice=(
                        (1, 4) if part_index == 1 else None
                    ),
                ).codes[0]
            )
        reset_windows.append(torch.cat(parts, dim=0))

    cached, cached_valid, _ = (
        build_reset_future_teacher_inputs_from_codes(
            intact,
            intact,
            targets,
            reset_code_windows=reset_windows,
            horizon_tokens=2,
            past_context_tokens=3,
            mask_token_id=31,
            motion_fps=25,
            frame_size=2,
        )
    )
    torch.testing.assert_close(cached, online)
    torch.testing.assert_close(cached_valid, online_valid)


def test_cache_signature_ignores_only_filesystem_locations():
    stored = {
        "schema": "miburi_reset_future_full_suffix",
        "schema_version": 3,
        "source_hdf5_path": "/build/repo/database.hdf5",
        "source_hdf5_size_bytes": 123456,
        "source_hdf5_mtime_ns": 987654321,
        "future_offset_tokens": 5,
        "future_window_mode": "full_remaining_suffix",
        "maximum_future_tokens": 120,
        "targets_per_clip": 120,
        "upper_codec": {
            "path": "/build/repo/upper.safetensors",
            "size_bytes": 100,
            "mtime_ns": 11,
            "sha256": "upper-hash",
        },
        "lower_codec": {
            "path": "/build/repo/lower.safetensors",
            "size_bytes": 200,
            "mtime_ns": 22,
            "sha256": "lower-hash",
        },
        "face_codec": {
            "path": "/build/repo/face.safetensors",
            "size_bytes": 300,
            "mtime_ns": 33,
            "sha256": "face-hash",
        },
    }
    relocated = copy.deepcopy(stored)
    relocated["source_hdf5_path"] = "/runtime/repo/database.hdf5"
    for key in ("upper_codec", "lower_codec", "face_codec"):
        relocated[key]["path"] = f"/runtime/repo/{key}.safetensors"

    assert reset_future_cache_signatures_match(
        json.dumps(stored, sort_keys=True),
        json.dumps(relocated, sort_keys=True),
    )

    changed_hash = copy.deepcopy(relocated)
    changed_hash["upper_codec"]["sha256"] = "different-hash"
    assert not reset_future_cache_signatures_match(
        stored,
        changed_hash,
    )

    changed_window = copy.deepcopy(relocated)
    changed_window["maximum_future_tokens"] = 60
    assert not reset_future_cache_signatures_match(
        stored,
        changed_window,
    )

    changed_dataset = copy.deepcopy(relocated)
    changed_dataset["source_hdf5_mtime_ns"] += 1
    assert not reset_future_cache_signatures_match(
        stored,
        changed_dataset,
    )

    assert not reset_future_cache_signatures_match(
        "not-json",
        relocated,
    )


def test_packed_variable_suffix_cache_round_trip():
    with tempfile.TemporaryDirectory() as temporary_dir:
        source_path = os.path.join(temporary_dir, "database.hdf5")
        codec_paths = []
        with open(source_path, "wb") as handle:
            handle.write(b"source")
        for part in ("upper", "lower", "face"):
            path = os.path.join(temporary_dir, f"{part}.safetensors")
            with open(path, "wb") as handle:
                handle.write(part.encode("utf-8"))
            codec_paths.append(path)
        args = SimpleNamespace(
            frame_chunk_size=2,
            pose_length=8,
            future_gesture_horizon_frames=2,
            future_window_ms=0.0,
            reset_future_targets_per_clip=2,
            dataset_ratio="goodspk_beatx_lowervalid",
            reset_future_cache_splits=["train"],
            beatx_cache_path=source_path,
            motion_fps=25,
            reset_future_manifest_seed=2342,
            upperlower_nfeats=258,
            lowertrans_nfeats=61,
            face_nfeats=106,
            transformer_heads=4,
            transformer_layers=8,
            convblock_layers=2,
            upperbodycodec_ckpt=codec_paths[0],
            lowerbodycodec_ckpt=codec_paths[1],
            facecodec_ckpt=codec_paths[2],
        )
        signature = build_reset_future_cache_signature(args)
        metadata = json.loads(signature)
        metadata.update(
            {
                "build_signature": signature,
                "num_codebooks": 2,
                "cardinality": 31,
                "maximum_future_tokens": 3,
                "future_offset_tokens": 1,
                "manifest_seed": 2342,
                "target_count": 2,
                "completed": True,
            }
        )
        with open(
            os.path.join(temporary_dir, METADATA_NAME),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(
            os.path.join(temporary_dir, MANIFEST_NAME), "w"
        ) as manifest:
            manifest.attrs["schema"] = RESET_FUTURE_CACHE_SCHEMA
            manifest.attrs["schema_version"] = RESET_FUTURE_CACHE_VERSION
            manifest.attrs["build_signature"] = signature
            manifest.create_dataset(
                "sequence_ids",
                data=np.asarray(["sequence"], dtype=object),
                dtype=string_dtype,
            )
            manifest.create_dataset(
                "sequence_splits",
                data=np.asarray(["train"], dtype=object),
                dtype=string_dtype,
            )
            manifest.create_dataset(
                "shard_names",
                data=np.asarray(["train_00000.h5"], dtype=object),
                dtype=string_dtype,
            )
            arrays = {
                "sequence_target_offsets": np.asarray([0, 2], np.int64),
                "target_token_index": np.asarray([0, 1], np.uint16),
                "future_start_token": np.asarray([1, 2], np.uint16),
                "valid_future_length": np.asarray([3, 2], np.uint16),
                "sequence_index": np.asarray([0, 0], np.int32),
                "shard_index": np.asarray([0, 0], np.int32),
                "shard_row": np.asarray([0, 1], np.int32),
                "shard_token_offset": np.asarray([0, 3], np.int64),
            }
            for name, values in arrays.items():
                manifest.create_dataset(name, data=values)

        packed = np.asarray(
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            dtype=np.uint16,
        )
        with h5py.File(
            os.path.join(temporary_dir, "train_00000.h5"), "w"
        ) as shard:
            shard.attrs["completed"] = True
            shard.attrs["build_signature"] = signature
            shard.create_dataset("reset_future_codes", data=packed)
            shard.create_dataset(
                "written", data=np.asarray([True, True], dtype=np.bool_)
            )
            shard.create_dataset(
                "manifest_row", data=np.asarray([0, 1], dtype=np.int64)
            )
            shard.create_dataset(
                "token_offset", data=np.asarray([0, 3, 5], dtype=np.int64)
            )

        cache = ResetFutureManifestCache(
            temporary_dir,
            args=args,
            expected_codebooks=2,
            expected_cardinality=31,
        )
        seen = {}
        for epoch in (0, 1):
            targets, windows = cache.load_batch(
                ["sequence"], split="train", epoch=epoch
            )
            target = int(targets.item())
            seen[target] = windows[0].codes.numpy()
        assert set(seen) == {0, 1}
        np.testing.assert_array_equal(seen[0], packed[:3].T)
        np.testing.assert_array_equal(seen[1], packed[3:].T)

        targets, windows = cache.load_batch(
            ["sequence"],
            split="train",
            epoch=99,
            minimum_valid_future_tokens=3,
        )
        assert int(targets.item()) == 0
        assert windows[0].valid_future_tokens == 3
        cache.close()


if __name__ == "__main__":
    test_hidden_interval_and_past_invariance()
    test_future_sensitivity_and_cache_contamination()
    test_batched_reset_matches_individual_resets()
    test_right_padded_reset_prefix_matches_individual_suffix()
    test_active_streaming_state_is_rejected()
    test_lower_boundary_velocity_is_internal_to_suffix()
    test_all_body_parts_are_reset_and_boundaries_align()
    test_reset_prefix_drop_masks_first_future_token()
    test_cached_variable_suffixes_match_online_construction()
    test_cache_signature_ignores_only_filesystem_locations()
    test_packed_variable_suffix_cache_round_trip()
    print(
        "Reset-future smoke tests passed "
        "(invariance/sensitivity/batched reset/cache/signature/boundary/"
        "all parts)."
    )
