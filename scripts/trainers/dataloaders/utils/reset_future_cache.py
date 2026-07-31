"""Fixed-manifest, sharded HDF5 cache for reset-future gesture codes."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import h5py
import numpy as np
import torch


RESET_FUTURE_CACHE_SCHEMA = "miburi_reset_future_fixed_window"
RESET_FUTURE_CACHE_VERSION = 2
METADATA_NAME = "metadata.json"
MANIFEST_NAME = "target_manifest.h5"


def _checkpoint_signature(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "size_bytes": None, "mtime_ns": None}
    absolute_path = os.path.abspath(path)
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(
            f"Gesture codec checkpoint does not exist: {absolute_path}"
        )
    stat = os.stat(absolute_path)
    digest = hashlib.sha256()
    with open(absolute_path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return {
        "path": absolute_path,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def build_reset_future_cache_signature_payload(args) -> dict[str, Any]:
    """Metadata that makes stale or incompatible caches fail loudly."""

    frame_size = int(args.frame_chunk_size)
    window_frames = round(
        float(args.future_window_ms)
        * float(args.motion_fps)
        / 1000.0
    )
    if window_frames <= 0 or window_frames % frame_size:
        raise ValueError(
            "future_window_ms must map to a positive whole number of gesture "
            "tokens for fixed-window cache construction."
        )
    return {
        "schema": RESET_FUTURE_CACHE_SCHEMA,
        "schema_version": RESET_FUTURE_CACHE_VERSION,
        "dataset_ratio": str(args.dataset_ratio),
        "dataset_splits": [
            str(split) for split in args.reset_future_cache_splits
        ],
        "source_hdf5_path": os.path.abspath(args.beatx_cache_path),
        "source_hdf5_size_bytes": int(
            os.path.getsize(args.beatx_cache_path)
        ),
        "source_hdf5_mtime_ns": int(
            os.stat(args.beatx_cache_path).st_mtime_ns
        ),
        "preprocessing_version": (
            "global_hdf5_axisangle_to_rot6d_transvel_boundaryzero_v1"
        ),
        "motion_fps": int(args.motion_fps),
        "pose_length": int(args.pose_length),
        "frame_size": frame_size,
        "future_offset_frames": int(
            args.future_gesture_horizon_frames
        ),
        "future_offset_tokens": (
            int(args.future_gesture_horizon_frames) // frame_size
        ),
        "future_window_frames": int(window_frames),
        "future_window_tokens": int(window_frames // frame_size),
        "targets_per_clip": int(args.reset_future_targets_per_clip),
        "target_sampling": "fixed_uniform_random_without_replacement",
        "manifest_seed": int(args.reset_future_manifest_seed),
        "upperlower_nfeats": int(args.upperlower_nfeats),
        "lowertrans_nfeats": int(args.lowertrans_nfeats),
        "face_nfeats": int(args.face_nfeats),
        "transformer_heads": int(args.transformer_heads),
        "transformer_layers": int(args.transformer_layers),
        "convblock_layers": int(args.convblock_layers),
        "upper_codec": _checkpoint_signature(args.upperbodycodec_ckpt),
        "lower_codec": _checkpoint_signature(args.lowerbodycodec_ckpt),
        "face_codec": _checkpoint_signature(args.facecodec_ckpt),
    }


def build_reset_future_cache_signature(args) -> str:
    payload = build_reset_future_cache_signature_payload(args)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def stable_sequence_seed(sequence_id: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"{int(seed)}:{sequence_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


@dataclass(frozen=True)
class CachedResetWindow:
    codes: torch.Tensor
    target_token: int
    future_start_token: int
    valid_future_tokens: int
    manifest_row: int


class ResetFutureManifestCache:
    """Read selected fixed-manifest targets and only their cached code rows."""

    def __init__(
        self,
        cache_dir: str,
        *,
        args,
        expected_codebooks: int,
        expected_cardinality: int,
        require_complete: bool = True,
    ):
        self.cache_dir = os.path.abspath(cache_dir)
        metadata_path = os.path.join(self.cache_dir, METADATA_NAME)
        manifest_path = os.path.join(self.cache_dir, MANIFEST_NAME)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Reset cache metadata not found: {metadata_path}"
            )
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Reset cache manifest not found: {manifest_path}"
            )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        if require_complete and not bool(
            self.metadata.get("completed", False)
        ):
            raise RuntimeError(
                f"Reset-future cache is incomplete: {self.cache_dir}"
            )
        expected_signature = build_reset_future_cache_signature(args)
        if self.metadata.get("build_signature") != expected_signature:
            raise RuntimeError(
                "Reset-future cache provenance does not match the current "
                "dataset, codec checkpoints, manifest, or window settings."
            )
        self.num_codebooks = int(self.metadata["num_codebooks"])
        self.cardinality = int(self.metadata["cardinality"])
        self.future_window_tokens = int(
            self.metadata["future_window_tokens"]
        )
        self.future_offset_tokens = int(
            self.metadata["future_offset_tokens"]
        )
        self.manifest_seed = int(self.metadata["manifest_seed"])
        if self.num_codebooks != int(expected_codebooks):
            raise RuntimeError(
                f"Reset cache has {self.num_codebooks} codebooks; expected "
                f"{expected_codebooks}."
            )
        if self.cardinality != int(expected_cardinality):
            raise RuntimeError(
                f"Reset cache cardinality is {self.cardinality}; expected "
                f"{expected_cardinality}."
            )

        try:
            self._manifest = h5py.File(manifest_path, "r", swmr=True)
        except Exception:
            self._manifest = h5py.File(manifest_path, "r")
        if str(self._manifest.attrs.get("build_signature", "")) != str(
            self.metadata["build_signature"]
        ):
            raise RuntimeError(
                "Reset target manifest provenance differs from metadata."
            )
        self.sequence_ids = tuple(
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
            for value in self._manifest["sequence_ids"][:]
        )
        self.sequence_splits = tuple(
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
            for value in self._manifest["sequence_splits"][:]
        )
        self.sequence_target_offsets = np.asarray(
            self._manifest["sequence_target_offsets"][:],
            dtype=np.int64,
        )
        self.target_token_indices = np.asarray(
            self._manifest["target_token_index"][:],
            dtype=np.int64,
        )
        self.future_start_tokens = np.asarray(
            self._manifest["future_start_token"][:],
            dtype=np.int64,
        )
        self.valid_future_lengths = np.asarray(
            self._manifest["valid_future_length"][:],
            dtype=np.int64,
        )
        self.target_sequence_indices = np.asarray(
            self._manifest["sequence_index"][:],
            dtype=np.int64,
        )
        self.shard_indices = np.asarray(
            self._manifest["shard_index"][:],
            dtype=np.int64,
        )
        self.shard_rows = np.asarray(
            self._manifest["shard_row"][:],
            dtype=np.int64,
        )
        self.shard_names = tuple(
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
            for value in self._manifest["shard_names"][:]
        )
        self._sequence_lookup = {
            sequence_id: index
            for index, sequence_id in enumerate(self.sequence_ids)
        }
        if len(self._sequence_lookup) != len(self.sequence_ids):
            raise RuntimeError(
                "Reset target manifest contains duplicate sequence IDs."
            )
        if (
            self.sequence_target_offsets.shape[0]
            != len(self.sequence_ids) + 1
        ):
            raise RuntimeError(
                "Manifest sequence_target_offsets has the wrong length."
            )
        target_rows = int(self.target_token_indices.shape[0])
        for values in (
            self.future_start_tokens,
            self.valid_future_lengths,
            self.target_sequence_indices,
            self.shard_indices,
            self.shard_rows,
        ):
            if values.shape != (target_rows,):
                raise RuntimeError(
                    "Reset target manifest arrays have inconsistent lengths."
                )
        self._shards: dict[int, h5py.File] = {}

    def close(self) -> None:
        manifest = getattr(self, "_manifest", None)
        if manifest is not None:
            manifest.close()
            self._manifest = None
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _open_shard(self, shard_index: int) -> h5py.File:
        shard = self._shards.get(shard_index)
        if shard is not None:
            return shard
        try:
            shard_name = self.shard_names[shard_index]
        except IndexError as exc:
            raise RuntimeError(
                f"Manifest references unknown shard {shard_index}."
            ) from exc
        path = os.path.join(self.cache_dir, shard_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Reset cache shard is missing: {path}"
            )
        try:
            shard = h5py.File(path, "r", swmr=True)
        except Exception:
            shard = h5py.File(path, "r")
        if not bool(shard.attrs.get("completed", False)):
            shard.close()
            raise RuntimeError(f"Reset cache shard is incomplete: {path}")
        if str(shard.attrs.get("build_signature", "")) != str(
            self.metadata["build_signature"]
        ):
            shard.close()
            raise RuntimeError(
                f"Reset cache shard provenance mismatch: {path}"
            )
        self._shards[shard_index] = shard
        return shard

    def _manifest_row_for_sequence(
        self,
        sequence_id: str,
        *,
        split: str,
        epoch: int,
    ) -> int:
        if sequence_id not in self._sequence_lookup:
            raise KeyError(
                f"Sequence {sequence_id!r} is absent from reset manifest."
            )
        sequence_index = self._sequence_lookup[sequence_id]
        manifest_split = self.sequence_splits[sequence_index]
        if manifest_split != split:
            raise RuntimeError(
                f"Sequence {sequence_id!r} is split={manifest_split!r} in "
                f"the manifest, not {split!r}."
            )
        start = int(self.sequence_target_offsets[sequence_index])
        end = int(self.sequence_target_offsets[sequence_index + 1])
        count = end - start
        if count <= 0:
            raise RuntimeError(
                f"Sequence {sequence_id!r} has no cached targets."
            )
        base_slot = stable_sequence_seed(
            sequence_id,
            self.manifest_seed,
        ) % count
        slot = (
            (base_slot + int(epoch)) % count
            if split == "train"
            else base_slot
        )
        manifest_row = start + slot
        if int(self.target_sequence_indices[manifest_row]) != sequence_index:
            raise RuntimeError(
                "Manifest target row points to the wrong sequence."
            )
        return manifest_row

    def load_batch(
        self,
        sequence_ids: Sequence[str],
        *,
        split: str,
        epoch: int,
    ) -> tuple[torch.Tensor, list[CachedResetWindow]]:
        manifest_rows = [
            self._manifest_row_for_sequence(
                str(sequence_id),
                split=split,
                epoch=epoch,
            )
            for sequence_id in sequence_ids
        ]
        grouped_rows = defaultdict(list)
        for batch_index, manifest_row in enumerate(manifest_rows):
            grouped_rows[int(self.shard_indices[manifest_row])].append(
                (
                    int(self.shard_rows[manifest_row]),
                    batch_index,
                    manifest_row,
                )
            )
        loaded_codes = [None] * len(manifest_rows)
        for shard_index, entries in grouped_rows.items():
            entries.sort()
            shard = self._open_shard(shard_index)
            row_indices = np.asarray(
                [entry[0] for entry in entries],
                dtype=np.int64,
            )
            stored_manifest_rows = np.asarray(
                shard["manifest_row"][row_indices],
                dtype=np.int64,
            )
            expected_manifest_rows = np.asarray(
                [entry[2] for entry in entries],
                dtype=np.int64,
            )
            if not np.array_equal(
                stored_manifest_rows,
                expected_manifest_rows,
            ):
                raise RuntimeError(
                    "Reset shard rows do not match the target manifest."
                )
            shard_codes = np.asarray(
                shard["reset_future_codes"][row_indices],
                dtype=np.int64,
            )
            for row_codes, (_, batch_index, _) in zip(
                shard_codes,
                entries,
            ):
                loaded_codes[batch_index] = row_codes

        windows = []
        targets = []
        for manifest_row, codes in zip(manifest_rows, loaded_codes):
            if codes is None:
                raise RuntimeError("A reset cache row was not loaded.")
            expected_shape = (
                self.num_codebooks,
                self.future_window_tokens,
            )
            if codes.shape != expected_shape:
                raise RuntimeError(
                    f"Cached reset window has shape {codes.shape}, expected "
                    f"{expected_shape}."
                )
            if codes.size and (
                codes.min() < 0 or codes.max() >= self.cardinality
            ):
                raise RuntimeError(
                    "Cached reset window contains an invalid code index."
                )
            target_token = int(
                self.target_token_indices[manifest_row]
            )
            future_start = int(
                self.future_start_tokens[manifest_row]
            )
            if future_start != target_token + self.future_offset_tokens:
                raise RuntimeError(
                    "Cached target/future offset is inconsistent."
                )
            valid_length = int(
                self.valid_future_lengths[manifest_row]
            )
            windows.append(
                CachedResetWindow(
                    codes=torch.from_numpy(codes),
                    target_token=target_token,
                    future_start_token=future_start,
                    valid_future_tokens=valid_length,
                    manifest_row=manifest_row,
                )
            )
            targets.append(target_token)
        return torch.tensor(targets, dtype=torch.long), windows
