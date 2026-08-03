"""Fixed-manifest, packed HDF5 cache for full reset-future suffixes."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import h5py
import numpy as np
import torch


RESET_FUTURE_CACHE_SCHEMA = "miburi_reset_future_full_suffix"
RESET_FUTURE_CACHE_VERSION = 3
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
    if int(args.pose_length) % frame_size:
        raise ValueError(
            "pose_length must align to a whole number of gesture tokens."
        )
    if int(args.future_gesture_horizon_frames) % frame_size:
        raise ValueError(
            "future_gesture_horizon_frames must align to a whole number of "
            "gesture tokens."
        )
    if float(args.future_window_ms) != 0.0:
        raise ValueError(
            "Full-suffix reset cache construction requires "
            "future_window_ms=0. A fixed reset window changes the teacher's "
            "information boundary."
        )
    gesture_tokens = int(args.pose_length) // frame_size
    offset_tokens = (
        int(args.future_gesture_horizon_frames) // frame_size
    )
    eligible_targets = gesture_tokens - offset_tokens
    if eligible_targets <= 0:
        raise ValueError(
            "The configured future offset leaves no reset suffix token."
        )
    requested_targets = int(args.reset_future_targets_per_clip)
    if requested_targets <= 0 or requested_targets > eligible_targets:
        raise ValueError(
            f"reset_future_targets_per_clip={requested_targets} is outside "
            f"the valid range [1, {eligible_targets}]."
        )
    target_sampling = (
        "exhaustive_all_eligible"
        if requested_targets == eligible_targets
        else "fixed_uniform_random_without_replacement"
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
        "future_window_mode": "full_remaining_suffix",
        "maximum_future_tokens": int(eligible_targets),
        "targets_per_clip": requested_targets,
        "eligible_targets_per_clip": int(eligible_targets),
        "target_sampling": target_sampling,
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


def _semantic_signature_payload(
    signature: str | dict[str, Any],
) -> dict[str, Any]:
    """Return cache identity fields without filesystem-location metadata.

    Absolute paths are retained in ``metadata.json`` for auditability, but
    moving an otherwise identical dataset/checkpoint/cache checkout must not
    invalidate its discrete RVQ codes. Checkpoint hashes, sizes, mtimes, and
    every dataset/manifest/preprocessing setting remain part of the identity.
    """

    if isinstance(signature, str):
        payload = json.loads(signature)
    elif isinstance(signature, dict):
        payload = copy.deepcopy(signature)
    else:
        raise TypeError(
            "Reset-future cache signature must be JSON text or a mapping."
        )
    if not isinstance(payload, dict):
        raise TypeError(
            "Reset-future cache signature payload must be a mapping."
        )

    payload.pop("source_hdf5_path", None)
    for key in ("upper_codec", "lower_codec", "face_codec"):
        checkpoint = payload.get(key)
        if isinstance(checkpoint, dict):
            checkpoint.pop("path", None)
    return payload


def reset_future_cache_signatures_match(
    stored_signature: str | dict[str, Any] | None,
    expected_signature: str | dict[str, Any] | None,
) -> bool:
    """Compare semantic cache identity while ignoring absolute locations."""

    if stored_signature is None or expected_signature is None:
        return False
    try:
        return _semantic_signature_payload(
            stored_signature
        ) == _semantic_signature_payload(expected_signature)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


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
    """Read selected manifest targets and their packed variable suffixes."""

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
        if not reset_future_cache_signatures_match(
            self.metadata.get("build_signature"),
            expected_signature,
        ):
            raise RuntimeError(
                "Reset-future cache provenance does not match the current "
                "dataset, codec checkpoints, manifest, or window settings."
            )
        self.num_codebooks = int(self.metadata["num_codebooks"])
        self.cardinality = int(self.metadata["cardinality"])
        self.maximum_future_tokens = int(
            self.metadata["maximum_future_tokens"]
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
        self.shard_token_offsets = np.asarray(
            self._manifest["shard_token_offset"][:],
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
        if target_rows != int(self.metadata["target_count"]):
            raise RuntimeError(
                "Reset target manifest length differs from metadata."
            )
        for values in (
            self.future_start_tokens,
            self.valid_future_lengths,
            self.target_sequence_indices,
            self.shard_indices,
            self.shard_rows,
            self.shard_token_offsets,
        ):
            if values.shape != (target_rows,):
                raise RuntimeError(
                    "Reset target manifest arrays have inconsistent lengths."
                )
        if (
            self.sequence_target_offsets[0] != 0
            or self.sequence_target_offsets[-1] != target_rows
            or (np.diff(self.sequence_target_offsets) <= 0).any()
        ):
            raise RuntimeError(
                "Manifest sequence target offsets are invalid."
            )
        if (
            (self.target_sequence_indices < 0).any()
            or (
                self.target_sequence_indices
                >= len(self.sequence_ids)
            ).any()
        ):
            raise RuntimeError(
                "Manifest target rows reference invalid sequences."
            )
        for sequence_index in range(len(self.sequence_ids)):
            start = int(
                self.sequence_target_offsets[sequence_index]
            )
            end = int(
                self.sequence_target_offsets[sequence_index + 1]
            )
            if not np.all(
                self.target_sequence_indices[start:end]
                == sequence_index
            ):
                raise RuntimeError(
                    "Manifest target rows are not grouped by sequence."
                )
        if (
            (self.valid_future_lengths <= 0).any()
            or (
                self.valid_future_lengths
                > self.maximum_future_tokens
            ).any()
        ):
            raise RuntimeError(
                "Manifest contains invalid reset-window lengths."
            )
        if not np.array_equal(
            self.future_start_tokens,
            self.target_token_indices + self.future_offset_tokens,
        ):
            raise RuntimeError(
                "Manifest target/future offsets are inconsistent."
            )
        gesture_steps = (
            int(self.metadata["pose_length"])
            // int(self.metadata["frame_size"])
        )
        if (
            self.future_start_tokens + self.valid_future_lengths
            > gesture_steps
        ).any():
            raise RuntimeError(
                "Manifest reset windows extend past their source clips."
            )
        if not np.array_equal(
            self.future_start_tokens + self.valid_future_lengths,
            np.full(target_rows, gesture_steps, dtype=np.int64),
        ):
            raise RuntimeError(
                "Manifest rows are not full remaining reset suffixes."
            )
        if (
            not self.shard_names
            or (self.shard_indices < 0).any()
            or (self.shard_indices >= len(self.shard_names)).any()
            or (self.shard_rows < 0).any()
        ):
            raise RuntimeError(
                "Manifest contains invalid reset-cache shard references."
            )
        self._shards: dict[int, h5py.File] = {}
        # Validate every shard at startup. A missing or damaged late shard
        # should stop the run now, not after several training epochs finally
        # select one of its target rows.
        for shard_index in range(len(self.shard_names)):
            self._open_shard(shard_index)

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
        codes = shard.get("reset_future_codes")
        written = shard.get("written")
        manifest_rows = shard.get("manifest_row")
        if codes is None or written is None or manifest_rows is None:
            shard.close()
            raise RuntimeError(
                f"Reset cache shard is missing required datasets: {path}"
            )
        token_offsets = shard.get("token_offset")
        if (
            codes.ndim != 2
            or codes.shape[1] != self.num_codebooks
            or codes.dtype != np.dtype(np.uint16)
            or manifest_rows.ndim != 1
            or token_offsets is None
            or token_offsets.shape != (manifest_rows.shape[0] + 1,)
        ):
            shard.close()
            raise RuntimeError(
                f"Reset cache shard has an invalid layout: {path}"
            )
        if written.shape != (manifest_rows.shape[0],):
            shard.close()
            raise RuntimeError(
                f"Reset cache shard written flags have an invalid layout: "
                f"{path}"
            )
        if not np.asarray(written[:], dtype=np.bool_).all():
            shard.close()
            raise RuntimeError(
                f"Completed reset cache shard still has gaps: {path}"
            )
        stored_manifest_rows = np.asarray(
            manifest_rows[:],
            dtype=np.int64,
        )
        expected_manifest_rows = np.flatnonzero(
            self.shard_indices == shard_index
        ).astype(np.int64)
        expected_manifest_rows = expected_manifest_rows[
            np.argsort(self.shard_rows[expected_manifest_rows])
        ]
        expected_shard_rows = self.shard_rows[
            expected_manifest_rows
        ]
        stored_token_offsets = np.asarray(
            token_offsets[:],
            dtype=np.int64,
        )
        expected_lengths = self.valid_future_lengths[
            expected_manifest_rows
        ]
        if (
            not np.array_equal(
                stored_manifest_rows,
                expected_manifest_rows,
            )
            or not np.array_equal(
                expected_shard_rows,
                np.arange(manifest_rows.shape[0], dtype=np.int64),
            )
            or stored_token_offsets[0] != 0
            or stored_token_offsets[-1] != codes.shape[0]
            or not np.array_equal(
                np.diff(stored_token_offsets),
                expected_lengths,
            )
            or not np.array_equal(
                self.shard_token_offsets[expected_manifest_rows],
                stored_token_offsets[:-1],
            )
        ):
            shard.close()
            raise RuntimeError(
                f"Reset cache shard rows disagree with the manifest: {path}"
            )
        self._shards[shard_index] = shard
        return shard

    def _manifest_row_for_sequence(
        self,
        sequence_id: str,
        *,
        split: str,
        epoch: int,
        minimum_valid_future_tokens: int,
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
        eligible_rows = np.flatnonzero(
            self.valid_future_lengths[start:end]
            >= int(minimum_valid_future_tokens)
        )
        count = int(eligible_rows.shape[0])
        if count <= 0:
            raise RuntimeError(
                f"Sequence {sequence_id!r} has no cached target with at "
                f"least {minimum_valid_future_tokens} future tokens."
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
        manifest_row = start + int(eligible_rows[slot])
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
        minimum_valid_future_tokens: int = 1,
    ) -> tuple[torch.Tensor, list[CachedResetWindow]]:
        if minimum_valid_future_tokens <= 0:
            raise ValueError(
                "minimum_valid_future_tokens must be positive."
            )
        manifest_rows = [
            self._manifest_row_for_sequence(
                str(sequence_id),
                split=split,
                epoch=epoch,
                minimum_valid_future_tokens=(
                    minimum_valid_future_tokens
                ),
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
                [entry[0] for entry in entries], dtype=np.int64
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
            token_offsets = np.asarray(
                shard["token_offset"][:], dtype=np.int64
            )
            for shard_row, batch_index, manifest_row in entries:
                token_start = int(token_offsets[shard_row])
                token_end = int(token_offsets[shard_row + 1])
                row_codes = np.asarray(
                    shard["reset_future_codes"][token_start:token_end],
                    dtype=np.int64,
                ).T
                expected_length = int(
                    self.valid_future_lengths[manifest_row]
                )
                if row_codes.shape != (
                    self.num_codebooks,
                    expected_length,
                ):
                    raise RuntimeError(
                        "Packed reset suffix length disagrees with the "
                        "manifest."
                    )
                loaded_codes[batch_index] = row_codes

        windows = []
        targets = []
        for manifest_row, codes in zip(manifest_rows, loaded_codes):
            if codes is None:
                raise RuntimeError("A reset cache row was not loaded.")
            valid_length = int(
                self.valid_future_lengths[manifest_row]
            )
            expected_shape = (self.num_codebooks, valid_length)
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
