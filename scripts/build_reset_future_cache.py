"""Build fixed-target, fixed-window reset codes in sharded HDF5 files."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict
from typing import Any

import h5py
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from miburi.models import encode_reset_suffix, loaders
from trainers.dataloaders import UNIFIEDDataset
from trainers.dataloaders.utils.reset_future_cache import (
    MANIFEST_NAME,
    METADATA_NAME,
    RESET_FUTURE_CACHE_SCHEMA,
    RESET_FUTURE_CACHE_VERSION,
    build_reset_future_cache_signature,
    stable_sequence_seed,
)
from trainers.utils import config
from trainers.utils import rotation_conversions as rc
from trainers.utils import tools as other_tools


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _write_json(path: str, payload: dict[str, Any]) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _codec_inputs(batch: dict[str, Any], device, args):
    upper = batch["motion_upper"].to(device, non_blocking=True)
    hands = batch["motion_hands"].to(device, non_blocking=True)
    lower = batch["motion_lower"].to(device, non_blocking=True)
    face = batch["motion_face"].to(device, non_blocking=True)
    translations = batch["transl"].to(device, non_blocking=True).clone()
    contacts = batch["contact"]
    expressions = batch["expressions"]
    if contacts is None or expressions is None:
        raise RuntimeError(
            "Reset cache construction requires contacts and expressions."
        )
    contacts = contacts.to(device, non_blocking=True)
    expressions = expressions.to(device, non_blocking=True)
    translations[:, :, 0] -= translations[:, 0:1, 0]
    translations[:, :, 2] -= translations[:, 0:1, 2]
    translation_velocity = other_tools.estimate_linear_velocity(
        translations,
        dt=1 / float(args.motion_fps),
    )

    batch_size, frames, upper_dim = upper.shape

    def to_6d(values):
        joints = values.shape[-1] // 3
        return rc.matrix_to_rotation_6d(
            rc.axis_angle_to_matrix(
                values.reshape(batch_size, frames, joints, 3)
            )
        ).reshape(batch_size, frames, joints * 6)

    codec_inputs = (
        torch.cat([to_6d(upper), to_6d(hands)], dim=-1),
        torch.cat(
            [to_6d(lower), translation_velocity, contacts],
            dim=-1,
        ),
        torch.cat([to_6d(face), expressions], dim=-1),
    )
    actual = tuple(values.shape[-1] for values in codec_inputs)
    expected = (
        int(args.upperlower_nfeats),
        int(args.lowertrans_nfeats),
        int(args.face_nfeats),
    )
    if actual != expected:
        raise RuntimeError(
            f"Codec input widths are {actual}, expected {expected}."
        )
    return codec_inputs


def _manifest_arrays(args):
    frame_size = int(args.frame_chunk_size)
    gesture_tokens = int(args.pose_length) // frame_size
    offset_tokens = (
        int(args.future_gesture_horizon_frames) // frame_size
    )
    window_frames = round(
        float(args.future_window_ms)
        * float(args.motion_fps)
        / 1000.0
    )
    window_tokens = window_frames // frame_size
    max_target = gesture_tokens - offset_tokens - window_tokens
    candidate_targets = np.arange(max_target + 1, dtype=np.int64)
    targets_per_clip = int(args.reset_future_targets_per_clip)
    if targets_per_clip <= 0:
        raise ValueError(
            "reset_future_targets_per_clip must be positive."
        )
    if targets_per_clip > candidate_targets.shape[0]:
        raise ValueError(
            f"Requested {targets_per_clip} targets but only "
            f"{candidate_targets.shape[0]} fit in each clip."
        )

    sequence_ids = []
    sequence_splits = []
    targets = []
    future_starts = []
    raw_starts = []
    raw_ends = []
    valid_lengths = []
    target_sequence_indices = []
    target_slots = []
    sequence_offsets = [0]
    datasets = {}
    for split in args.reset_future_cache_splits:
        dataset = UNIFIEDDataset(
            args,
            split,
            only_motion=True,
            dataset_ratio=args.dataset_ratio,
            debug=False,
            varying_frame_length=False,
            ret_rawaudio=False,
            ret_vad=False,
        )
        datasets[split] = dataset
        for ref in dataset._chunk_refs:
            sequence_id = str(ref.chunk_id)
            rng = np.random.default_rng(
                stable_sequence_seed(
                    sequence_id,
                    int(args.reset_future_manifest_seed),
                )
            )
            sampled = np.sort(
                rng.choice(
                    candidate_targets,
                    size=targets_per_clip,
                    replace=False,
                )
            )
            sequence_index = len(sequence_ids)
            sequence_ids.append(sequence_id)
            sequence_splits.append(split)
            for target_slot, target in enumerate(sampled):
                future_start = int(target) + offset_tokens
                targets.append(int(target))
                future_starts.append(future_start)
                raw_start = future_start * frame_size
                raw_starts.append(raw_start)
                raw_ends.append(raw_start + window_frames)
                valid_lengths.append(window_tokens)
                target_sequence_indices.append(sequence_index)
                target_slots.append(target_slot)
            sequence_offsets.append(len(targets))
    if len(sequence_ids) != len(set(sequence_ids)):
        raise RuntimeError(
            "Selected source caches contain duplicate filechunk_id values."
        )

    target_count = len(targets)
    shard_capacity = int(args.reset_future_cache_shard_targets)
    if shard_capacity <= 0:
        raise ValueError(
            "reset_future_cache_shard_targets must be positive."
        )
    shard_indices = np.empty(target_count, dtype=np.int32)
    shard_rows = np.empty(target_count, dtype=np.int32)
    shard_names = []
    split_shard_counts = defaultdict(int)
    split_shard_fill = defaultdict(int)
    for sequence_index, split in enumerate(sequence_splits):
        row_start = sequence_offsets[sequence_index]
        row_end = sequence_offsets[sequence_index + 1]
        for row in range(row_start, row_end):
            shard_number = split_shard_counts[split]
            fill = split_shard_fill[split]
            if fill >= shard_capacity:
                shard_number += 1
                split_shard_counts[split] = shard_number
                fill = 0
            shard_name = f"{split}_{shard_number:05d}.h5"
            if shard_name not in shard_names:
                shard_names.append(shard_name)
            shard_indices[row] = shard_names.index(shard_name)
            shard_rows[row] = fill
            split_shard_fill[split] = fill + 1

    return {
        "sequence_ids": sequence_ids,
        "sequence_splits": sequence_splits,
        "sequence_target_offsets": np.asarray(
            sequence_offsets,
            dtype=np.int64,
        ),
        "target_token_index": np.asarray(targets, dtype=np.uint16),
        "future_start_token": np.asarray(
            future_starts,
            dtype=np.uint16,
        ),
        "raw_future_start_frame": np.asarray(
            raw_starts,
            dtype=np.uint16,
        ),
        "raw_future_end_frame": np.asarray(
            raw_ends,
            dtype=np.uint16,
        ),
        "valid_future_length": np.asarray(
            valid_lengths,
            dtype=np.uint16,
        ),
        "sequence_index": np.asarray(
            target_sequence_indices,
            dtype=np.int32,
        ),
        "target_slot": np.asarray(target_slots, dtype=np.uint8),
        "shard_index": shard_indices,
        "shard_row": shard_rows,
        "shard_names": shard_names,
        "datasets": datasets,
        "gesture_tokens": gesture_tokens,
        "offset_tokens": offset_tokens,
        "window_tokens": window_tokens,
        "window_frames": window_frames,
    }


def _write_manifest(
    path: str,
    manifest,
    *,
    build_signature: str,
) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = RESET_FUTURE_CACHE_SCHEMA
        h5.attrs["schema_version"] = RESET_FUTURE_CACHE_VERSION
        h5.attrs["build_signature"] = build_signature
        h5.create_dataset(
            "sequence_ids",
            data=np.asarray(manifest["sequence_ids"], dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "sequence_splits",
            data=np.asarray(manifest["sequence_splits"], dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "shard_names",
            data=np.asarray(manifest["shard_names"], dtype=object),
            dtype=string_dtype,
        )
        for name in (
            "sequence_target_offsets",
            "target_token_index",
            "future_start_token",
            "raw_future_start_frame",
            "raw_future_end_frame",
            "valid_future_length",
            "sequence_index",
            "target_slot",
            "shard_index",
            "shard_row",
        ):
            h5.create_dataset(name, data=manifest[name])


def _validate_existing_manifest(
    path: str,
    manifest,
    *,
    build_signature: str,
) -> None:
    """Require an existing resume manifest to match every generated field."""

    string_fields = (
        "sequence_ids",
        "sequence_splits",
        "shard_names",
    )
    array_fields = (
        "sequence_target_offsets",
        "target_token_index",
        "future_start_token",
        "raw_future_start_frame",
        "raw_future_end_frame",
        "valid_future_length",
        "sequence_index",
        "target_slot",
        "shard_index",
        "shard_row",
    )
    with h5py.File(path, "r") as existing:
        if str(existing.attrs.get("build_signature", "")) != (
            build_signature
        ):
            raise RuntimeError(
                "Existing target manifest provenance differs."
            )
        for field in string_fields:
            stored = tuple(
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
                for value in existing[field][:]
            )
            expected = tuple(str(value) for value in manifest[field])
            if stored != expected:
                raise RuntimeError(
                    f"Existing target manifest field {field!r} differs."
                )
        for field in array_fields:
            stored = np.asarray(existing[field][:])
            expected = np.asarray(manifest[field])
            if stored.dtype != expected.dtype or not np.array_equal(
                stored,
                expected,
            ):
                raise RuntimeError(
                    f"Existing target manifest field {field!r} differs."
                )


def _compression_kwargs(args) -> dict[str, Any]:
    compression = str(args.reset_future_cache_compression).lower()
    if compression == "none":
        return {}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    if compression == "gzip":
        return {
            "compression": "gzip",
            "compression_opts": 1,
            "shuffle": True,
        }
    raise ValueError(f"Unknown compression mode {compression!r}.")


def _initialize_shards(
    cache_dir: str,
    manifest,
    *,
    num_codebooks: int,
    cardinality: int,
    build_signature: str,
    args,
) -> None:
    compression = _compression_kwargs(args)
    for shard_index, shard_name in enumerate(manifest["shard_names"]):
        rows = np.where(manifest["shard_index"] == shard_index)[0]
        if rows.size == 0:
            continue
        shard_path = os.path.join(cache_dir, shard_name)
        if os.path.exists(shard_path):
            with h5py.File(shard_path, "r") as existing:
                expected = (
                    rows.size,
                    num_codebooks,
                    manifest["window_tokens"],
                )
                if (
                    existing["reset_future_codes"].shape != expected
                    or str(existing.attrs.get("build_signature", ""))
                    != build_signature
                ):
                    raise RuntimeError(
                        f"Existing shard {shard_path} is incompatible."
                    )
            continue
        with h5py.File(shard_path, "w", libver="latest") as shard:
            shard.attrs["schema"] = RESET_FUTURE_CACHE_SCHEMA
            shard.attrs["schema_version"] = RESET_FUTURE_CACHE_VERSION
            shard.attrs["build_signature"] = build_signature
            shard.attrs["completed"] = False
            shard.attrs["cardinality"] = cardinality
            shard.attrs["num_codebooks"] = num_codebooks
            shard.attrs["reset_prefix_drop_baked_into_cache"] = False
            shard.attrs["default_reset_prefix_drop_tokens"] = int(
                args.reset_prefix_drop_tokens
            )
            shard.attrs["future_window_tokens"] = manifest[
                "window_tokens"
            ]
            shard.create_dataset(
                "reset_future_codes",
                shape=(
                    rows.size,
                    num_codebooks,
                    manifest["window_tokens"],
                ),
                dtype=np.uint16,
                chunks=(
                    min(256, rows.size),
                    num_codebooks,
                    manifest["window_tokens"],
                ),
                **compression,
            )
            shard.create_dataset(
                "written",
                shape=(rows.size,),
                dtype=np.bool_,
                chunks=(min(4096, rows.size),),
            )
            shard.create_dataset(
                "manifest_row",
                data=rows.astype(np.int64),
            )


def _pending_manifest_rows(cache_dir: str, manifest) -> set[int]:
    pending = set()
    for shard_index, shard_name in enumerate(manifest["shard_names"]):
        path = os.path.join(cache_dir, shard_name)
        with h5py.File(path, "r") as shard:
            written = np.asarray(shard["written"][:], dtype=np.bool_)
            manifest_rows = np.asarray(
                shard["manifest_row"][:],
                dtype=np.int64,
            )
            pending.update(manifest_rows[~written].tolist())
    return pending


def _write_encoded_rows(
    cache_dir: str,
    manifest,
    manifest_rows: list[int],
    codes: np.ndarray,
) -> None:
    grouped = defaultdict(list)
    for source_index, manifest_row in enumerate(manifest_rows):
        shard_index = int(manifest["shard_index"][manifest_row])
        shard_row = int(manifest["shard_row"][manifest_row])
        grouped[shard_index].append(
            (shard_row, source_index)
        )
    for shard_index, entries in grouped.items():
        entries.sort()
        shard_rows = np.asarray(
            [entry[0] for entry in entries],
            dtype=np.int64,
        )
        source_rows = np.asarray(
            [entry[1] for entry in entries],
            dtype=np.int64,
        )
        path = os.path.join(
            cache_dir,
            manifest["shard_names"][shard_index],
        )
        with h5py.File(path, "r+") as shard:
            shard["reset_future_codes"][shard_rows] = codes[source_rows]
            shard["written"][shard_rows] = True
            shard.flush()


def _mark_completed(cache_dir: str, manifest) -> None:
    for shard_name in manifest["shard_names"]:
        path = os.path.join(cache_dir, shard_name)
        with h5py.File(path, "r+") as shard:
            if not np.asarray(shard["written"][:]).all():
                raise RuntimeError(
                    f"Cannot complete cache; shard still has gaps: {path}"
                )
            shard.attrs["completed"] = True
            shard.flush()


def build_cache(args) -> None:
    supported_dataset_ratios = {
        "full_beatx_lowervalid",
        "goodspk_beatx_lowervalid",
    }
    if args.dataset_ratio not in supported_dataset_ratios:
        raise ValueError(
            "Reset-future cache construction currently supports the "
            "lower-valid BEATX full and good-speaker subsets; got "
            f"dataset_ratio={args.dataset_ratio!r}."
        )
    if not args.reset_future_cache_dir:
        raise ValueError("--reset_future_cache_dir is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("Reset cache construction requires CUDA.")
    if int(args.pose_length) % int(args.frame_chunk_size):
        raise ValueError(
            "pose_length must align to the gesture-token frame size."
        )
    if int(args.future_gesture_horizon_frames) % int(
        args.frame_chunk_size
    ):
        raise ValueError(
            "Future offset must align to a gesture-token boundary."
        )

    cache_dir = os.path.abspath(args.reset_future_cache_dir)
    if (
        args.reset_future_cache_rebuild
        and os.path.isdir(cache_dir)
        and os.listdir(cache_dir)
    ):
        raise RuntimeError(
            "Refusing to recursively delete a populated cache directory. "
            "Choose a new --reset_future_cache_dir for a rebuild."
        )
    os.makedirs(cache_dir, exist_ok=True)
    metadata_path = os.path.join(cache_dir, METADATA_NAME)
    manifest_path = os.path.join(cache_dir, MANIFEST_NAME)
    signature = build_reset_future_cache_signature(args)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        if existing_metadata.get("build_signature") != signature:
            raise RuntimeError(
                "Existing cache metadata does not match this build."
            )

    manifest = _manifest_arrays(args)
    if os.path.exists(manifest_path):
        _validate_existing_manifest(
            manifest_path,
            manifest,
            build_signature=signature,
        )
    else:
        _write_manifest(
            manifest_path,
            manifest,
            build_signature=signature,
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    other_tools.set_random_seed(args)
    codecs = loaders.get_gesture_codecs(
        (
            args.upperbodycodec_ckpt,
            args.lowerbodycodec_ckpt,
            args.facecodec_ckpt,
        ),
        device=device,
        codec_kwargs=vars(args),
    )
    for codec in codecs:
        codec.eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
    codec_frame_sizes = {
        int(codec.frame_size) for codec in codecs
    }
    configured_frame_size = int(args.frame_chunk_size)
    if codec_frame_sizes != {configured_frame_size}:
        raise RuntimeError(
            "Gesture codec frame sizes must all match frame_chunk_size="
            f"{configured_frame_size}; got {sorted(codec_frame_sizes)}."
        )
    num_codebooks = sum(int(codec.num_codebooks) for codec in codecs)
    cardinalities = {int(codec.cardinality) for codec in codecs}
    if len(cardinalities) != 1:
        raise RuntimeError(
            f"Codec cardinalities differ: {cardinalities}."
        )
    cardinality = cardinalities.pop()
    if cardinality > np.iinfo(np.uint16).max + 1:
        raise RuntimeError("Codec codes do not fit uint16.")
    estimated_code_bytes = (
        int(manifest["target_token_index"].shape[0])
        * num_codebooks
        * int(manifest["window_tokens"])
        * np.dtype(np.uint16).itemsize
    )
    logger.info(
        "Reset cache plan: sequences={} targets={} window_tokens={} "
        "codebooks={} uncompressed_codes={:.2f} MiB",
        len(manifest["sequence_ids"]),
        int(manifest["target_token_index"].shape[0]),
        int(manifest["window_tokens"]),
        num_codebooks,
        estimated_code_bytes / (1024 ** 2),
    )

    _initialize_shards(
        cache_dir,
        manifest,
        num_codebooks=num_codebooks,
        cardinality=cardinality,
        build_signature=signature,
        args=args,
    )
    # Reuse the canonical serialized payload so large codec checkpoints are
    # hashed only once per builder invocation.
    metadata = json.loads(signature)
    metadata.update(
        {
            "build_signature": signature,
            "num_codebooks": num_codebooks,
            "cardinality": cardinality,
            "future_window_tokens": manifest["window_tokens"],
            "future_offset_tokens": manifest["offset_tokens"],
            "manifest_seed": int(args.reset_future_manifest_seed),
            "sequence_count": len(manifest["sequence_ids"]),
            "target_count": int(
                manifest["target_token_index"].shape[0]
            ),
            "shard_count": len(manifest["shard_names"]),
            "code_dtype": "uint16",
            "reset_prefix_drop_baked_into_cache": False,
            "default_reset_prefix_drop_tokens": int(
                args.reset_prefix_drop_tokens
            ),
            "git": _git_state(),
            "completed": False,
            "updated_at_unix": time.time(),
        }
    )
    _write_json(metadata_path, metadata)

    pending = _pending_manifest_rows(cache_dir, manifest)
    lower_velocity_start = int(args.lowertrans_nfeats) - 7
    if lower_velocity_start < 0:
        raise ValueError(
            "lowertrans_nfeats must include xyz velocity and four contacts."
        )
    if int(args.reset_future_cache_build_batch_size) <= 0:
        raise ValueError(
            "reset_future_cache_build_batch_size must be positive."
        )
    processed_batches = 0
    for split in args.reset_future_cache_splits:
        dataset = manifest["datasets"][split]
        loader = DataLoader(
            dataset,
            batch_size=int(args.reset_future_cache_build_batch_size),
            shuffle=False,
            num_workers=int(args.reset_future_cache_build_workers),
            drop_last=False,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            persistent_workers=(
                int(args.reset_future_cache_build_workers) > 0
            ),
        )
        sequence_lookup = {
            sequence_id: index
            for index, sequence_id in enumerate(manifest["sequence_ids"])
        }
        for batch in tqdm(
            loader,
            desc=f"encode reset windows:{split}",
            unit="batch",
        ):
            codec_inputs = None
            windows_by_part = [[], [], []]
            rows_to_encode = []
            for batch_index, sequence_id_value in enumerate(
                batch["filechunk_id"]
            ):
                sequence_id = str(sequence_id_value)
                sequence_index = sequence_lookup[sequence_id]
                row_start = int(
                    manifest["sequence_target_offsets"][sequence_index]
                )
                row_end = int(
                    manifest["sequence_target_offsets"][sequence_index + 1]
                )
                sequence_pending = [
                    row
                    for row in range(row_start, row_end)
                    if row in pending
                ]
                if not sequence_pending:
                    continue
                if codec_inputs is None:
                    codec_inputs = _codec_inputs(batch, device, args)
                for manifest_row in sequence_pending:
                    raw_start = int(
                        manifest["raw_future_start_frame"][manifest_row]
                    )
                    raw_end = int(
                        manifest["raw_future_end_frame"][manifest_row]
                    )
                    for part_index, motion in enumerate(codec_inputs):
                        window = motion[
                            batch_index,
                            raw_start:raw_end,
                        ].clone()
                        if part_index == 1:
                            window[
                                0,
                                lower_velocity_start:
                                lower_velocity_start + 3,
                            ] = 0
                        windows_by_part[part_index].append(window)
                    rows_to_encode.append(manifest_row)
            if not rows_to_encode:
                continue

            reset_parts = []
            with torch.inference_mode():
                for part_index, (codec, part_windows) in enumerate(
                    zip(codecs, windows_by_part)
                ):
                    windows = torch.stack(part_windows, dim=0)
                    encoding = encode_reset_suffix(
                        codec,
                        windows,
                        suffix_start_frame=0,
                        zero_first_frame_feature_slice=(
                            (
                                lower_velocity_start,
                                lower_velocity_start + 3,
                            )
                            if part_index == 1
                            else None
                        ),
                    )
                    reset_parts.append(encoding.codes)
            reset_codes = torch.cat(reset_parts, dim=1)
            expected_shape = (
                len(rows_to_encode),
                num_codebooks,
                manifest["window_tokens"],
            )
            if reset_codes.shape != expected_shape:
                raise RuntimeError(
                    f"Reset batch shape {tuple(reset_codes.shape)} does not "
                    f"match {expected_shape}."
                )
            reset_np = (
                reset_codes.detach().cpu().numpy().astype(np.uint16)
            )
            _write_encoded_rows(
                cache_dir,
                manifest,
                rows_to_encode,
                reset_np,
            )
            pending.difference_update(rows_to_encode)
            processed_batches += 1
            if (
                args.reset_future_cache_max_batches is not None
                and processed_batches
                >= int(args.reset_future_cache_max_batches)
            ):
                metadata["updated_at_unix"] = time.time()
                metadata["remaining_targets"] = len(pending)
                _write_json(metadata_path, metadata)
                logger.warning(
                    "Stopped early with {} targets remaining.",
                    len(pending),
                )
                return

    if pending:
        raise RuntimeError(
            f"Cache build ended with {len(pending)} unwritten targets."
        )
    _mark_completed(cache_dir, manifest)
    metadata["completed"] = True
    metadata["remaining_targets"] = 0
    metadata["updated_at_unix"] = time.time()
    _write_json(metadata_path, metadata)
    logger.info(
        "Fixed-window reset cache complete: dir={} sequences={} targets={} "
        "shards={} window_tokens={}",
        cache_dir,
        metadata["sequence_count"],
        metadata["target_count"],
        metadata["shard_count"],
        metadata["future_window_tokens"],
    )


if __name__ == "__main__":
    build_cache(config.parse_args())
