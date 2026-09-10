"""Frozen codec reconstruction audit; no trainers, optimizers or output writes.

Use ``get_preflight(settings)`` before ``run_audit(settings, progress=callback)``.
The latter returns JSON-compatible long-form records suitable for pandas. Each
sample is read directly (no DataLoader cropping/padding/resampling), encoded with
all RVQ stages, and decoded with a fresh streaming state. ``mode='batched'`` is
an explicitly separate full-clip encode/decode experiment. Notebook callers own
all report writing. Imports and the real experiment parser are loaded lazily.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import types
from typing import Any, Callable


DEFAULT_CONFIG = "configs/gtdm3_teacher_c_face1_cos800_rvq_beatx_scott.yaml"
PARTS = ("upper", "lower", "face")
CHECKPOINT_FIELDS = dict(upper="upperbodycodec_ckpt", lower="lowerbodycodec_ckpt", face="facecodec_ckpt")


@dataclass
class AuditSettings:
    lm_config: str = DEFAULT_CONFIG
    repository_root: str | None = None
    device: str = "cuda:0"
    splits: tuple[str, ...] = ("train", "val")
    parts: tuple[str, ...] = PARTS
    max_samples_per_split: int | None = None
    seed: int = 2342
    mode: str = "streaming"
    geometry: bool = False
    warmup_frames: int = 0
    # Keys are existing parser fields, e.g. beatx_cache_path, deps_path,
    # upperbodycodec_ckpt, lowerbodycodec_ckpt, facecodec_ckpt.
    path_overrides: dict[str, str | None] = field(default_factory=dict)


def _root(settings: AuditSettings) -> Path:
    return Path(settings.repository_root or Path(__file__).resolve().parents[1]).expanduser().resolve()


def _path(value: str | Path, root: Path) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def _load_file(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _data_modules(root: Path):
    """Private package avoids importing trainers/__init__.py and its models."""
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    package = f"_codec_audit_data_{suffix}"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(root / "scripts/trainers/dataloaders")]
        module.__package__ = package
        sys.modules[package] = module
    dataset = importlib.import_module(package + ".unified_dataset")
    rotations = importlib.import_module(package + ".utils.rotation_conversions")
    kinematics = importlib.import_module(package + ".utils.motion_kinematics")
    if not getattr(dataset, "_audit_readonly_index", False):
        original = dataset._load_chunk_metadata

        def readonly_metadata(*args, **kwargs):
            # 'auto' may create NPZ/JSON/lock files alongside the source HDF5.
            kwargs["index_cache_mode"] = "off"
            return original(*args, **kwargs)

        dataset._load_chunk_metadata = readonly_metadata
        dataset._audit_readonly_index = True
    return dataset, rotations, kinematics


@contextmanager
def _argv(argv):
    original = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def parse_lm_config(settings: AuditSettings):
    """Use the production parser while preserving notebook/kernel arguments."""
    root = _root(settings)
    config_path = _path(settings.lm_config, root)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    name = "_codec_audit_config_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    module = _load_file(name, root / "scripts/trainers/utils/config.py")
    with _argv(["codec_reconstruction_audit", "--config", str(config_path), "--is_train", "False", "--wandb", "False"]):
        args = module.parse_args()
    allowed = {"beatx_cache_path", "embody3d_cache_path", "cache_path", "deps_path", "beatx_data_path", "embody3d_path", *CHECKPOINT_FIELDS.values()}
    for key, value in settings.path_overrides.items():
        if key not in allowed:
            raise ValueError(f"Unsupported path override {key!r}; allowed: {sorted(allowed)}")
        setattr(args, key, value)
    for key in allowed:
        value = getattr(args, key, None)
        if value:
            resolved = str(_path(value, root))
            # Production geometry code concatenates deps_path with smplx_2020.
            setattr(args, key, resolved + "/" if key == "deps_path" else resolved)
    args.body_part = "full"
    args.is_train = False
    args.wandb = False
    args.ddp = False
    args.debug = False
    args.varying_frame_length = False
    args.loader_workers = 0
    return args


def _validate(settings: AuditSettings):
    if settings.mode not in {"streaming", "batched"}:
        raise ValueError("mode must be 'streaming' or 'batched'")
    for values, allowed, label in ((settings.parts, PARTS, "parts"), (settings.splits, ("train", "val", "test"), "splits")):
        if not values or len(set(values)) != len(values) or set(values) - set(allowed):
            raise ValueError(f"Invalid {label}: {values}")
    if settings.max_samples_per_split is not None and settings.max_samples_per_split < 1:
        raise ValueError("max_samples_per_split must be positive or None")
    if settings.warmup_frames < 0:
        raise ValueError("warmup_frames must be nonnegative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_preflight(settings: AuditSettings) -> dict:
    """Resolve configuration and checkpoint hashes without constructing models."""
    _validate(settings)
    args = parse_lm_config(settings)
    root = _root(settings)
    checkpoints = {}
    for part in settings.parts:
        value = getattr(args, CHECKPOINT_FIELDS[part], None)
        path = Path(value) if value else None
        checkpoints[part] = {"path": str(path) if path else None, "exists": bool(path and path.is_file())}
        if path and path.is_file():
            checkpoints[part].update(size_bytes=path.stat().st_size, sha256=_sha256(path))
    caches = {key: {"path": getattr(args, key, None), "exists": bool(getattr(args, key, None) and Path(getattr(args, key)).is_file())}
              for key in ("beatx_cache_path", "embody3d_cache_path")}
    required_cache_fields = [field for fragment, field in (("beatx", "beatx_cache_path"), ("embody", "embody3d_cache_path"))
                             if fragment in args.dataset_ratio.lower()]
    missing = [f"{part} checkpoint: {item['path']}" for part, item in checkpoints.items() if not item["exists"]]
    missing += [f"{key}: {caches[key]['path']}" for key in required_cache_fields if not caches[key]["exists"]]
    if settings.geometry and not (Path(args.deps_path) / "smplx_2020").is_dir():
        missing.append(f"SMPL-X assets: {Path(args.deps_path) / 'smplx_2020'}")
    return dict(settings=json.loads(json.dumps(asdict(settings), default=str)), repository_root=str(root),
                lm_config=str(_path(settings.lm_config, root)), config_sha256=_sha256(_path(settings.lm_config, root)),
                dataset_ratio=args.dataset_ratio, pose_length=args.pose_length, motion_fps=args.motion_fps,
                frame_chunk_size=args.frame_chunk_size, checkpoints=checkpoints, caches=caches,
                required_cache_fields=required_cache_fields, missing_requirements=missing,
                requirements={"geometry": settings.geometry, "deps_path": args.deps_path,
                              "device": settings.device, "audio_or_language_model_required": False})


def _strip_checkpoint_prefixes(weights):
    normalized = {}
    for name, value in weights.items():
        while name.startswith(("module.", "m.")):
            name = name.split(".", 1)[1]
        if name in normalized:
            raise ValueError(f"Checkpoint prefix normalization collides at {name}")
        normalized[name] = value
    return normalized


def load_codecs(args, *, device="cpu", parts=PARTS) -> dict:
    """Exact BaseGLMTrainer.get_dep_model constructors; strict checkpoint load."""
    import torch
    from safetensors.torch import load_file
    from miburi.models import GestureMimiCodec, loaders

    result = {}
    for part in parts:
        is_face = part == "face"
        # Production deliberately uses get_uppergesturecodec_kwargs for lower.
        kwargs = copy.deepcopy(loaders.get_facegesturecodec_kwargs() if is_face else loaders.get_uppergesturecodec_kwargs())
        nfeats = getattr(args, dict(upper="upperlower_nfeats", lower="lowertrans_nfeats", face="face_nfeats")[part])
        model = GestureMimiCodec(num_frames=args.num_frames, frame_chunk_size=args.frame_chunk_size,
                                 nfeats=nfeats, motion_fps=args.motion_fps,
                                 num_heads=args.transformer_heads // (2 if is_face else 1),
                                 transformer_layers=args.transformer_layers // (2 if is_face else 1),
                                 convblock_layers=args.convblock_layers, **kwargs)
        path = getattr(args, CHECKPOINT_FIELDS[part], None)
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Missing {part} codec checkpoint: {path}")
        model.load_state_dict(_strip_checkpoint_prefixes(load_file(path, device="cpu")), strict=True)
        model.requires_grad_(False).eval().to(device=torch.device(device))
        result[part] = model
    return result


def _fingerprint(model) -> str:
    import torch
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def prepare_codec_inputs(sample: dict, args, *, repository_root=None, device="cpu", parts=PARTS) -> dict:
    """Production rotation order and lower mixed-difference velocity features."""
    import torch
    _, rc, kinematics = _data_modules(Path(repository_root or Path(__file__).resolve().parents[1]))
    pose = {}
    for name in ("upper", "hands", "lower", "face"):
        value = sample["motion_" + name].to(device=device, dtype=torch.float32)
        if value.ndim == 2:
            value = value.reshape(value.shape[0], -1, 3)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"Unexpected {name} motion shape: {tuple(value.shape)}")
        pose[name] = rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(value)).reshape(1, value.shape[0], -1)
    transl = sample["transl"].to(device=device, dtype=torch.float32).unsqueeze(0).clone()
    if transl.shape[1] < 2:
        raise ValueError("At least two frames are required for production translation velocity")
    transl[:, :, (0, 2)] -= transl[:, :1, (0, 2)].clone()
    velocity = kinematics.estimate_linear_velocity(transl, 1 / args.motion_fps)
    contact = sample.get("contact")
    expressions = sample.get("expressions")
    if ("lower" in parts and contact is None) or ("face" in parts and expressions is None):
        raise ValueError("Requested codec has missing contact/expression targets; these cannot be replaced with zeros")
    contact = contact.to(device=device, dtype=torch.float32).unsqueeze(0) if contact is not None else None
    expressions = expressions.to(device=device, dtype=torch.float32).unsqueeze(0) if expressions is not None else None
    inputs = {}
    if "upper" in parts:
        inputs["upper"] = torch.cat((pose["upper"], pose["hands"]), -1)
    if "lower" in parts:
        inputs["lower"] = torch.cat((pose["lower"], velocity, contact), -1)
    if "face" in parts:
        inputs["face"] = torch.cat((pose["face"], expressions), -1)
    expected = {"upper": 258, "lower": 61, "face": 106}
    for part, value in inputs.items():
        if value.shape != (1, transl.shape[1], expected[part]) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {part} features: {tuple(value.shape)}; expected [1,T,{expected[part]}] and finite values")
    return dict(inputs=inputs, translation=transl, velocity=velocity, expressions=expressions,
                joint_counts={key: value.shape[-1] // 6 for key, value in pose.items()})


def reconstruct_codec(model, inputs, *, mode="streaming"):
    """Return ``(decoded[B,T,D], codes[B,K,T_tokens])`` without changing weights."""
    import torch
    if mode not in {"streaming", "batched"}:
        raise ValueError(mode)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Codec must be frozen and in eval mode")
    if getattr(model, "_streaming_state", None) is not None:
        raise ValueError("Codec already has an active streaming state")
    frames = inputs.shape[1]
    chunk = model.frame_chunk_size
    if frames == 0 or frames % chunk:
        raise ValueError(f"{frames} frames are not divisible by codec chunk size {chunk}; refusing to drop/pad frames")
    channels = model.channels
    channels = channels[0] if isinstance(channels, (tuple, list)) else channels
    if inputs.shape[-1] != channels:
        raise ValueError(f"Codec expects {channels} features, received {inputs.shape[-1]}")
    with torch.no_grad():
        if mode == "batched":
            codes = model.encode(inputs)
            decoded = model.decode(codes)
        else:
            all_codes, all_decoded = [], []
            with model.streaming(batch_size=inputs.shape[0]):
                for offset in range(0, frames, chunk):
                    code = model.encode(inputs[:, offset:offset + chunk])
                    out = model.decode(code)
                    if code.shape[-1] != 1 or out.shape[1] != chunk:
                        raise ValueError("Codec streaming alignment differs from its declared frame chunk size")
                    all_codes.append(code.clone())
                    all_decoded.append(out.clone())
            codes = torch.cat(all_codes, -1)
            decoded = torch.cat(all_decoded, 1)
    if decoded.shape != inputs.shape or codes.shape[-1] != frames // chunk:
        raise ValueError("Encode/decode frame alignment changed; no truncation is permitted")
    if not torch.isfinite(decoded).all():
        raise ValueError("Non-finite codec reconstruction")
    return decoded, codes


def _integrate_velocity(velocity, initial, *, chunk_size, fps, kinematics):
    import torch
    chunks = []
    carry = initial
    for offset in range(0, velocity.shape[1], chunk_size):
        positions, carry = kinematics.velocity2position_mixeddiff(velocity[:, offset:offset + chunk_size], 1 / fps, carry)
        chunks.append(positions)
    return torch.cat(chunks, 1)


def _metric_kwargs(part, prepared, decoded, args, rc, kinematics):
    counts = prepared["joint_counts"]
    joints = counts["upper"] + counts["hands"] if part == "upper" else counts[part]
    target = prepared["inputs"][part]
    as_rot = lambda value: rc.rotation_6d_to_matrix(value[:, :, :joints * 6].reshape(1, value.shape[1], joints, 6))
    regions = {"upper": list(range(counts["upper"])), "hands": list(range(counts["upper"], joints))} if part == "upper" else {"jaw" if part == "face" else "lower": list(range(joints))}
    kwargs = dict(target_rotations=as_rot(target), reconstructed_rotations=as_rot(decoded), rotation_regions=regions)
    if part == "face":
        kwargs.update(target_expressions=target[:, :, joints * 6:], reconstructed_expressions=decoded[:, :, joints * 6:])
    if part == "lower":
        velocity = decoded[:, :, joints * 6:joints * 6 + 3]
        reconstructed_translation = _integrate_velocity(velocity, prepared["translation"][:, 0], chunk_size=args.frame_chunk_size, fps=args.motion_fps, kinematics=kinematics)
        kwargs.update(target_translation=prepared["translation"], reconstructed_translation=reconstructed_translation,
                      target_translation_velocity=prepared["velocity"], reconstructed_translation_velocity=velocity,
                      target_contacts=target[:, :, joints * 6 + 3:], reconstructed_contacts=decoded[:, :, joints * 6 + 3:])
    return kwargs


def _geometry_kwargs(part, sample, prepared, metric_kwargs, dataset, rc):
    import torch
    target_pose = sample["motion"].to(prepared["translation"].device).unsqueeze(0)
    reconstructed_pose = target_pose.clone()
    mask = dataset.face_mask if part == "face" else dataset.lower_mask if part == "lower" else dataset.upper_mask + dataset.hands_mask
    indices = [i for i, value in enumerate(mask) if value]
    part_pose = rc.matrix_to_axis_angle(metric_kwargs["reconstructed_rotations"])
    if part == "upper":
        # Codec ordering is upper first then hands; SMPL-X indices interleave.
        indices = [i for i, value in enumerate(dataset.upper_mask) if value] + [i for i, value in enumerate(dataset.hands_mask) if value]
    reconstructed_pose[:, :, indices] = part_pose
    if not torch.isfinite(target_pose).all() or not torch.isfinite(sample["beta"]).all():
        raise ValueError(f"Non-finite geometry pose or beta in {sample['filechunk_id']}")
    if prepared["expressions"] is None or not torch.isfinite(prepared["expressions"]).all():
        raise ValueError(f"Missing/non-finite geometry expression in {sample['filechunk_id']}")
    return dict(target_pose_aa=target_pose, reconstructed_pose_aa=reconstructed_pose,
                betas=sample["beta"].to(target_pose.device).unsqueeze(0),
                target_expressions=prepared["expressions"],
                reconstructed_expressions=metric_kwargs.get("reconstructed_expressions", prepared["expressions"]),
                target_translation=prepared["translation"],
                reconstructed_translation=metric_kwargs.get("reconstructed_translation", prepared["translation"]),
                joint_regions={part: indices}, include_joints=(part != "face"), include_face_vertices=(part == "face"))


def _overlaps(samples):
    overlaps = []
    by_source = {}
    for sample in samples:
        by_source.setdefault((sample["dataset_name"], sample["file_id"]), []).append(sample)
    for (dataset, file_id), group in by_source.items():
        if not file_id:
            continue
        group.sort(key=lambda item: item["chunk_startsec"])
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                if right["chunk_startsec"] >= left["chunk_endsec"]:
                    break
                if left["split"] != right["split"]:
                    overlaps.append(dict(dataset_name=dataset, file_id=file_id,
                                         left_split=left["split"], right_split=right["split"],
                                         left_id=left["filechunk_id"], right_id=right["filechunk_id"]))
    return overlaps


def run_audit(settings: AuditSettings, progress: Callable[[dict], None] | None = None) -> dict:
    """Run the audit. Fail on invalid/missing inputs; never resample or write files.

    ``progress`` receives split, completed, total, filechunk_id and elapsed_seconds.
    Metric records are long-form; counts are the metric helper's actual element
    denominators, so split RMSE is computed from pooled squared error, not a mean
    of per-clip RMSE. Usage histograms include all valid tokens, including warmup.
    """
    import numpy as np
    import torch
    from scripts.codec_reconstruction_metrics import reconstruction_metrics, finalize_metrics, merge_metric_sums

    _validate(settings)
    started = time.perf_counter()
    metadata = get_preflight(settings)
    if metadata["missing_requirements"]:
        raise FileNotFoundError("Audit preflight is missing required inputs (no models were loaded):\n" + "\n".join(metadata["missing_requirements"]))
    args = parse_lm_config(settings)
    root = _root(settings)
    dataset_module, rc, kinematics = _data_modules(root)
    codecs = load_codecs(args, device=settings.device, parts=settings.parts)
    before = {part: _fingerprint(codec) for part, codec in codecs.items()}
    metadata["models"] = {part: dict(parameter_count=sum(p.numel() for p in codec.parameters()),
                                           channels=codec.channels, codebooks=codec.num_codebooks,
                                           cardinality=codec.cardinality, frame_rate=codec.frame_rate,
                                           latent_dtype=str(codec.latent_dtype), state_sha256_before=before[part])
                          for part, codec in codecs.items()}
    geometry = None
    if settings.geometry:
        from scripts.codec_reconstruction_metrics import SMPLXGeometryEvaluator
        geometry = SMPLXGeometryEvaluator(args, device=settings.device)
    result = dict(summary_records=[], sample_records=[], codebook_records=[], code_usage_records=[], metadata=metadata)
    aggregates, sample_counts, histograms = {}, {}, {}
    sample_manifest, split_manifest = [], {}
    try:
        for split in settings.splits:
            dataset = dataset_module.UNIFIEDDataset(args, split, only_motion=True, dataset_ratio=args.dataset_ratio,
                                                     varying_frame_length=False, debug=False,
                                                     runtime_quality_max_resample_attempts=1)
            try:
                indices = sorted(range(len(dataset)), key=lambda i: dataset._chunk_refs[i].chunk_id)
                if settings.max_samples_per_split is not None:
                    random.Random(f"{settings.seed}:{split}").shuffle(indices)
                    indices = indices[:settings.max_samples_per_split]
                split_manifest[split] = dict(available_samples=len(dataset), selected_ids=[dataset._chunk_refs[i].chunk_id for i in indices])
                for completed, index in enumerate(indices, 1):
                    reference = dataset._chunk_refs[index]
                    if not dataset._chunk_is_valid(index):
                        raise ValueError(f"Invalid pose_valid/motion at {split}:{reference.chunk_id}; audit refuses replacement")
                    sample = dataset[index]
                    if sample["filechunk_id"] != reference.chunk_id:
                        raise ValueError(f"Dataset substituted requested sample {reference.chunk_id}")
                    prepared = prepare_codec_inputs(sample, args, repository_root=root, device=settings.device, parts=settings.parts)
                    frames = prepared["translation"].shape[1]
                    if settings.warmup_frames >= frames:
                        raise ValueError(f"warmup_frames excludes every frame of {reference.chunk_id}")
                    identity = dict(split=split, filechunk_id=reference.chunk_id, file_id=sample["file_id"],
                                    dataset_name=sample["dataset_name"], chunk_startsec=float(sample["chunk_startsec"]),
                                    chunk_endsec=float(sample["chunk_endsec"]), frames=frames)
                    sample_manifest.append(identity)
                    for part, codec in codecs.items():
                        if part == "lower" and not bool(sample["lower_valid_mask"]):
                            raise ValueError(f"Lower-body target marked invalid: {reference.chunk_id}")
                        decoded, codes = reconstruct_codec(codec, prepared["inputs"][part], mode=settings.mode)
                        kwargs = _metric_kwargs(part, prepared, decoded, args, rc, kinematics)
                        stats = reconstruction_metrics(**kwargs, warmup_frames=settings.warmup_frames, fps=args.motion_fps)
                        if geometry is not None:
                            geometry_stats = geometry.evaluate(**_geometry_kwargs(part, sample, prepared, kwargs, dataset, rc),
                                                               warmup_frames=settings.warmup_frames, fps=args.motion_fps)
                            if set(stats) & set(geometry_stats):
                                raise ValueError("Geometry and core metric names collide")
                            stats.update(geometry_stats)
                        key = (split, part)
                        merge_metric_sums(aggregates.setdefault(key, {}), stats)
                        sample_counts[key] = sample_counts.get(key, 0) + 1
                        values = finalize_metrics(stats)
                        for metric, value in values.items():
                            result["sample_records"].append(dict(identity, part=part, mode=settings.mode,
                                metric=metric, value=value, count=stats[metric]["count"], reduction=stats[metric]["reduction"]))
                        ids = codes.detach().cpu().numpy()[0]
                        if ids.shape[0] != codec.num_codebooks or ids.min() < 0 or ids.max() >= codec.cardinality:
                            raise ValueError(f"Invalid {part} codec token IDs or stage count")
                        counts = histograms.setdefault(key, np.zeros((codec.num_codebooks, codec.cardinality), dtype=np.int64))
                        for stage, stage_ids in enumerate(ids):
                            counts[stage] += np.bincount(stage_ids, minlength=codec.cardinality)
                    if progress:
                        progress(dict(split=split, completed=completed, total=len(indices), filechunk_id=reference.chunk_id,
                                      elapsed_seconds=time.perf_counter() - started))
            finally:
                dataset.close()
    finally:
        for part, codec in codecs.items():
            after = _fingerprint(codec)
            metadata["models"][part]["state_sha256_after"] = after
            if after != before[part]:
                raise RuntimeError(f"Frozen {part} codec parameters/buffers changed during audit")
    for (split, part), stats in aggregates.items():
        for metric, value in finalize_metrics(stats).items():
            result["summary_records"].append(dict(split=split, part=part, metric=metric, value=value,
                count=stats[metric]["count"], reduction=stats[metric]["reduction"], n_samples=sample_counts[(split, part)], mode=settings.mode))
    for (split, part), all_counts in histograms.items():
        train_counts = histograms.get(("train", part))
        for stage, counts in enumerate(all_counts):
            total = int(counts.sum())
            probabilities = counts[counts > 0] / total
            entropy = float(-(probabilities * np.log(probabilities)).sum())
            unseen = None if train_counts is None else float(counts[train_counts[stage] == 0].sum() / total)
            result["codebook_records"].append(dict(split=split, part=part, stage=stage + 1, codebook_size=len(counts),
                token_count=total, active_codes=int((counts > 0).sum()), active_fraction=float((counts > 0).mean()),
                entropy_nats=entropy, perplexity=float(np.exp(entropy)), unseen_train_mass=unseen))
            for code in np.flatnonzero(counts):
                result["code_usage_records"].append(dict(split=split, part=part, stage=stage + 1, code=int(code), count=int(counts[code])))
    duplicates = {}
    for item in sample_manifest:
        duplicates.setdefault((item["dataset_name"], item["filechunk_id"]), set()).add(item["split"])
    metadata.update(created_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                    split_manifest=split_manifest, sample_manifest=sample_manifest,
                    cross_split_duplicate_ids=[dict(dataset_name=key[0], filechunk_id=key[1], splits=sorted(value)) for key, value in duplicates.items() if len(value) > 1],
                    cross_split_source_overlaps=_overlaps(sample_manifest),
                    state_unchanged=True, usage_includes_warmup=True, invalid_policy="fail_without_substitution",
                    translation_integration="production velocity2position_mixeddiff in successive frame_chunk_size blocks with carried final_pos",
                    index_cache_mode="off", padding_or_truncation=False,
                    notes=["Usage entropy/perplexity describe frozen codec assignments, not gesture-model predictive uncertainty.",
                           "A sampled training subset may understate training support; unseen_train_mass uses only evaluated training samples.",
                           "Source overlap checks cover selected samples only and depend on source file/time metadata.",
                           "Translation position metrics include production velocity integration effects; direct velocity RMSE isolates velocity reconstruction."])
    return result
