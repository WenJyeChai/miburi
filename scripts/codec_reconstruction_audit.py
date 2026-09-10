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
METRIC_SCOPES = ("all_frames", "valid_frames", "fully_valid_clips")
USAGE_SCOPE = "all_evaluated_frames"


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


def _read_sample_quality(dataset, index, settings, args):
    """Validate raw HDF5 structure, then classify finite inputs before loading.

    False pose_valid flags are a metric mask, never a reason to replace or
    remove a frame. Malformed masks and missing/incorrectly shaped fields are
    schema errors. Only NaN/Inf in required numerical inputs cause a clip skip.
    """
    import numpy as np
    import torch

    reference = dataset._chunk_refs[index]
    label = f"{dataset.loader_type}:{reference.chunk_id}"
    group = dataset._get_h5(reference.hdf5_path)[reference.chunk_id]

    def read(field):
        if field not in group:
            raise KeyError(f"Missing required HDF5 field {field!r} at {label}")
        try:
            values = np.asarray(group[field][...])
        except (OSError, TypeError) as error:
            raise RuntimeError(f"Cannot read HDF5 field {field!r} at {label} in {reference.hdf5_path}") from error
        if values.dtype.kind not in "buif":
            raise TypeError(f"HDF5 field {field!r} at {label} must contain real numeric values; got {values.dtype}")
        return values

    arrays = {field: read(field) for field in ("motion", "transl", "pose_valid", "betas")}
    motion = arrays["motion"]
    if motion.ndim != 3 or motion.shape[1:] != (55, 3):
        raise ValueError(f"motion at {label} must have shape [T,55,3]; got {motion.shape}")
    frames = motion.shape[0]
    if frames < 2 or frames % args.frame_chunk_size:
        raise ValueError(f"motion at {label} has {frames} frames; need at least two and a multiple of frame_chunk_size={args.frame_chunk_size}")
    if "fulllength" not in args.dataset_ratio and frames != args.pose_length:
        raise ValueError(f"motion at {label} has {frames} frames; expected pose_length={args.pose_length}")
    if settings.warmup_frames >= frames:
        raise ValueError(f"warmup_frames excludes every frame at {label}")
    if arrays["transl"].shape != (frames, 3):
        raise ValueError(f"transl at {label} must have shape [{frames},3]; got {arrays['transl'].shape}")
    pose_valid = arrays["pose_valid"]
    if pose_valid.shape != (frames,) or not np.isin(pose_valid, (0, 1)).all():
        raise ValueError(f"pose_valid at {label} must have shape [{frames}] with only boolean/0/1 values")
    betas = arrays["betas"]
    if not ((betas.ndim == 1 and betas.shape[0] > 0) or
            (betas.ndim == 2 and betas.shape[0] == frames and betas.shape[1] > 0)):
        raise ValueError(f"betas at {label} must be [C] or [{frames},C]; got {betas.shape}")
    if settings.geometry and betas.shape[-1] < int(getattr(args, "smplx_num_betas", 300)):
        raise ValueError(f"betas at {label} has too few coefficients for the geometry model")

    finite_fields = ["motion", "transl"]
    if "lower" in settings.parts:
        if int(reference.lower_valid) != 1:
            raise ValueError(f"Lower-body target marked invalid at {label}")
        arrays["contacts"] = read("contacts")
        if arrays["contacts"].shape != (frames, 4):
            raise ValueError(f"contacts at {label} must have shape [{frames},4]; got {arrays['contacts'].shape}")
        finite_fields.append("contacts")
    if "face" in settings.parts or settings.geometry:
        arrays["expressions"] = read("expressions")
        expression_dim = int(getattr(args, "smplx_num_expression_coeffs", 100)) if settings.geometry else 100
        if expression_dim != 100 or arrays["expressions"].shape != (frames, 100):
            raise ValueError(f"expressions at {label} must have shape [{frames},100] matching the current face codec/geometry model")
        finite_fields.append("expressions")
    if settings.geometry:
        finite_fields.append("betas")

    def text(value):
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    start = float(group.attrs.get("chunk_startsec", 0.0))
    end = float(group.attrs.get("chunk_endsec", 0.0))
    if not np.isfinite([start, end]).all() or end < start:
        raise ValueError(f"Invalid chunk time metadata at {label}: {start}, {end}")
    identity = dict(split=dataset.loader_type, filechunk_id=reference.chunk_id,
                    file_id=text(group.attrs.get("file_id", "")), dataset_name=reference.dataset_name,
                    chunk_startsec=start, chunk_endsec=end, frames=frames)
    # Complete all schema checks before allowing numerical corruption to skip.
    nonfinite = [field for field in finite_fields if not np.isfinite(arrays[field]).all()]
    valid_count = int(np.count_nonzero(pose_valid))
    quality = dict(identity, valid_frames=valid_count, flagged_frames=frames - valid_count,
                   valid_fraction=valid_count / frames, fully_valid=valid_count == frames,
                   status="skipped" if nonfinite else "evaluated",
                   reason="nonfinite_required_input" if nonfinite else "",
                   nonfinite_fields=nonfinite)
    return identity, torch.from_numpy(pose_valid.astype(np.bool_, copy=True)), quality


def _direct_velocity_valid_mask(pose_valid, warmup_frames):
    """Conservative validity for the production mixed finite-difference target."""
    effective = pose_valid.clone().bool()
    effective[:warmup_frames] = False
    stencil = effective.clone()
    stencil[0] &= effective[1]
    stencil[-1] &= effective[-2]
    if len(effective) > 2:
        stencil[1:-1] &= effective[:-2] & effective[2:]
    return stencil


def _quality_summary(split, available, records):
    evaluated = [record for record in records if record["status"] == "evaluated"]
    evaluated_frames = sum(record["frames"] for record in evaluated)
    valid_frames = sum(record["valid_frames"] for record in evaluated)
    return dict(split=split, available_clips=available, selected_clips=len(records),
                evaluated_clips=len(evaluated), skipped_clips=len(records) - len(evaluated),
                fully_valid_clips=sum(record["fully_valid"] for record in evaluated),
                flagged_clips=sum(not record["fully_valid"] for record in evaluated),
                selected_fully_valid_clips=sum(record["fully_valid"] for record in records),
                selected_flagged_clips=sum(not record["fully_valid"] for record in records),
                selected_frames=sum(record["frames"] for record in records), evaluated_frames=evaluated_frames,
                valid_frames=valid_frames, flagged_frames=evaluated_frames - valid_frames,
                valid_fraction=valid_frames / evaluated_frames if evaluated_frames else None)


def run_audit(settings: AuditSettings, progress: Callable[[dict], None] | None = None) -> dict:
    """Audit finite clips intact, with three explicit metric quality scopes.

    NaN/Inf in required raw inputs skips a whole clip, with its identity/reason
    retained. Missing fields, invalid shapes/masks and model errors remain fatal.
    ``progress`` receives split, completed, total, filechunk_id, status, reason
    and elapsed_seconds for both evaluated and skipped clips.
    Metric records are long-form; counts are the metric helper's actual element
    denominators, so split RMSE is computed from pooled squared error, not a mean
    of per-clip RMSE. A finite clip is encoded once per part; all_frames and
    valid_frames reuse that reconstruction. fully_valid_clips reuses all_frames
    statistics only if every original pose_valid flag was true. Usage includes
    every token from evaluated clips once, including flagged/warmup frames.
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
    result = dict(summary_records=[], sample_records=[], codebook_records=[], code_usage_records=[],
                  quality_records=[], quality_summary_records=[], skipped_records=[], metadata=metadata)
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
                split_manifest[split] = dict(available_samples=len(dataset),
                                            selected_ids=[dataset._chunk_refs[i].chunk_id for i in indices],
                                            evaluated_ids=[], skipped_ids=[])
                split_quality = []
                for completed, index in enumerate(indices, 1):
                    reference = dataset._chunk_refs[index]
                    identity, pose_valid, quality = _read_sample_quality(dataset, index, settings, args)
                    sample_manifest.append(identity)
                    result["quality_records"].append(quality)
                    split_quality.append(quality)
                    if quality["status"] == "skipped":
                        result["skipped_records"].append(dict(quality))
                        split_manifest[split]["skipped_ids"].append(reference.chunk_id)
                        if progress:
                            progress(dict(split=split, completed=completed, total=len(indices), filechunk_id=reference.chunk_id,
                                          status="skipped", reason=quality["reason"], nonfinite_fields=quality["nonfinite_fields"],
                                          elapsed_seconds=time.perf_counter() - started))
                        continue
                    sample = dataset[index]
                    if sample["filechunk_id"] != reference.chunk_id:
                        raise ValueError(f"Dataset substituted requested sample {reference.chunk_id}")
                    prepared = prepare_codec_inputs(sample, args, repository_root=root, device=settings.device, parts=settings.parts)
                    if prepared["translation"].shape[1] != identity["frames"]:
                        raise ValueError(f"Dataset changed frame count for {reference.chunk_id}")
                    velocity_valid = _direct_velocity_valid_mask(pose_valid, settings.warmup_frames)
                    velocity_all = _direct_velocity_valid_mask(torch.ones_like(pose_valid), settings.warmup_frames)
                    for part, codec in codecs.items():
                        decoded, codes = reconstruct_codec(codec, prepared["inputs"][part], mode=settings.mode)
                        kwargs = _metric_kwargs(part, prepared, decoded, args, rc, kinematics)
                        geometry_kwargs = _geometry_kwargs(part, sample, prepared, kwargs, dataset, rc) if geometry is not None else None

                        def measure(valid_mask=None, direct_velocity_mask=None):
                            stats = reconstruction_metrics(**kwargs, valid_mask=valid_mask,
                                translation_velocity_valid_mask=direct_velocity_mask,
                                warmup_frames=settings.warmup_frames, fps=args.motion_fps)
                            if geometry is not None:
                                geometry_stats = geometry.evaluate(**geometry_kwargs, valid_mask=valid_mask,
                                    warmup_frames=settings.warmup_frames, fps=args.motion_fps)
                                if set(stats) & set(geometry_stats):
                                    raise ValueError("Geometry and core metric names collide")
                                stats.update(geometry_stats)
                            return stats

                        all_stats = measure(direct_velocity_mask=velocity_all if part == "lower" else None)
                        # Fully valid clips have identical effective masks and
                        # velocity stencils, including warmup exclusions.
                        valid_stats = all_stats if quality["fully_valid"] else measure(
                            pose_valid, velocity_valid if part == "lower" else None)
                        # SMPL-X can return no keys when the complete mask is false.
                        for metric, template in all_stats.items():
                            valid_stats.setdefault(metric, dict(sum=0.0, count=0, reduction=template["reduction"]))
                        scoped_stats = dict(all_frames=all_stats, valid_frames=valid_stats)
                        if quality["fully_valid"]:
                            scoped_stats["fully_valid_clips"] = all_stats
                        for scope, stats in scoped_stats.items():
                            key = (split, part, scope)
                            merge_metric_sums(aggregates.setdefault(key, {}), stats)
                            sample_counts[key] = sample_counts.get(key, 0) + 1
                            for metric, value in finalize_metrics(stats).items():
                                result["sample_records"].append(dict(identity, part=part, mode=settings.mode,
                                    metric_scope=scope, metric=metric, value=value, count=stats[metric]["count"],
                                    reduction=stats[metric]["reduction"]))
                        ids = codes.detach().cpu().numpy()[0]
                        if ids.shape[0] != codec.num_codebooks or ids.min() < 0 or ids.max() >= codec.cardinality:
                            raise ValueError(f"Invalid {part} codec token IDs or stage count")
                        counts = histograms.setdefault((split, part), np.zeros((codec.num_codebooks, codec.cardinality), dtype=np.int64))
                        for stage, stage_ids in enumerate(ids):
                            counts[stage] += np.bincount(stage_ids, minlength=codec.cardinality)
                    split_manifest[split]["evaluated_ids"].append(reference.chunk_id)
                    if progress:
                        progress(dict(split=split, completed=completed, total=len(indices), filechunk_id=reference.chunk_id,
                                      status="evaluated", reason="", nonfinite_fields=[],
                                      elapsed_seconds=time.perf_counter() - started))
                result["quality_summary_records"].append(_quality_summary(split, len(dataset), split_quality))
            finally:
                dataset.close()
    finally:
        for part, codec in codecs.items():
            after = _fingerprint(codec)
            metadata["models"][part]["state_sha256_after"] = after
            if after != before[part]:
                raise RuntimeError(f"Frozen {part} codec parameters/buffers changed during audit")
    for (split, part, scope), stats in aggregates.items():
        for metric, value in finalize_metrics(stats).items():
            result["summary_records"].append(dict(split=split, part=part, metric=metric, value=value,
                metric_scope=scope, count=stats[metric]["count"], reduction=stats[metric]["reduction"],
                n_samples=sample_counts[(split, part, scope)], mode=settings.mode))
    for (split, part), all_counts in histograms.items():
        train_counts = histograms.get(("train", part))
        for stage, counts in enumerate(all_counts):
            total = int(counts.sum())
            probabilities = counts[counts > 0] / total
            entropy = float(-(probabilities * np.log(probabilities)).sum())
            unseen = None if train_counts is None else float(counts[train_counts[stage] == 0].sum() / total)
            result["codebook_records"].append(dict(split=split, part=part, stage=stage + 1, usage_scope=USAGE_SCOPE, codebook_size=len(counts),
                token_count=total, active_codes=int((counts > 0).sum()), active_fraction=float((counts > 0).mean()),
                entropy_nats=entropy, perplexity=float(np.exp(entropy)), unseen_train_mass=unseen))
            for code in np.flatnonzero(counts):
                result["code_usage_records"].append(dict(split=split, part=part, stage=stage + 1, usage_scope=USAGE_SCOPE, code=int(code), count=int(counts[code])))
    support = []
    for split in settings.splits:
        for part in settings.parts:
            for scope in METRIC_SCOPES:
                stats = aggregates.get((split, part, scope), {})
                supported = [metric for metric, value in stats.items() if value["count"] > 0]
                support.append(dict(split=split, part=part, metric_scope=scope,
                                    n_samples=sample_counts.get((split, part, scope), 0),
                                    metrics_with_support=supported,
                                    unsupported_metrics=[metric for metric, value in stats.items() if value["count"] == 0]))
    duplicates = {}
    for item in sample_manifest:
        duplicates.setdefault((item["dataset_name"], item["filechunk_id"]), set()).add(item["split"])
    metadata.update(created_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                    split_manifest=split_manifest, sample_manifest=sample_manifest,
                    cross_split_duplicate_ids=[dict(dataset_name=key[0], filechunk_id=key[1], splits=sorted(value)) for key, value in duplicates.items() if len(value) > 1],
                    cross_split_source_overlaps=_overlaps(sample_manifest),
                    state_unchanged=True, usage_includes_warmup=True, usage_scope=USAGE_SCOPE,
                    invalid_policy="skip_nonfinite_required_inputs_only; finite_pose_flags_are_metric_masks",
                    result_schema_version=2, quality_schema_version=2, metric_scopes=list(METRIC_SCOPES),
                    quality_summary_records=result["quality_summary_records"],
                    scope_support_records=support, no_support_scopes=[item for item in support if not item["metrics_with_support"]],
                    quality_policy={"encoding": "Encode every finite evaluated clip intact once per part; no frame removal or replacement.",
                                    "all_frames": "All evaluated frames after warmup, including finite pose_valid=False frames.",
                                    "valid_frames": "Only pose_valid=True frames after warmup; temporal differences require valid source-frame support.",
                                    "fully_valid_clips": "Reuse all_frames metrics only for clips with every original pose_valid flag true, before warmup exclusion.",
                                    "nonfinite": "Skip a whole clip only for NaN/Inf in required raw codec/geometry inputs; preserve selected identity and reason.",
                                    "schema_errors": "Missing fields, malformed shapes/flags, read failures and model failures are fatal.",
                                    "quality_summary_denominators": "fully_valid_clips/flagged_clips and valid_frames/flagged_frames/valid_fraction refer to evaluated clips before warmup; selected_* includes skipped clips.",
                                    "usage": "All tokens of evaluated clips counted once, including flagged and warmup frames; no validity filtering."},
                    translation_integration="production velocity2position_mixeddiff in successive frame_chunk_size blocks with carried final_pos",
                    index_cache_mode="off", padding_or_truncation=False,
                    notes=["Usage entropy/perplexity describe frozen codec assignments, not gesture-model predictive uncertainty.",
                           "A sampled training subset may understate training support; unseen_train_mass uses only evaluated training samples.",
                           "Quality flags do not remove encoder context: valid-frame predictions can depend on finite flagged neighboring frames.",
                           "Source overlap checks cover selected samples only and depend on source file/time metadata.",
                           "Translation position metrics include production velocity integration effects; direct velocity RMSE isolates velocity reconstruction."])
    return result
