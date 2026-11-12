"""Evaluate saved ForeMotion motion on semantic-present and beat-only time.

Run from the repository root; see docs/gesture_category_evaluation.md. Uses
ordinary test.py gt.npz/pred.npz exports, local assets and BEAT sem/ annotations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import gesture_category_composition as annotations

MODELS = ("baseline", "foremotion")
CATEGORIES = ("semantic", "beat")
EVAL_FPS = 30
UPPER = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
HANDS = tuple(range(25, 55))
REGIONS = {"upper_hands": UPPER + HANDS, "upper_body": UPPER, "hands": HANDS}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames=None):
    if rows or fieldnames:
        annotations.write_csv(Path(path), rows, fieldnames)


def metric_module():
    """Import existing metric helpers without trainers/__init__ or model downloads."""
    name = "_gesture_category_metric_utils"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / "scripts/trainers/utils")]
        sys.modules[name] = package
    return importlib.import_module(name + ".metrics")


def category_segments(rows, start, end):
    """Disjoint exact-time segments. Unknown labels exclude all overlapping time."""
    events = defaultdict(Counter)
    events[start]; events[end]
    for row in rows:
        lo, hi = max(start, row.start), min(end, row.end)
        if hi > lo:
            events[lo][row.category] += 1
            events[hi][row.category] -= 1
    active = Counter()
    previous = start
    result = []
    for timestamp in sorted(events):
        known = {c for c in annotations.CATEGORIES if active[c] > 0}
        extra = {c for c, count in active.items() if count > 0 and c not in {*annotations.CATEGORIES, "none"}}
        label = ("excluded" if extra else "semantic" if known - {"beat"} else
                 "beat" if "beat" in known else "none" if active["none"] > 0 else "unlabeled")
        if timestamp > previous:
            if result and result[-1][2] == label and result[-1][1] == previous:
                result[-1] = (result[-1][0], timestamp, label)
            else:
                result.append((previous, timestamp, label))
        active.update(events[timestamp])
        previous = timestamp
    return result


def frame_weights(segments, frames, fps=EVAL_FPS):
    """Integrate exact overlap with frame bins [t/fps,(t+1)/fps)."""
    result = {category: np.zeros(frames, dtype=np.float64) for category in CATEGORIES}
    for start, end, category in segments:
        if category not in result:
            continue
        lo, hi = max(0, int(math.floor(start * fps))), min(frames, int(math.ceil(end * fps)))
        if hi > lo:
            times = np.arange(lo, hi) / fps
            result[category][lo:hi] += np.maximum(0, np.minimum(times + 1 / fps, end) - np.maximum(times, start)) * fps
    for category in result:
        result[category] = np.clip(result[category], 0, 1)
    if np.any(result["semantic"] + result["beat"] > 1 + 1e-8):
        raise ValueError("Category masks overlap")
    return result


def point_categories(times, segments):
    result = np.full(len(times), "", dtype="U8")
    for start, end, category in segments:
        if category in CATEGORIES:
            result[(times >= start) & (times < end)] = category
    return result


def select_windows(weights, frames=32, stride=32, purity=0.8, excluded_intervals=()):
    """Select continuous windows using only the reference annotation labels."""
    result = []
    length = len(weights["semantic"])
    for start in range(0, length - frames + 1, stride):
        if any(lo < (start + frames) / EVAL_FPS and hi > start / EVAL_FPS for lo, hi in excluded_intervals):
            continue
        for category in CATEGORIES:
            fraction = float(weights[category][start:start + frames].mean())
            if fraction >= purity - 1e-12:
                result.append((start, start + frames, category, fraction))
    return result


def resolve_pairs(cli, recordings):
    selected = {r.recording_id for r in recordings}
    result = {}
    if cli.pairs_manifest:
        base = cli.pairs_manifest.resolve().parent
        rows = annotations._table(cli.pairs_manifest)
        for row in rows:
            fid = row.get("recording_id", "")
            if fid not in selected:
                continue
            seed = row.get("seed", "").strip()
            if not seed:
                raise ValueError("Each motion-manifest row needs a generation seed")
            pair = {"recording_id": fid, "seed": seed}
            for model in MODELS:
                raw = row.get(model + "_path") or (row.get("prediction_path") if model == "foremotion" else None)
                if not raw:
                    if model == "baseline":
                        continue
                    raise ValueError(f"Missing prediction_path or foremotion_path for {fid}")
                path = Path(raw).expanduser()
                pair[model] = (path if path.is_absolute() else base / path).resolve()
                raw_reference = row.get(model + "_reference_path") or row.get("reference_path")
                reference = Path(raw_reference).expanduser() if raw_reference else pair[model].parent / "gt.npz"
                pair[model + "_reference"] = (reference if reference.is_absolute() else base / reference).resolve()
            key = (seed, fid)
            if key in result:
                raise ValueError(f"Duplicate model pair: {key}")
            result[key] = pair
    else:
        for recording in recordings:
            pair = {"recording_id": recording.recording_id, "seed": str(cli.generation_seed)}
            for model in MODELS:
                if getattr(cli, model + "_dir") is None:
                    continue
                folder = (getattr(cli, model + "_dir") / recording.recording_id).resolve()
                pair[model], pair[model + "_reference"] = folder / "pred.npz", folder / "gt.npz"
            result[pair["seed"], recording.recording_id] = pair
    if not result:
        raise ValueError("No prediction exports match the selected recordings")
    for seed in sorted({key[0] for key in result}):
        actual = {fid for s, fid in result if s == seed}
        if actual != selected:
            raise ValueError(f"Seed {seed} is missing selected recordings: {sorted(selected - actual)}")
    model_sets = {tuple(model for model in MODELS if model in pair) for pair in result.values()}
    if len(model_sets) != 1 or not all("foremotion" in models for models in model_sets):
        raise ValueError("Every selected recording/seed must provide the same models; baseline paths are optional but cannot be partial")
    models = next(iter(model_sets))
    for pair in result.values():
        if "baseline" in pair and pair["baseline"] == pair["foremotion"]:
            raise ValueError("Baseline and ForeMotion must point to separate prediction files")
        for key in (*models, *(model + "_reference" for model in models)):
            if not pair[key].is_file():
                raise FileNotFoundError(f"Missing {key} motion for {pair['recording_id']}: {pair[key]}; generate test.py exports with --save True")
    return result


def read_motion(path):
    with np.load(path, allow_pickle=False) as data:
        pose = np.asarray(data["poses"], dtype=np.float32)
        if pose.ndim == 2 and pose.shape[1] == 165:
            pose = pose.reshape(-1, 55, 3)
        if pose.ndim != 3 or pose.shape[1:] != (55, 3) or len(pose) < 2:
            raise ValueError(f"{path}: expected at least two frames of 55 axis-angle joints")
        beta = np.asarray(data["betas"], dtype=np.float32)
        if beta.ndim == 2:
            if len(beta) not in (1, len(pose)) or not np.allclose(beta, beta[:1], atol=1e-6, rtol=0):
                raise ValueError(f"{path}: expected a constant body shape")
            beta = beta[0]
        if beta.shape != (300,):
            raise ValueError(f"{path}: expected 300 SMPL-X shape coefficients")
        result = {"poses": pose, "betas": beta,
                  "declared_fps": float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else None}
        for key in ("trans", "expressions"):
            if key in data:
                result[key] = np.asarray(data[key], dtype=np.float32)
    for key, values in result.items():
        if isinstance(values, np.ndarray) and not np.isfinite(values).all():
            raise ValueError(f"{path}: nonfinite {key}")
    return result


def read_pair(pair, recording, input_fps, max_tail_frames):
    models = tuple(model for model in MODELS if model in pair)
    motions = {model: read_motion(pair[model]) for model in models}
    references = {model: read_motion(pair[model + "_reference"]) for model in models}
    reference = references[models[0]]
    for model in models:
        other = references[model]
        for key in ("poses", "betas", "trans", "expressions"):
            if (key in reference) != (key in other):
                raise ValueError(f"{recording.recording_id}: reference fields differ across models ({key})")
            if key in reference and (reference[key].shape != other[key].shape or not np.allclose(reference[key], other[key], atol=1e-6, rtol=0)):
                raise ValueError(f"{recording.recording_id}: models have different ground truth ({key})")
        if motions[model]["poses"].shape != reference["poses"].shape:
            raise ValueError(f"{recording.recording_id}: prediction/reference frame counts differ; no temporal warping or automatic truncation")
        if not np.allclose(motions[model]["betas"], reference["betas"], atol=1e-6, rtol=0):
            raise ValueError(f"{recording.recording_id}: predicted and reference body shapes differ")
    if recording.start_seconds != 0:
        raise ValueError("Prediction exports must start at recording time zero")
    expected = (recording.end_seconds - recording.start_seconds) * input_fps
    missing = expected - len(reference["poses"])
    if missing < -1.001 or missing > max_tail_frames + 1e-3:
        raise ValueError(f"{recording.recording_id}: motion duration disagrees with evaluation bounds at --input-fps {input_fps}; difference={missing:.3f} frames")
    return motions, reference


class MotionEvaluator:
    """Local frozen SMPL-X, optional BEAT feature encoder and rhythm helpers."""

    def __init__(self, cli):
        import torch
        import smplx

        required = [("SMPL-X assets", cli.smplx_path)]
        if "fgd" in cli.metrics:
            required.append(("FGD weights", cli.fgd_weights))
        if "beat" in cli.metrics:
            required.append(("mean velocity", cli.mean_velocity))
        for name, path in required:
            if not path.exists():
                raise FileNotFoundError(f"Missing local {name}: {path}")
        self.cli, self.torch = cli, torch
        self.device = torch.device(cli.device)
        self.helpers = metric_module()
        self.body = smplx.create(str(cli.smplx_path), model_type="smplx", gender="NEUTRAL_2020",
                                 ext="npz", use_pca=False, flat_hand_mean=True, num_betas=300,
                                 num_expression_coeffs=100).to(self.device).eval().requires_grad_(False)
        self.encoder = None
        if "fgd" in cli.metrics:
            config = types.SimpleNamespace(vae_length=240, vae_test_dim=330, variational=False,
                                           vae_layer=4, vae_grow=[1, 1, 2, 1],
                                           data_path_1=str(cli.smplx_path.resolve()) + "/")
            self.encoder = self.helpers.VAESKConv(config)
            from scripts.codec_reconstruction_audit import _strip_checkpoint_prefixes
            state = torch.load(cli.fgd_weights, map_location="cpu", weights_only=True)["model_state"]
            self.encoder.load_state_dict(_strip_checkpoint_prefixes(state), strict=True)
            self.encoder.to(self.device).eval().requires_grad_(False)
        if "beat" in cli.metrics:
            self.mean_velocity = np.load(cli.mean_velocity).reshape(-1)
            if (self.mean_velocity.shape != (55,) or not np.isfinite(self.mean_velocity).all()
                    or np.any(self.mean_velocity < 0) or np.any(self.mean_velocity[list(UPPER)] <= 0)):
                raise ValueError("Beat alignment mean-velocity file needs 55 nonnegative finite values, positive for upper-body joints")
            # Static unscored joints can have zero reference velocity.
            self.mean_velocity = np.where(self.mean_velocity > 0, self.mean_velocity, 1)

    def resample(self, poses, input_fps):
        if input_fps == EVAL_FPS:
            return poses.copy()
        return self.helpers.match_motion_fps(poses.copy(), source_fps=input_fps, target_fps=EVAL_FPS).astype(np.float32)

    def joints(self, poses, beta):
        torch = self.torch
        pieces = []
        with torch.no_grad():
            for start in range(0, len(poses), self.cli.chunk_size):
                pose = torch.as_tensor(poses[start:start + self.cli.chunk_size], device=self.device)
                n = len(pose)
                zero = torch.zeros(n, 3, device=self.device)
                out = self.body(betas=torch.as_tensor(beta, device=self.device)[None].expand(n, -1),
                                global_orient=pose[:, 0], body_pose=pose[:, 1:22].reshape(n, 63),
                                left_hand_pose=pose[:, 25:40].reshape(n, 45), right_hand_pose=pose[:, 40:55].reshape(n, 45),
                                jaw_pose=pose[:, 22], leye_pose=pose[:, 23], reye_pose=pose[:, 24],
                                expression=torch.zeros(n, 100, device=self.device), transl=zero, return_verts=False)
                pieces.append(out.joints[:, :55].cpu().numpy())
        joints = np.concatenate(pieces)
        return joints - joints[:, :1]  # pelvis-relative; preserve orientation and articulation

    def features(self, poses, reference, windows):
        torch = self.torch
        if not windows:
            return []
        all_features = []
        with torch.no_grad():
            for offset in range(0, len(windows), self.cli.feature_batch_size):
                group = windows[offset:offset + self.cli.feature_batch_size]
                values = np.stack([poses[lo:hi] for lo, hi, _, _ in group]).copy()
                # Existing GestureMetrics protocol substitutes the reference jaw.
                values[:, :, 22] = np.stack([reference[lo:hi, 22] for lo, hi, _, _ in group])
                rotations = self.helpers.axis_angle_to_matrix(torch.as_tensor(values, device=self.device))
                features = self.helpers.matrix_to_rotation_6d(rotations).reshape(len(group), self.cli.window_frames, 330)
                latent = self.encoder.map2latent(features).reshape(len(group), -1, 240).cpu().numpy()
                all_features.extend(latent)
        return all_features

    def audio_onsets(self, recording_id, duration):
        import librosa

        path = self.cli.data_dir / "wave16k" / f"{recording_id}.wav"
        wave, _ = librosa.load(path, sr=16000, mono=True)
        if len(wave) / 16000 + 1 / EVAL_FPS < duration:
            raise ValueError(f"Audio shorter than evaluated motion: {path}")
        onsets = librosa.onset.onset_detect(y=wave[:int(duration * 16000)], sr=16000, hop_length=512, units="time")
        return onsets[(onsets >= self.cli.beat_margin_seconds) & (onsets < duration - self.cli.beat_margin_seconds)]

    def rhythm_scores(self, joints, onsets, segments=()):
        # Detect extrema on complete continuous motion, never concatenated category spans.
        detector = self.helpers.alignment(0.3, 7, self.mean_velocity, upper_body=list(UPPER))
        beats = detector.load_pose(joints.reshape(len(joints), -1), 0, len(joints), EVAL_FPS, True)
        candidates = []
        for j in UPPER:
            times = np.asarray(beats[j]) / EVAL_FPS
            valid = np.ones(len(times), dtype=bool)
            for start, end, label in segments:
                if label == "excluded":
                    # Central differences plus order-7 extrema need clean neighbors.
                    valid &= (times < start - 8 / EVAL_FPS) | (times >= end + 8 / EVAL_FPS)
            candidates.append(times[valid])
        return onset_alignment_scores(onsets, candidates, sigma=0.3)


def onset_alignment_scores(onsets, motion_beats, sigma):
    values = []
    for beats in motion_beats:
        if len(beats):
            nearest = np.min(np.abs(onsets[:, None] - beats[None]), axis=1)
            values.append(np.exp(-nearest ** 2 / (2 * sigma ** 2)))
        else:
            values.append(np.zeros(len(onsets)))
    return np.mean(values, axis=0)


def frame_metric_stats(predicted, reference, predicted_pose, reference_pose, weights, thresholds):
    from scipy.spatial.transform import Rotation

    if (predicted.ndim != 3 or predicted.shape[1:] != (55, 3)
            or any(value.shape != predicted.shape for value in (reference, predicted_pose, reference_pose))):
        raise ValueError("Geometry/pose dimensions differ")
    if any(not np.isfinite(value).all() for value in (predicted, reference, predicted_pose, reference_pose)):
        raise ValueError("Nonfinite geometry or resampled pose")
    distances = np.linalg.norm(predicted - reference, axis=-1) * 1000
    p = Rotation.from_rotvec(predicted_pose.reshape(-1, 3))
    r = Rotation.from_rotvec(reference_pose.reshape(-1, 3))
    angles = np.rad2deg((p.inv() * r).magnitude()).reshape(len(reference), 55)
    result = []
    for category, weight in weights.items():
        for region, joints in REGIONS.items():
            count = float(weight.sum() * len(joints))
            values = {"mpjpe_mm": distances[:, joints], "rotation_error_deg": angles[:, joints]}
            values.update({f"pck_{threshold:g}mm": (distances[:, joints] < threshold) for threshold in thresholds})
            for metric, errors in values.items():
                result.append({"category": category, "region": region, "metric": metric,
                               "numerator": float(np.sum(errors * weight[:, None])), "denominator": count})
    return result


def aggregate_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["seed"], row["category"], row["region"], row["metric"]].append(row)
    result = []
    for (model, seed, category, region, metric), group in sorted(grouped.items()):
        denominator = sum(r["denominator"] for r in group)
        result.append({"model": model, "seed": seed, "category": category, "region": region, "metric": metric,
                       "value": sum(r["numerator"] for r in group) / denominator if denominator else None,
                       "recordings_with_observations": sum(r["denominator"] > 0 for r in group),
                       "denominator": denominator})
    return result


def seed_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["category"], row["region"], row["metric"]].append(row)
    result = []
    for (model, category, region, metric), group in sorted(grouped.items()):
        values = [r["value"] for r in group if r["value"] is not None]
        result.append({"model": model, "category": category, "region": region, "metric": metric,
                       "mean": float(np.mean(values)) if values else None,
                       "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       "seeds_with_observations": len(values)})
    return result


def category_statistics(rows, bootstrap, seed):
    """Absolute category scores, averaging seeds within each recording first."""
    cells = defaultdict(list)
    for row in rows:
        cells[row["recording_id"], row["model"], row["category"], row["region"], row["metric"]].append(row)
    grouped = defaultdict(list)
    for (fid, model, category, region, metric), group in cells.items():
        if len({r["seed"] for r in group}) != len(group):
            raise ValueError("Duplicate per-recording metric rows")
        counts = np.array([r["denominator"] for r in group])
        if not np.allclose(counts, counts[:1]):
            raise ValueError("Generation seeds evaluated different target observations")
        grouped[model, category, region, metric].append((np.mean([r["numerator"] for r in group]), counts[0]))
    rng = np.random.default_rng(seed)
    result = []
    for (model, category, region, metric), observations in sorted(grouped.items()):
        values = np.asarray(observations)
        values = values[values[:, 1] > 0]
        value = float(values[:, 0].sum() / values[:, 1].sum()) if len(values) else None
        low = high = None
        if bootstrap and len(values) >= 2:
            picks = rng.integers(len(values), size=(bootstrap, len(values)))
            distribution = values[picks, 0].sum(1) / values[picks, 1].sum(1)
            low, high = np.quantile(distribution, [0.025, 0.975]).tolist()
        result.append({"model": model, "category": category, "region": region, "metric": metric,
                       "value": value, "ci95_low": low, "ci95_high": high,
                       "recordings_with_observations": len(values)})
    return result


def paired_statistics(rows, bootstrap, seed):
    """Average generation seeds within recording; resample paired recordings."""
    cells = defaultdict(list)
    for row in rows:
        if row["model"] in MODELS:
            cells[row["recording_id"], row["model"], row["category"], row["region"], row["metric"]].append(row)
    reduced = {}
    for key, group in cells.items():
        if len({r["seed"] for r in group}) != len(group):
            raise ValueError("Duplicate per-recording metric rows")
        counts = np.array([r["denominator"] for r in group])
        if not np.allclose(counts, counts[:1]):
            raise ValueError("Generation seeds evaluated different target observations")
        reduced[key] = np.array([np.mean([r["numerator"] for r in group]), counts[0]])
    ids = sorted({row["recording_id"] for row in rows})
    scopes = sorted({(r["region"], r["metric"]) for r in rows if r["model"] in MODELS})
    rng = np.random.default_rng(seed)
    differences, interactions = [], []

    def ratio(values):
        return values[..., 0].sum(-1) / values[..., 1].sum(-1)

    for region, metric in scopes:
        arrays = {}
        for category in CATEGORIES:
            arrays[category] = {}
            for model in MODELS:
                arrays[category][model] = np.array([reduced[fid, model, category, region, metric] for fid in ids])
            if not np.array_equal(arrays[category]["baseline"][:, 1], arrays[category]["foremotion"][:, 1]):
                raise ValueError("Model pairs evaluated different observations")
            valid = np.flatnonzero(arrays[category]["baseline"][:, 1] > 0)
            baseline = float(ratio(arrays[category]["baseline"][valid])) if len(valid) else None
            foremotion = float(ratio(arrays[category]["foremotion"][valid])) if len(valid) else None
            delta = foremotion - baseline if len(valid) else None
            low = high = None
            if bootstrap and len(valid) >= 2:
                picks = rng.choice(valid, size=(bootstrap, len(valid)), replace=True)
                distribution = ratio(arrays[category]["foremotion"][picks]) - ratio(arrays[category]["baseline"][picks])
                low, high = np.quantile(distribution, [0.025, 0.975]).tolist()
            differences.append({"category": category, "region": region, "metric": metric, "paired_recordings": len(valid),
                                "baseline": baseline, "foremotion": foremotion, "foremotion_minus_baseline": delta,
                                "ci95_low": low, "ci95_high": high})
        both = np.flatnonzero((arrays["semantic"]["baseline"][:, 1] > 0) & (arrays["beat"]["baseline"][:, 1] > 0))
        # Positive improvements always favor ForeMotion; positive interaction favors semantic time.
        sign = 1 if metric in {"mpjpe_mm", "rotation_error_deg"} else -1

        def improvement(category, picks):
            return sign * (ratio(arrays[category]["baseline"][picks]) - ratio(arrays[category]["foremotion"][picks]))

        sem = float(improvement("semantic", both)) if len(both) else None
        beat = float(improvement("beat", both)) if len(both) else None
        low = high = None
        if bootstrap and len(both) >= 2:
            picks = rng.choice(both, size=(bootstrap, len(both)), replace=True)
            distribution = improvement("semantic", picks) - improvement("beat", picks)
            low, high = np.quantile(distribution, [0.025, 0.975]).tolist()
        interactions.append({"region": region, "metric": metric, "recordings_with_both_categories": len(both),
                             "semantic_improvement": sem, "beat_improvement": beat,
                             "semantic_minus_beat_improvement": sem - beat if len(both) else None,
                             "ci95_low": low, "ci95_high": high,
                             "primary": region == "upper_hands" and metric == "mpjpe_mm"})
    return differences, interactions


def frechet_features(a, b):
    """PSD eigendecomposition avoids nonsymmetric covariance-product sqrtm."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if min(len(a), len(b)) < 2 or a.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("FGD requires two or more finite feature rows per distribution")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite FGD features")
    ca, cb = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(ca)
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    middle = root @ cb @ root
    cross_trace = np.sqrt(np.maximum(np.linalg.eigvalsh((middle + middle.T) / 2), 0)).sum()
    value = np.sum((a.mean(0) - b.mean(0)) ** 2) + np.trace(ca) + np.trace(cb) - 2 * cross_trace
    if value < -1e-5 * (1 + np.trace(ca) + np.trace(cb)):
        raise ValueError(f"Numerically invalid negative FGD: {value}")
    return max(0.0, float(value))


def fgd_rows(features, seeds, minimum_windows, models=MODELS):
    result = []
    for seed in seeds:
        for category in CATEGORIES:
            target = features.get((seed, category, "ground_truth"), [])
            count = len(target)
            for model in models:
                predicted = features.get((seed, category, model), [])
                if len(predicted) != count:
                    raise ValueError("FGD models have different window counts")
                status, value = "insufficient_windows", None
                rows = np.concatenate(target) if count else np.empty((0, 240))
                if count >= minimum_windows and len(rows) >= 2:
                    value = frechet_features(np.concatenate(predicted), rows)
                    status = "computed_low_feature_count" if len(rows) <= rows.shape[1] else "computed"
                result.append({"model": model, "seed": seed, "category": category, "region": "full_body",
                               "metric": "fgd_continuous_windows", "value": value, "status": status,
                               "windows": count, "feature_rows": len(rows), "feature_dimensions": rows.shape[1]})
    return result


def write_report(output, differences, interactions, features, coverage, category_scores):
    fmt = lambda v: "unavailable" if v is None else f"{v:.6g}"
    lines = ["# Semantic-present versus beat-only motion evaluation", "",
             "Performance is measured against the recorded reference motion within each annotated category.", "",
             "| Category | Model | Region | Metric | Value | 95% CI |", "|---|---|---|---|---:|---:|"]
    for row in category_scores:
        if row["region"] != "upper_hands" and row["metric"] != "beat_alignment":
            continue
        lines.append(f"| {row['category']} | {row['model']} | {row['region']} | {row['metric']} | {fmt(row['value'])} | "
                     f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] |")
    if differences:
        lines += ["", "Optional paired model comparison. Differences are ForeMotion minus baseline.", "",
                  "| Category | Region | Metric | Baseline | ForeMotion | Difference [95% CI] |", "|---|---|---|---:|---:|---:|"]
    for row in differences:
        if row["region"] != "upper_hands" and row["metric"] != "beat_alignment":
            continue
        interval = f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"
        lines.append(f"| {row['category']} | {row['region']} | {row['metric']} | {fmt(row['baseline'])} | {fmt(row['foremotion'])} | {fmt(row['foremotion_minus_baseline'])} {interval} |")
    for row in interactions:
        if row["primary"]:
            lines += ["", f"Primary interaction (upper/hands MPJPE): {fmt(row['semantic_minus_beat_improvement'])} mm, "
                      f"95% CI [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]; "
                      f"{row['recordings_with_both_categories']} recordings contain both categories.",
                      "Positive interaction means a larger ForeMotion improvement during semantic-present time."]
    if features:
        lines += ["", "| Category | Model | Seed | Window FGD | Windows | Status |", "|---|---|---|---:|---:|---|"]
        for row in features:
            lines.append(f"| {row['category']} | {row['model']} | {row['seed']} | {fmt(row['value'])} | {row['windows']} | {row['status']} |")
    lines += ["", f"Evaluated recordings: {len(coverage)}. See coverage.csv for semantic/beat exposure and exclusions.",
              "Intervals resample whole recordings, conditional on the supplied model outputs, generation seeds and selected recordings. They are not multiplicity-adjusted.",
              "Generation seeds are averaged within recording before bootstrapping. They are not additional independent recordings.",
              "Window FGD uses continuous fixed-length windows and is not interchangeable with whole-utterance benchmark FGD. No FGD confidence interval is inferred from framewise errors.",
              "Category scores describe the evaluated system. Establishing a larger improvement from ForeMotion training requires a separate matched model comparison.",
              "Motion fidelity at reference semantic intervals does not by itself establish semantic appropriateness. A different gesture may also convey the same meaning."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cli):
    if cli.input_fps <= 0 or not math.isfinite(cli.input_fps) or cli.bootstrap < 0 or cli.chunk_size < 1:
        raise ValueError("Invalid frame rate, bootstrap count or chunk size")
    if cli.bootstrap_seed < 0 or cli.generation_seed < 0:
        raise ValueError("Seeds must be nonnegative")
    if cli.window_frames < 32 or cli.window_frames % 32 or cli.window_stride < 1 or not 0.5 < cli.window_purity <= 1:
        raise ValueError("FGD windows must be multiples of 32 frames, stride positive, purity > 0.5 and <= 1")
    if cli.min_fgd_windows < 2 or cli.feature_batch_size < 1 or cli.max_tail_frames < 0:
        raise ValueError("Invalid FGD minimum/batch size or tail tolerance")
    if cli.beat_margin_seconds < 0 or any(not math.isfinite(t) or t <= 0 for t in cli.pck_thresholds_mm):
        raise ValueError("Invalid beat margin or PCK thresholds")
    if len(cli.pck_thresholds_mm) != len(set(cli.pck_thresholds_mm)):
        raise ValueError("PCK thresholds must be distinct")
    if cli.pairs_manifest:
        if cli.baseline_dir or cli.foremotion_dir:
            raise ValueError("Use --motion-manifest instead of directory arguments")
    elif cli.foremotion_dir is None:
        raise ValueError("Supply --motion-dir (or --foremotion-dir), or --motion-manifest")
    if cli.max_recordings is not None and cli.max_recordings < 1:
        raise ValueError("max-recordings must be positive")
    recordings = annotations.load_recordings(cli)
    if cli.max_recordings is not None:
        recordings = recordings[:cli.max_recordings]
    pairs = resolve_pairs(cli, recordings)
    models = tuple(model for model in MODELS if model in next(iter(pairs.values())))
    comparison = "baseline" in models
    selected_ids = {recording.recording_id for recording in recordings}
    print(f"Selected {len(recordings)} '{cli.split}' recordings; only their matching motion exports and annotations will be evaluated.", flush=True)
    annotation_dir = cli.annotations_dir or cli.data_dir / "sem"
    candidates = defaultdict(list)
    for path in annotation_dir.rglob("*"):
        if path.stem in selected_ids and path.is_file() and path.suffix.lower() in {".txt", ".csv", ".tsv"}:
            candidates[path.stem].append(path)
    sources = {}
    for recording in recordings:
        paths = candidates[recording.recording_id]
        if len(paths) != 1:
            raise ValueError(f"Need exactly one annotation file for {recording.recording_id}; found {paths}")
        sources[recording.recording_id] = paths[0]
    output = cli.output_dir or ROOT / "reports" / f"gesture_category_evaluation_{datetime.now():%Y%m%d_%H%M%S}"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new/empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "selected_recordings.csv", [{**asdict(recording), "split": cli.split,
                                                    "annotation_path": str(sources[recording.recording_id].resolve())}
                                                   for recording in recordings])
    seeds = sorted({seed for seed, _ in pairs})
    metadata = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
                "recordings": len(recordings), "generation_seeds": seeds, "partial_run": cli.max_recordings is not None,
                "selected_recording_ids": [recording.recording_id for recording in recordings],
                "recording_selection": "HDF5 split and speaker filters; require matching prediction exports and exact-ID annotations; other annotation files and export folders are ignored",
                "analysis_mode": "paired_model_comparison" if comparison else "single_model_category_performance",
                "models": list(models),
                "categories": {"semantic": "deictic/iconic/metaphoric union, including simultaneous beat",
                               "beat": "beat annotations outside semantic intervals"},
                "annotation_policy": "Unknown/flagged spans excluded; no-gesture and unlabeled time excluded",
                "frame_rate_policy": "input-fps explicitly overrides NPZ metadata; evaluate at 30 fps with existing rotational SLERP",
                "geometry": "SMPL-X metres converted to millimetres; zero translation, pelvis-relative; no rotation/Procrustes alignment; flat_hand_mean=True",
                "primary_endpoint": ("upper_hands/mpjpe_mm semantic-minus-beat improvement interaction" if comparison
                                     else "upper_hands/mpjpe_mm performance within each gesture category"),
                "inference": "No generator or training is run. Saved system motion is scored against its gt.npz reference.",
                "statistics": "Average seeds within recording, then whole-recording bootstrap for category scores; paired comparisons only when a baseline is explicitly supplied",
                "fgd": "Existing BEAT encoder, full-body 6D rotations and reference jaw substitution, continuous category-qualified windows",
                "beat": "Existing normalized speed-minimum detector; per-audio-onset Gaussian match averaged over 13 upper-body joints, classified by onset time; continuous full-motion candidates",
                "script_sha256": sha256(Path(__file__)), "annotation_reader_sha256": sha256(Path(annotations.__file__)),
                "metric_helpers_sha256": sha256(ROOT / "scripts/trainers/utils/metrics.py"), "numpy_version": np.__version__}
    metadata_path = output / "metadata.json"
    rows, coverage, interval_rows, window_rows, inputs = [], [], [], [], []
    feature_store = defaultdict(list)
    started = time.monotonic()
    try:
        evaluator = MotionEvaluator(cli)
        for recording_number, recording in enumerate(recordings, 1):
            fid = recording.recording_id
            print(f"Recording {recording_number}/{len(recordings)}: {fid}", flush=True)
            source = sources[fid]
            annotation_rows = annotations.load_annotations(source, cli.unknown_labels)
            canonical_reference = None
            for seed_index, seed in enumerate(seeds):
                pair = pairs[seed, fid]
                motions, reference = read_pair(pair, recording, cli.input_fps, cli.max_tail_frames)
                if canonical_reference is None:
                    canonical_reference = reference
                elif (reference["poses"].shape != canonical_reference["poses"].shape
                      or not np.allclose(reference["poses"], canonical_reference["poses"], atol=1e-6, rtol=0)
                      or not np.allclose(reference["betas"], canonical_reference["betas"], atol=1e-6, rtol=0)):
                    raise ValueError(f"{fid}: generation seeds have different references")
                target_pose = evaluator.resample(reference["poses"], cli.input_fps)
                predicted_poses = {model: evaluator.resample(motions[model]["poses"], cli.input_fps) for model in models}
                frames = len(target_pose)
                end = min(recording.end_seconds, frames / EVAL_FPS)
                segments = category_segments(annotation_rows, 0, end)
                weights = frame_weights(segments, frames)
                if seed_index == 0:
                    stats = annotations.measure_intervals(annotation_rows, 0, end)
                    coverage.append({**asdict(recording), "evaluated_seconds": end, "evaluated_frames": frames,
                                     "input_frames": len(reference["poses"]), "source_annotation": str(source.resolve()),
                                     "annotation_sha256": sha256(source), "semantic_seconds": weights["semantic"].sum() / EVAL_FPS,
                                     "beat_seconds": weights["beat"].sum() / EVAL_FPS,
                                     "semantic_events": sum(c == "semantic" for _, _, c in segments),
                                     "beat_events": sum(c == "beat" for _, _, c in segments),
                                     **{key: stats[key] for key in ("excluded_seconds", "unlabeled_seconds", "explicit_no_gesture_seconds", "beat_semantic_overlap_seconds")}})
                    interval_rows.extend({"recording_id": fid, "start_seconds": lo, "end_seconds": hi, "category": c} for lo, hi, c in segments)
                target_joints = evaluator.joints(target_pose, reference["betas"])
                predicted_joints = {}
                for model in models:
                    predicted_joints[model] = evaluator.joints(predicted_poses[model], reference["betas"])
                    stats = frame_metric_stats(predicted_joints[model], target_joints, predicted_poses[model], target_pose, weights, cli.pck_thresholds_mm)
                    rows.extend({"recording_id": fid, "speaker": recording.speaker, "model": model, "seed": seed, **row} for row in stats)
                    inputs.append({"recording_id": fid, "seed": seed, "model": model,
                                   "prediction": str(pair[model]), "prediction_sha256": sha256(pair[model]),
                                   "reference": str(pair[model + "_reference"]), "reference_sha256": sha256(pair[model + "_reference"]),
                                   "declared_prediction_fps": motions[model]["declared_fps"], "used_input_fps": cli.input_fps})
                if "beat" in cli.metrics:
                    onsets = evaluator.audio_onsets(fid, end)
                    labels = point_categories(onsets, segments)
                    for model, joints in {**predicted_joints, "ground_truth": target_joints}.items():
                        scores = evaluator.rhythm_scores(joints, onsets, segments)
                        for category in CATEGORIES:
                            mask = labels == category
                            rows.append({"recording_id": fid, "speaker": recording.speaker, "model": model, "seed": seed,
                                         "category": category, "region": "upper_body", "metric": "beat_alignment",
                                         "numerator": float(scores[mask].sum()), "denominator": int(mask.sum())})
                if "fgd" in cli.metrics:
                    windows = select_windows(weights, cli.window_frames, cli.window_stride, cli.window_purity,
                                             [(lo, hi) for lo, hi, category in segments if category == "excluded"])
                    if seed_index == 0:
                        window_rows.extend({"recording_id": fid, "start_frame_30fps": lo, "end_frame_30fps": hi,
                                            "start_seconds": lo / EVAL_FPS, "end_seconds": hi / EVAL_FPS,
                                            "category": category, "category_fraction": purity} for lo, hi, category, purity in windows)
                        for category in CATEGORIES:
                            covered = np.zeros(frames, dtype=bool)
                            selected_windows = [(lo, hi) for lo, hi, c, _ in windows if c == category]
                            for lo, hi in selected_windows:
                                covered[lo:hi] = True
                            coverage[-1][f"fgd_{category}_windows"] = len(selected_windows)
                            coverage[-1][f"fgd_{category}_seconds_covered"] = float(weights[category][covered].sum() / EVAL_FPS)
                    for model, pose in {**predicted_poses, "ground_truth": target_pose}.items():
                        features = evaluator.features(pose, target_pose, windows)
                        for (_, _, category, _), feature in zip(windows, features, strict=True):
                            feature_store[seed, category, model].append(feature)
                write_csv(output / "per_recording_metrics.csv", rows)
                print(f"  seed {seed}: {frames} frames; semantic={weights['semantic'].sum() / EVAL_FPS:.2f}s, beat={weights['beat'].sum() / EVAL_FPS:.2f}s", flush=True)
        dataset_rows = aggregate_metrics(rows)
        category_scores = category_statistics(rows, cli.bootstrap, cli.bootstrap_seed)
        differences, interactions = paired_statistics(rows, cli.bootstrap, cli.bootstrap_seed) if comparison else ([], [])
        features = fgd_rows(feature_store, seeds, cli.min_fgd_windows, models) if "fgd" in cli.metrics else []
        write_csv(output / "category_metrics.csv", category_scores)
        write_csv(output / "metrics_by_seed.csv", dataset_rows)
        write_csv(output / "metrics_summary.csv", seed_summary(dataset_rows + features))
        write_csv(output / "paired_comparison.csv", differences)
        write_csv(output / "semantic_vs_beat_interaction.csv", interactions)
        write_csv(output / "fgd_by_seed.csv", features)
        write_csv(output / "coverage.csv", coverage)
        write_csv(output / "annotation_intervals.csv", interval_rows)
        if "fgd" in cli.metrics:
            write_csv(output / "fgd_windows.csv", window_rows,
                      ["recording_id", "start_frame_30fps", "end_frame_30fps", "start_seconds", "end_seconds", "category", "category_fraction"])
        write_csv(output / "input_manifest.csv", inputs)
        write_report(output, differences, interactions, features, coverage, category_scores)
        metadata.update(status="complete", elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        metadata.update(status="failed", error=str(exc), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Category evaluation complete: {output.resolve()}", flush=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-hdf5", type=Path, required=True)
    parser.add_argument("--motion-dir", "--foremotion-dir", dest="foremotion_dir", type=Path,
                        help="System test.py output root containing <recording>/pred.npz and gt.npz")
    parser.add_argument("--baseline-dir", type=Path, help="Optional extra model for a paired comparison; not needed for system evaluation")
    parser.add_argument("--motion-manifest", "--pairs-manifest", dest="pairs_manifest", type=Path,
                        help="CSV of saved system motions across generation seeds; alternative to --motion-dir")
    parser.add_argument("--generation-seed", type=int, default=2342, help="Provenance label for directory mode; does not regenerate motion")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/beat_english_v2.0.0"))
    parser.add_argument("--annotations-dir", type=Path)
    parser.add_argument("--dataset-ratio", choices=annotations.RATIOS, default="scott_beatx_fulllength")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--speakers", nargs="+")
    parser.add_argument("--motion-fps", type=float, default=25, help="Fallback frame rate for evaluation HDF5 selection")
    parser.add_argument("--input-fps", type=float, default=25, help="Actual NPZ array rate; overrides legacy test.py metadata incorrectly declaring 30")
    parser.add_argument("--max-tail-frames", type=int, default=1, help="Allowed missing input tail frames relative to full-sequence cache")
    parser.add_argument("--metrics", nargs="+", choices=("geometry", "fgd", "beat"), default=["geometry", "fgd", "beat"], help="Geometry is always computed; fgd and beat add feature/rhythm metrics")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smplx-path", type=Path, default=Path("assets_dep/smplx_2020"))
    parser.add_argument("--fgd-weights", type=Path, default=Path("datasets/beat_english_v2.0.0/weights/AESKConv_240_100.bin"))
    parser.add_argument("--mean-velocity", type=Path, default=Path("datasets/beat_english_v2.0.0/weights/mean_vel_smplxflame_30.npy"))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--pck-thresholds-mm", nargs="+", type=float, default=[50, 100, 150])
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--window-purity", type=float, default=0.8)
    parser.add_argument("--min-fgd-windows", type=int, default=20)
    parser.add_argument("--beat-margin-seconds", type=float, default=2)
    parser.add_argument("--unknown-labels", choices=("separate", "error"), default="separate")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2342)
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Error: {exc}")
