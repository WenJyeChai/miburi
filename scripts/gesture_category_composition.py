"""Audit BEATX gesture-category time by speaker, without loading a model.

See docs/gesture_category_composition.md for inputs, denominators and overlap
policies. Native BEAT sem/*.txt rows are: type start end duration score words.
Only type and the start/end timestamps determine category duration.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import wave

import numpy as np


CATEGORIES = ("beat", "deictic", "iconic", "metaphoric")
# Observed in native sem/ files. Exclude its span conservatively; the marker
# is not evidence for any of the four gesture categories or for no gesture.
EXCLUDED_LABELS = {"need_cut"}
POLICIES = ("semantic-over-beat", "fractional")
GOOD_SPEAKERS = {"wayne", "lawrence", "stewart", "solomon"}
RATIOS = (
    "full_beatx_fulllength", "scott_beatx_fulllength",
    "goodspk_beatx_fulllength", "full_beatx_eval",
)


@dataclass(frozen=True)
class Recording:
    recording_id: str
    speaker: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class Annotation:
    category: str
    start: float
    end: float


def text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def finite_float(value, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: nonfinite number {value!r}")
    return number


def category_from_label(label: str) -> str:
    """Decode BEAT type IDs, never the separate 0.0--1.0 relevance score."""
    label = label.strip().casefold()
    if label in EXCLUDED_LABELS:
        return label
    tokens = re.split(r"[^a-z0-9.]+", label)
    named = {c for c in CATEGORIES if c in tokens}
    if "metaphorical" in tokens:
        named.add("metaphoric")
    if len(named) > 1:
        raise ValueError(f"Ambiguous category {label!r}; use one interval per label")
    match = re.match(r"^(\d+)(?:\.0+)?(?:$|[_\s-])", label)
    coded = None
    if match:
        number = int(match.group(1))
        if not 0 <= number <= 10:
            raise ValueError(f"Unknown BEAT type ID in {label!r}")
        coded = (
            "none" if number == 0 else "beat" if number == 1 else
            "deictic" if number <= 4 else "iconic" if number <= 7 else "metaphoric"
        )
    if named:
        category = named.pop()
        if coded is not None and coded != category:
            raise ValueError(f"Type ID and category name disagree in {label!r}")
        return category
    if coded is not None:
        return coded
    if label in {"none", "no_gesture", "no gesture", "no-gesture", "silence", "rest"}:
        return "none"
    raise ValueError(f"Unknown gesture label {label!r}")


def _table(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        if not reader.fieldnames:
            raise ValueError(f"{path}: missing table header")
        return [
            {str(k).strip().casefold(): v.strip() if v else "" for k, v in row.items() if k is not None}
            for row in reader if any(v for v in row.values())
        ]


def _field(row: dict, *names: str, default=None):
    return next((row[n] for n in names if row.get(n) not in (None, "")), default)


def load_annotations(path: Path) -> list[Annotation]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        rows = [
            (i, _field(row, "category", "type", "types", "label"),
             _field(row, "start_seconds", "start"), _field(row, "end_seconds", "end"))
            for i, row in enumerate(_table(path), start=2)
        ]
    else:
        rows = []
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0].casefold() in {"type", "types", "category"}:
                continue
            if len(fields) < 3:
                raise ValueError(f"{path}:{i}: expected type, start and end")
            rows.append((i, fields[0], fields[1], fields[2]))
    annotations = []
    for line, label, start, end in rows:
        try:
            if label is None:
                raise ValueError("missing category/type column")
            category = category_from_label(label)
            start = finite_float(start, "start timestamp")
            end = finite_float(end, "end timestamp")
            if end < start:
                raise ValueError("end precedes start")
            annotations.append(Annotation(category, start, end))
        except ValueError as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    return annotations


def measure_intervals(annotations: list[Annotation], start: float, end: float,
                      policy: str = "semantic-over-beat") -> dict:
    """Integrate half-open intervals; duplicate labels never add extra time."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown overlap policy {policy!r}")
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Recording bounds must be finite with end > start")
    events = defaultdict(Counter)
    events[start]; events[end]
    clipped = outside = zero_length = 0
    for annotation in annotations:
        if annotation.category not in {*CATEGORIES, "none", *EXCLUDED_LABELS}:
            raise ValueError(f"Unknown category {annotation.category!r}")
        if not math.isfinite(annotation.start) or not math.isfinite(annotation.end):
            raise ValueError("Nonfinite annotation timestamp")
        if annotation.end < annotation.start:
            raise ValueError("Annotation end precedes start")
        if annotation.end == annotation.start:
            zero_length += 1
            continue
        lo, hi = max(start, annotation.start), min(end, annotation.end)
        if hi <= lo:
            outside += 1
            continue
        clipped += (lo != annotation.start or hi != annotation.end)
        events[lo][annotation.category] += 1
        events[hi][annotation.category] -= 1
    allocated = dict.fromkeys(CATEGORIES, 0.0)
    union = dict.fromkeys(CATEGORIES, 0.0)
    diagnostics = dict.fromkeys((
        "gesture_seconds", "explicit_no_gesture_seconds", "unlabeled_seconds", "excluded_seconds",
        "cross_category_overlap_seconds", "beat_semantic_overlap_seconds",
        "semantic_category_overlap_seconds", "none_gesture_conflict_seconds",
    ), 0.0)
    active = Counter()
    previous = start
    for timestamp in sorted(events):
        duration = timestamp - previous
        categories = {c for c in CATEGORIES if active[c] > 0}
        semantic = categories - {"beat"}
        if any(active[label] > 0 for label in EXCLUDED_LABELS):
            # Mask rather than just dropping the row: a broad overlapping
            # beat interval must not reclassify a span flagged need_cut.
            diagnostics["excluded_seconds"] += duration
        elif categories:
            diagnostics["gesture_seconds"] += duration
            for category in categories:
                union[category] += duration
            recipients = semantic if policy == "semantic-over-beat" and semantic else categories
            for category in recipients:
                allocated[category] += duration / len(recipients)
            if len(categories) > 1:
                diagnostics["cross_category_overlap_seconds"] += duration
            if "beat" in categories and semantic:
                diagnostics["beat_semantic_overlap_seconds"] += duration
            if len(semantic) > 1:
                diagnostics["semantic_category_overlap_seconds"] += duration
            if active["none"] > 0:
                diagnostics["none_gesture_conflict_seconds"] += duration
        elif active["none"] > 0:
            diagnostics["explicit_no_gesture_seconds"] += duration
        else:
            diagnostics["unlabeled_seconds"] += duration
        active.update(events[timestamp])
        previous = timestamp
    if not math.isclose(sum(allocated.values()), diagnostics["gesture_seconds"], abs_tol=1e-7):
        raise RuntimeError("Allocated category time does not equal the gesture-time union")
    accounted = sum(diagnostics[key] for key in (
        "gesture_seconds", "explicit_no_gesture_seconds", "unlabeled_seconds", "excluded_seconds",
    ))
    if not math.isclose(accounted, end - start, abs_tol=1e-7):
        raise RuntimeError("Coverage and excluded intervals do not conserve recording time")
    return {
        **diagnostics, "allocated": allocated, "category_union": union,
        "annotation_rows": len(annotations), "clipped_rows": clipped,
        "outside_rows": outside, "zero_length_rows": zero_length,
        "need_cut_rows": sum(a.category == "need_cut" for a in annotations),
    }


def _speaker(recording_id: str) -> str:
    fields = recording_id.split("_")
    if len(fields) < 3 or not fields[0].isdigit():
        raise ValueError(f"Cannot infer speaker from {recording_id!r}; provide a speaker column")
    return fields[1].casefold()


def _select(speaker: str, ratio: str, speakers: list[str] | None) -> bool:
    if speakers and speaker not in {s.casefold() for s in speakers}:
        return False
    if ratio == "full_beatx_fulllength":
        return speaker not in {"carla", "itoi"}
    if ratio == "scott_beatx_fulllength":
        return speaker == "scott"
    if ratio == "goodspk_beatx_fulllength":
        return speaker in GOOD_SPEAKERS
    if ratio == "full_beatx_eval":
        return True
    raise ValueError(f"Unsupported dataset ratio {ratio!r}")


def _validate_recording(recording: Recording) -> Recording:
    if re.search(r"_C\d+$", recording.recording_id):
        raise ValueError(f"{recording.recording_id}: chunked input; supply full-utterance evaluation recordings")
    if recording.start_seconds < 0 or recording.end_seconds <= recording.start_seconds:
        raise ValueError(f"{recording.recording_id}: invalid evaluation time bounds")
    return recording


def load_recordings(args) -> list[Recording]:
    recordings = {}

    def add(recording):
        _validate_recording(recording)
        previous = recordings.get(recording.recording_id)
        if previous is not None and previous != recording:
            raise ValueError(f"Conflicting duplicate recording {recording.recording_id}")
        recordings[recording.recording_id] = recording

    if args.eval_hdf5:
        import h5py

        with h5py.File(args.eval_hdf5, "r") as database:
            fps = finite_float(database.attrs.get("motion_fps", args.motion_fps), "motion fps")
            if fps <= 0:
                raise ValueError("Motion fps must be positive")
            ids = [text(v) for v in database["chunk_ids"][:]] if "chunk_ids" in database else [
                k for k, v in database.items() if isinstance(v, h5py.Group)
            ]
            splits = [text(v) for v in database["chunk_splits"][:]] if "chunk_splits" in database else None
            if splits is not None and len(splits) != len(ids):
                raise ValueError("HDF5 chunk_ids/chunk_splits lengths differ")
            for i, key in enumerate(ids):
                group = database[key]
                split = splits[i] if splits is not None else text(group.attrs.get("split", ""))
                if split == "dev":
                    split = "val"
                if split != args.split:
                    continue
                recording_id = text(group.attrs.get("file_id", re.sub(r"_C\d+$", "", key)))
                speaker = _speaker(recording_id)
                if not _select(speaker, args.dataset_ratio, args.speakers):
                    continue
                if re.search(r"_C\d+$", key):
                    raise ValueError(f"{key}: training chunks are not evaluation utterances; use a --full_sequence cache")
                start = finite_float(group.attrs.get("chunk_startsec", 0.0), key)
                if start != 0:
                    raise ValueError(f"{key}: nonzero start; expected a full-utterance cache")
                if "motion" in group:
                    duration = group["motion"].shape[0] / fps
                    declared = finite_float(group.attrs.get("chunk_endsec", duration), key)
                    if abs(declared - duration) > 1.0 / fps + 1e-6:
                        raise ValueError(f"{key}: motion length disagrees with chunk_endsec")
                    end = min(duration, declared)
                else:
                    end = finite_float(group.attrs.get("chunk_endsec"), key)
                add(Recording(recording_id, speaker, start, end))
    else:
        for row in _table(args.manifest):
            split = _field(row, "split", "type")
            split = "val" if split == "dev" else split
            if split is not None and split != args.split:
                continue
            recording_id = _field(row, "recording_id", "file_id", "id")
            if not recording_id:
                raise ValueError("Manifest needs recording_id, file_id or id")
            speaker = _field(row, "speaker", default=None) or _speaker(recording_id)
            speaker = speaker.casefold()
            if not _select(speaker, args.dataset_ratio, args.speakers):
                continue
            start = finite_float(_field(row, "start_seconds", default=0), recording_id)
            end = _field(row, "end_seconds")
            if end is not None:
                end = finite_float(end, recording_id)
            elif _field(row, "duration_seconds") is not None:
                end = start + finite_float(row["duration_seconds"], recording_id)
            else:
                audio = args.data_dir / "wave16k" / f"{recording_id}.wav"
                if not audio.is_file():
                    raise ValueError(f"{recording_id}: provide duration_seconds/end_seconds or {audio}")
                with wave.open(str(audio), "rb") as handle:
                    end = handle.getnframes() / handle.getframerate()
            add(Recording(recording_id, speaker, start, end))
    if not recordings:
        raise ValueError("No evaluation recordings match the split and speaker filters")
    return sorted(recordings.values(), key=lambda r: (r.speaker, r.recording_id))


def _safe_ratio(numerator: float, denominator: float):
    return numerator / denominator if denominator > 0 else None


def bootstrap_composition(values: np.ndarray, samples: int, rng) -> list[tuple]:
    """Resample complete recordings; compute duration-pooled ratios each time."""
    if samples == 0 or len(values) < 2 or values.sum() <= 0:
        return [(None, None)] * len(CATEGORIES)
    totals = values[rng.integers(0, len(values), size=(samples, len(values)))].sum(axis=1)
    totals = totals[totals.sum(axis=1) > 0]
    if not len(totals):
        return [(None, None)] * len(CATEGORIES)
    bounds = np.quantile(totals / totals.sum(axis=1, keepdims=True), [0.025, 0.975], axis=0)
    return list(zip(bounds[0].tolist(), bounds[1].tolist()))


def summarize(rows: list[dict], bootstrap: int, seed: int) -> tuple[list, list]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["speaker"]].append(row)
    speaker_rows, category_rows = [], []
    rng = np.random.default_rng(seed)
    for speaker, members in sorted(groups.items()):
        valid = [r for r in members if r["status"] == "annotated"]
        excluded = sum(r["excluded_seconds"] for r in valid)
        exposure = sum(r["evaluation_seconds"] for r in valid) - excluded
        gesture = sum(r["gesture_seconds"] for r in valid)
        summary = {
            "speaker": speaker, "selected_recordings": len(members),
            "annotated_recordings": len(valid), "missing_recordings": len(members) - len(valid),
            "selected_seconds": sum(r["evaluation_seconds"] for r in members),
            "excluded_seconds": excluded,
            "need_cut_rows": sum(r["need_cut_rows"] for r in valid),
            "missing_annotation_seconds": sum(r["evaluation_seconds"] for r in members if r["status"] == "missing"),
            "evaluated_seconds": exposure, "gesture_seconds": gesture,
            "gesture_fraction_of_evaluated_time": _safe_ratio(gesture, exposure),
        }
        for field in (
            "explicit_no_gesture_seconds", "unlabeled_seconds", "cross_category_overlap_seconds",
            "beat_semantic_overlap_seconds", "semantic_category_overlap_seconds", "none_gesture_conflict_seconds",
        ):
            summary[field] = sum(r[field] for r in valid)
        summary["annotation_coverage_fraction"] = _safe_ratio(exposure - summary["unlabeled_seconds"], exposure)
        analyzable = [r for r in valid if r["evaluation_seconds"] > r["excluded_seconds"]]
        values = np.array([[r[f"{c}_seconds"] for c in CATEGORIES] for r in analyzable], dtype=float).reshape(-1, 4)
        intervals = bootstrap_composition(values, bootstrap, rng)
        for i, category in enumerate(CATEGORIES):
            seconds = sum(r[f"{category}_seconds"] for r in valid)
            summary[f"{category}_seconds"] = seconds
            summary[f"{category}_fraction"] = _safe_ratio(seconds, gesture)
            category_rows.append({
                "speaker": speaker, "category": category, "allocated_seconds": seconds,
                "category_union_seconds": sum(r[f"{category}_union_seconds"] for r in valid),
                "fraction_of_gesture_time": _safe_ratio(seconds, gesture),
                "fraction_of_evaluated_time": _safe_ratio(seconds, exposure),
                "ci95_low": intervals[i][0], "ci95_high": intervals[i][1],
                "recordings_with_category": sum(r[f"{category}_seconds"] > 0 for r in valid),
            })
        speaker_rows.append(summary)
    return speaker_rows, category_rows


def cohort_summary(rows: list[dict], bootstrap: int, seed: int) -> list[dict]:
    result = []
    rng = np.random.default_rng(seed)
    for name, subset in (
        ("all_selected", rows), ("scott", [r for r in rows if r["speaker"] == "scott"]),
        ("other_speakers", [r for r in rows if r["speaker"] != "scott"]),
    ):
        valid = [r for r in subset if r["status"] == "annotated"]
        if not valid:
            continue
        analyzable = [r for r in valid if r["evaluation_seconds"] > r["excluded_seconds"]]
        values = np.array([[r[f"{c}_seconds"] for c in CATEGORIES] for r in analyzable]).reshape(-1, 4)
        total = values.sum()
        bounds = bootstrap_composition(values, bootstrap, rng)
        for i, category in enumerate(CATEGORIES):
            result.append({
                "cohort": name, "category": category, "recordings": len(valid),
                "analyzable_recordings": len(analyzable),
                "speakers": len({r["speaker"] for r in valid}),
                "allocated_seconds": float(values[:, i].sum()),
                "fraction_of_gesture_time": _safe_ratio(float(values[:, i].sum()), total),
                "ci95_low": bounds[i][0], "ci95_high": bounds[i][1],
            })
    return result


def permutation_test(rows: list[dict], permutations: int, seed: int) -> dict:
    """Optional exploratory omnibus test with recordings as sampling units."""
    result = {"permutations": permutations, "unit": "recording", "weighting": "equal recording"}
    if permutations == 0:
        return {**result, "status": "disabled"}
    valid = [r for r in rows if r["status"] == "annotated" and r["gesture_seconds"] > 0]
    names = sorted({r["speaker"] for r in valid})
    counts = Counter(r["speaker"] for r in valid)
    if len(names) < 2 or min(counts.values(), default=0) < 2:
        return {**result, "status": "insufficient_recordings", "reason": "Need at least two recordings in each of two speakers"}
    values = np.array([[r[f"{c}_seconds"] / r["gesture_seconds"] for c in CATEGORIES] for r in valid])
    labels = np.array([names.index(r["speaker"]) for r in valid])
    sizes = np.bincount(labels)
    mean = values.mean(axis=0)
    total = float(np.square(values - mean).sum())

    def effect(labels):
        sums = np.zeros((len(names), 4))
        np.add.at(sums, labels, values)
        return float((sizes[:, None] * np.square(sums / sizes[:, None] - mean)).sum() / total) if total > 0 else 0.0

    observed = effect(labels)
    rng = np.random.default_rng(seed)
    exceed = sum(effect(rng.permutation(labels)) >= observed - 1e-12 for _ in range(permutations))
    return {
        **result, "status": "ok", "recordings": len(valid), "speakers": len(names),
        "between_speaker_variance_fraction": observed, "p_value": (exceed + 1) / (permutations + 1),
        "assumption": "Recordings are exchangeable across speaker labels under the null; repeated prompts and recording conditions can violate this.",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output: Path, speakers: list[dict], cohorts: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = ("#648FFF", "#DC267F", "#FFB000", "#785EF0")
    labels = [f"{s['speaker']} (n={s['annotated_recordings']})" for s in speakers]
    y = np.arange(len(speakers))
    height = max(3.6, 0.36 * len(speakers) + 1.7)

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, height), layout="constrained")
    left = np.zeros(len(speakers))
    for category, color in zip(CATEGORIES, colors):
        values = np.array([s[f"{category}_fraction"] or 0 for s in speakers])
        ax.barh(y, values, left=left, color=color, label=category.capitalize())
        left += values
    for i, speaker in enumerate(speakers):
        if speaker["gesture_seconds"] == 0:
            ax.text(0.01, i, "No labeled gesture time", va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 1), xlabel="Share of annotated gesture time", title="Gesture composition by speaker")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
    save(fig, "speaker_composition")

    fig, ax = plt.subplots(figsize=(10, height), layout="constrained")
    left = np.zeros(len(speakers))
    for field, label, color in (
        ("gesture_seconds", "Categorized gesture", "#648FFF"),
        ("explicit_no_gesture_seconds", "Explicit no gesture", "#A2C6A2"),
        ("unlabeled_seconds", "Unlabeled time", "#D4D4D4"),
        ("excluded_seconds", "Excluded: need_cut", "#FFB000"),
        ("missing_annotation_seconds", "Missing annotation file", "#DC267F"),
    ):
        values = np.array([
            s[field] / s["selected_seconds"]
            for s in speakers
        ])
        ax.barh(y, values, left=left, color=color, label=label)
        left += values
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 1), xlabel="Share of selected evaluation time", title="Annotation coverage by speaker")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
    save(fig, "annotation_coverage")

    group_names = [g for g in ("scott", "other_speakers") if any(r["cohort"] == g for r in cohorts)]
    if len(group_names) == 2:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        x = np.arange(4)
        for i, (name, label) in enumerate(zip(group_names, ("Scott", "Other speakers pooled"))):
            rows = [next(r for r in cohorts if r["cohort"] == name and r["category"] == c) for c in CATEGORIES]
            values = np.array([r["fraction_of_gesture_time"] if r["fraction_of_gesture_time"] is not None else np.nan for r in rows])
            ax.bar(x + (i - 0.5) * 0.36, values, 0.36, label=label, color=("#648FFF", "#FFB000")[i])
        ax.set(xticks=x, xticklabels=[c.capitalize() for c in CATEGORIES], ylim=(0, 1), ylabel="Share of annotated gesture time", title="Scott versus the other selected speakers")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False)
        save(fig, "scott_vs_other_speakers")


def run(args) -> Path:
    if args.bootstrap < 0 or args.permutations < 0:
        raise ValueError("Bootstrap and permutation counts must be nonnegative")
    annotations_dir = args.annotations_dir or args.data_dir / "sem"
    if not annotations_dir.is_dir():
        raise ValueError(
            f"Annotation folder not found: {annotations_dir}. Supply --annotations-dir pointing to BEAT sem/; "
            "the project's standard dataset downloader does not fetch sem/."
        )
    recordings = load_recordings(args)
    candidates = defaultdict(list)
    for path in annotations_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".tsv"}:
            candidates[path.stem].append(path)
    rows = []
    for recording in recordings:
        paths = candidates[recording.recording_id]
        if len(paths) > 1:
            raise ValueError(f"Ambiguous annotation files for {recording.recording_id}: {paths}")
        row = {**asdict(recording), "evaluation_seconds": recording.end_seconds - recording.start_seconds}
        if not paths:
            if args.missing_annotations == "error":
                raise ValueError(f"Missing annotation file for {recording.recording_id}; use --missing-annotations skip to report an incomplete audit")
            row.update(status="missing", annotation_path="", annotation_sha256="")
            stats = None
        else:
            path = paths[0]
            stats = measure_intervals(load_annotations(path), recording.start_seconds, recording.end_seconds, args.overlap_policy)
            row.update(status="annotated", annotation_path=str(path.resolve()), annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for field in (
            "gesture_seconds", "explicit_no_gesture_seconds", "unlabeled_seconds", "excluded_seconds",
            "cross_category_overlap_seconds", "beat_semantic_overlap_seconds", "semantic_category_overlap_seconds",
            "none_gesture_conflict_seconds", "annotation_rows", "clipped_rows", "outside_rows", "zero_length_rows", "need_cut_rows",
        ):
            row[field] = stats[field] if stats is not None else None
        for category in CATEGORIES:
            row[f"{category}_seconds"] = stats["allocated"][category] if stats is not None else None
            row[f"{category}_union_seconds"] = stats["category_union"][category] if stats is not None else None
        rows.append(row)
    if not any(r["status"] == "annotated" for r in rows):
        raise ValueError("None of the selected recordings has an annotation file")
    speakers, categories = summarize(rows, args.bootstrap, args.seed)
    cohorts = cohort_summary(rows, args.bootstrap, args.seed)
    test = permutation_test(rows, args.permutations, args.seed)
    output = args.output_dir or Path("reports") / f"gesture_composition_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "recording_summary.csv", rows)
    write_csv(output / "speaker_summary.csv", speakers)
    write_csv(output / "speaker_categories.csv", categories)
    write_csv(output / "cohort_categories.csv", cohorts)
    write_csv(output / "selected_recordings.csv", [asdict(r) for r in recordings])
    source = args.eval_hdf5 or args.manifest
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "selection_source": {"path": str(source.resolve()), "bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns},
        "selected_recordings_sha256": hashlib.sha256((output / "selected_recordings.csv").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "overlap_policy": args.overlap_policy,
        "composition_denominator": "Union of time labeled as at least one of the four gesture categories, after need_cut exclusions; excludes none and unlabeled time",
        "evaluation_time_denominator": "Duration with annotation files available, minus the union of need_cut spans",
        "excluded_interval_labels": sorted(EXCLUDED_LABELS),
        "excluded_interval_policy": "Conservative analysis choice: mask flagged spans before category allocation, including overlapping gesture labels; no formal upstream definition of need_cut was verified",
        "excluded_seconds": sum(s["excluded_seconds"] for s in speakers),
        "need_cut_rows": sum(s["need_cut_rows"] for s in speakers),
        "weighting": "Duration pooled within each speaker/cohort; cohorts may include the same recordings",
        "intervals": "Half-open [start,end), clipped to evaluation bounds; same-category duplicates count once",
        "confidence_intervals": "95% percentile bootstrap, whole recordings resampled within each speaker (or within pooled cohort); unavailable with fewer than two recordings",
        "selected_recordings": len(recordings), "annotated_recordings": sum(r["status"] == "annotated" for r in rows),
        "speakers": len(speakers), "exploratory_permutation_test": test,
        "interpretation": "Describes the selected recordings. Speaker, text content and recording conditions may be confounded; this does not establish the cause of ForeMotion performance differences.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.no_plots:
        plot_results(output, speakers, cohorts)
    lines = [
        f"Selected {len(recordings)} recordings across {len(speakers)} speakers.",
        f"Annotation files found: {metadata['annotated_recordings']}/{len(recordings)}.",
        f"Overlap policy: {args.overlap_policy}. Fractions are within annotated gesture time.",
        f"Excluded need_cut spans: {metadata['excluded_seconds']:.3f} seconds ({metadata['need_cut_rows']} source rows).",
        "Allocated seconds sum to the gesture-time union. Category-union columns can overlap; do not stack them.",
        "Unlabeled time is not assumed to be beat or no gesture. Empty files yield zero labeled time.",
        "Bootstrap units are complete recordings, not frames; cohort intervals describe the selected speaker set.",
        f"Exploratory permutation test: {json.dumps(test)}",
        metadata["interpretation"],
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines[:4]:
        print(line)
    print(f"Results: {output.resolve()}")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--eval-hdf5", type=Path, help="Existing full-sequence BEATX evaluation cache (preferred)")
    selection.add_argument("--manifest", type=Path, help="CSV/TSV recording list: id, optional split/type, speaker, duration_seconds")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/beat_english_v2.0.0"))
    parser.add_argument("--annotations-dir", type=Path, help="Default: <data-dir>/sem; one TXT/CSV/TSV per recording")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--dataset-ratio", choices=RATIOS, default="full_beatx_fulllength")
    parser.add_argument("--speakers", nargs="+", help="Optional speaker names, intersected with dataset-ratio filter")
    parser.add_argument("--motion-fps", type=float, default=25, help="Fallback if HDF5 has no motion_fps attribute")
    parser.add_argument("--overlap-policy", choices=POLICIES, default="semantic-over-beat")
    parser.add_argument("--missing-annotations", choices=("error", "skip"), default="error")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Recording bootstrap replicates; 0 disables intervals")
    parser.add_argument("--permutations", type=int, default=0, help="Optional exploratory recording-level omnibus test")
    parser.add_argument("--seed", type=int, default=2342)
    parser.add_argument("--output-dir", type=Path, help="New/empty directory; default: timestamped directory under reports/")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError, wave.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
