"""Scientific-control checks for saved-motion category evaluation; no downloads.

Run: python -B -m scripts.smoke_test_gesture_category_evaluation
"""
from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from scripts import gesture_category_evaluation as evaluation
from scripts.gesture_category_composition import Annotation, Recording


def write_motion(path, frames=60, value=0, fps=30, betas=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, poses=np.full((frames, 55, 3), value, dtype=np.float32),
             betas=np.zeros(300, np.float32) if betas is None else betas,
             trans=np.zeros((frames, 3), np.float32), expressions=np.zeros((frames, 100), np.float32),
             mocap_frame_rate=fps)


class CategoryEvaluationChecks(unittest.TestCase):
    def test_exact_masks_semantic_union_beats_duplicates_and_unknown_exclusions(self):
        rows = [Annotation("beat", 0, 4)] * 2 + [Annotation("iconic", 1, 2),
                Annotation("deictic", 1.5, 3), Annotation("need_cut", 2.5, 2.75),
                Annotation("self_touch", 3.5, 3.75)]
        segments = evaluation.category_segments(rows, 0, 4)
        weights = evaluation.frame_weights(segments, 4, fps=1)
        np.testing.assert_allclose(weights["semantic"], [0, 1, 0.75, 0])
        np.testing.assert_allclose(weights["beat"], [1, 0, 0, 0.75])
        self.assertLessEqual(np.max(weights["semantic"] + weights["beat"]), 1)

    def test_touching_endpoints_no_gesture_unlabeled_and_subframe_events(self):
        rows = [Annotation("beat", -1, 0.25), Annotation("iconic", 0.25, 0.5),
                Annotation("none", 0.5, 0.75)]
        segments = evaluation.category_segments(rows, 0, 1)
        weights = evaluation.frame_weights(segments, 1, fps=1)
        self.assertEqual(weights["semantic"].sum(), 0.25)
        self.assertEqual(weights["beat"].sum(), 0.25)
        labels = evaluation.point_categories(np.array([0, 0.25, 0.5, 0.75, 1]), segments)
        self.assertEqual(labels.tolist(), ["beat", "semantic", "", "", ""])

    def test_fgd_windows_keep_time_contiguous_and_share_selection(self):
        weights = {"semantic": np.r_[np.ones(32), np.zeros(64), np.ones(16)],
                   "beat": np.r_[np.zeros(32), np.ones(64), np.zeros(16)]}
        windows = evaluation.select_windows(weights, frames=32, stride=32, purity=0.8)
        self.assertEqual(windows, [(0, 32, "semantic", 1.0), (32, 64, "beat", 1.0), (64, 96, "beat", 1.0)])
        mixed = {"semantic": np.tile([1, 0], 32), "beat": np.tile([0, 1], 32)}
        self.assertEqual(evaluation.select_windows(mixed, purity=0.8), [])
        excluded = evaluation.select_windows(weights, frames=32, stride=32, purity=0.8,
                                               excluded_intervals=[(0.5, 0.51)])
        self.assertEqual(excluded, windows[1:])

    def test_geometry_metrics_millimetres_thresholds_and_exact_duration_weighting(self):
        target = np.zeros((2, 55, 3))
        predicted = target.copy()
        predicted[0, :, 0] = 0.06
        predicted[1, :, 0] = 0.12
        pose = np.zeros_like(target)
        rows = evaluation.frame_metric_stats(predicted, target, pose, pose,
                                             {"semantic": np.array([0.5, 1.0]), "beat": np.array([0.5, 0.0])}, [50, 100])
        values = {(r["category"], r["region"], r["metric"]): r for r in rows}
        mpjpe = values["semantic", "upper_hands", "mpjpe_mm"]
        self.assertAlmostEqual(mpjpe["numerator"] / mpjpe["denominator"], 100)
        pck = values["semantic", "upper_hands", "pck_100mm"]
        self.assertAlmostEqual(pck["numerator"] / pck["denominator"], 1 / 3)
        self.assertEqual(values["beat", "upper_hands", "pck_50mm"]["numerator"], 0)
        self.assertEqual(values["semantic", "hands", "rotation_error_deg"]["numerator"], 0)

    def test_pair_reader_overrides_legacy_fps_and_rejects_different_references(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pair = {}
            for model in evaluation.MODELS:
                pair[model], pair[model + "_reference"] = root / model / "pred.npz", root / model / "gt.npz"
                write_motion(pair[model], frames=50, value=0.1, fps=30)
                write_motion(pair[model + "_reference"], frames=50, fps=30)
            recording = Recording("2_scott_0_1_1", "scott", 0, 2)
            _, target = evaluation.read_pair(pair, recording, input_fps=25, max_tail_frames=1)
            self.assertEqual(len(target["poses"]), 50)
            self.assertEqual(target["declared_fps"], 30)
            with self.assertRaisesRegex(ValueError, "duration disagrees"):
                evaluation.read_pair(pair, recording, input_fps=30, max_tail_frames=1)
            write_motion(pair["foremotion_reference"], frames=50, value=0.2)
            with self.assertRaisesRegex(ValueError, "different ground truth"):
                evaluation.read_pair(pair, recording, 25, 1)

    def test_pair_reader_rejects_missing_frames_nonfinite_inputs_and_shape_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pair = {}
            for model in evaluation.MODELS:
                pair[model], pair[model + "_reference"] = root / model / "pred.npz", root / model / "gt.npz"
                write_motion(pair[model], frames=50)
                write_motion(pair[model + "_reference"], frames=50)
            recording = Recording("2_scott_0_1_1", "scott", 0, 2)
            write_motion(pair["foremotion"], frames=49)
            with self.assertRaisesRegex(ValueError, "frame counts differ"):
                evaluation.read_pair(pair, recording, 25, 1)
            write_motion(pair["foremotion"], frames=50, value=float("nan"))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                evaluation.read_pair(pair, recording, 25, 1)

    def test_seed_manifest_requires_complete_paired_recordings_and_unique_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("baseline", "foremotion"):
                write_motion(root / name / "pred.npz")
                write_motion(root / name / "gt.npz")
            manifest = root / "pairs.csv"
            manifest.write_text("recording_id,seed,baseline_path,foremotion_path\n2_scott_0_1_1,1,baseline/pred.npz,foremotion/pred.npz\n", encoding="utf-8")
            cli = evaluation.parse_args(["--eval-hdf5", "unused", "--pairs-manifest", str(manifest)])
            selected = [Recording("2_scott_0_1_1", "scott", 0, 2)]
            pairs = evaluation.resolve_pairs(cli, selected)
            self.assertEqual(len(pairs), 1)
            self.assertTrue(pairs["1", selected[0].recording_id]["baseline"].is_absolute())
            with self.assertRaisesRegex(ValueError, "missing selected recordings"):
                evaluation.resolve_pairs(cli, selected + [Recording("2_scott_0_2_2", "scott", 0, 2)])
            with manifest.open("a", encoding="utf-8") as stream:
                stream.write("2_scott_0_1_1,1,baseline/pred.npz,foremotion/pred.npz\n")
            with self.assertRaisesRegex(ValueError, "Duplicate model pair"):
                evaluation.resolve_pairs(cli, selected)

    def test_onset_alignment_uses_real_time_and_absent_motion_beats_score_zero(self):
        onsets = np.array([1.0, 3.0])
        exact = evaluation.onset_alignment_scores(onsets, [onsets.copy()] * 13, 0.3)
        shifted = evaluation.onset_alignment_scores(onsets, [onsets + 0.3] * 13, 0.3)
        stationary = evaluation.onset_alignment_scores(onsets, [np.array([])] * 13, 0.3)
        np.testing.assert_allclose(exact, 1)
        np.testing.assert_allclose(shifted, np.exp(-0.5))
        np.testing.assert_allclose(stationary, 0)
        self.assertEqual(len(evaluation.onset_alignment_scores(np.array([]), [onsets] * 13, 0.3)), 0)

    def test_rhythm_adapter_excludes_flagged_neighbors_and_preserves_continuous_input(self):
        seen = []

        class Detector:
            def __init__(self, *args, **kwargs):
                pass

            def load_pose(self, joints, start, end, fps, without_file):
                seen.append((len(joints), start, end, fps, without_file))
                return [np.array([15, 45])] * 55

        evaluator = object.__new__(evaluation.MotionEvaluator)
        evaluator.helpers = types.SimpleNamespace(alignment=Detector)
        evaluator.mean_velocity = np.ones(55)
        onsets = np.array([0.8, 1.5])
        scores = evaluator.rhythm_scores(np.zeros((60, 55, 3)), onsets, [(0.45, 0.55, "excluded")])
        np.testing.assert_allclose(scores, [np.exp(-0.7 ** 2 / (2 * 0.3 ** 2)), 1])
        self.assertEqual(seen, [(60, 0, 60, 30, True)])

    def test_frechet_distance_has_known_shift_and_singular_identical_controls(self):
        values = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        self.assertAlmostEqual(evaluation.frechet_features(values, values), 0, places=6)
        self.assertAlmostEqual(evaluation.frechet_features(values, values + 2), 8, places=6)
        with self.assertRaises(ValueError):
            evaluation.frechet_features(values[:1], values[:1])

    def test_fgd_missing_category_or_small_sample_is_unavailable(self):
        features = {("1", "beat", model): [np.zeros((2, 3))] * 2 for model in (*evaluation.MODELS, "ground_truth")}
        rows = evaluation.fgd_rows(features, ["1"], minimum_windows=2)
        self.assertTrue(all(r["value"] is None for r in rows if r["category"] == "semantic"))
        self.assertTrue(all(r["value"] == 0 for r in rows if r["category"] == "beat"))
        rows = evaluation.fgd_rows(features, ["1"], minimum_windows=3)
        self.assertTrue(all(r["status"] == "insufficient_windows" for r in rows))

    def test_paired_interaction_and_generation_seed_averaging(self):
        rows = []
        for fid in ("one", "two"):
            for seed, multiplier in (("1", 0.5), ("2", 1.5)):
                for category in evaluation.CATEGORIES:
                    count = 10 if category == "semantic" else 100
                    for model in evaluation.MODELS:
                        value = 10 if model == "baseline" else (6 if category == "semantic" else 9)
                        rows.append({"recording_id": fid, "seed": seed, "model": model, "category": category,
                                     "region": "upper_hands", "metric": "mpjpe_mm", "numerator": value * multiplier * count,
                                     "denominator": count})
        differences, interactions = evaluation.paired_statistics(rows, 100, 1)
        self.assertEqual([r["foremotion_minus_baseline"] for r in differences], [-4, -1])
        self.assertEqual(interactions[0]["semantic_minus_beat_improvement"], 3)
        self.assertEqual(interactions[0]["ci95_low"], 3)
        self.assertEqual(interactions[0]["ci95_high"], 3)
        self.assertEqual(interactions[0]["recordings_with_both_categories"], 2)
        for row in rows:
            if row["recording_id"] == "two" and row["category"] == "semantic":
                row["denominator"] = row["numerator"] = 0
        _, interactions = evaluation.paired_statistics(rows, 100, 1)
        self.assertEqual(interactions[0]["recordings_with_both_categories"], 1)
        self.assertIsNone(interactions[0]["ci95_low"])

    def test_higher_is_better_interaction_uses_consistent_improvement_sign(self):
        rows = []
        for category in evaluation.CATEGORIES:
            for model in evaluation.MODELS:
                value = 0.2 if model == "baseline" else (0.8 if category == "semantic" else 0.3)
                rows.append({"recording_id": "one", "seed": "1", "model": model, "category": category,
                             "region": "upper_hands", "metric": "pck_100mm", "numerator": value * 10, "denominator": 10})
        _, interactions = evaluation.paired_statistics(rows, 0, 1)
        self.assertAlmostEqual(interactions[0]["semantic_minus_beat_improvement"], 0.5)

    def test_complete_pipeline_exports_all_metrics_on_selected_test_pairs(self):
        class FakeEvaluator:
            def __init__(self, cli):
                self.cli = cli

            def resample(self, pose, fps):
                return pose.copy()

            def joints(self, pose, beta):
                return pose.copy()

            def features(self, pose, reference, windows):
                return [pose[lo:hi].mean(axis=(0, 1))[None].repeat(2, 0) for lo, hi, _, _ in windows]

            def audio_onsets(self, fid, duration):
                return np.array([0.5, 2.0])

            def rhythm_scores(self, joints, onsets, segments=()):
                return np.full(len(onsets), 0.8 - float(joints.mean()))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "eval.hdf5"
            ids = ["2_scott_0_1_1", "2_scott_0_2_2", "2_scott_0_9_9"]
            with h5py.File(database_path, "w") as database:
                database.attrs["motion_fps"] = 30
                database.create_dataset("chunk_ids", data=np.array(ids, dtype="S"))
                database.create_dataset("chunk_splits", data=np.array(["test", "test", "train"], dtype="S"))
                for fid in ids:
                    group = database.create_group(fid)
                    group.attrs.update(file_id=fid, chunk_startsec=0.0, chunk_endsec=128 / 30)
                    group.create_dataset("motion", data=np.zeros((128, 55, 3), dtype=np.float32))
            input_hash = evaluation.sha256(database_path)
            sem = root / "sem"
            sem.mkdir()
            for fid in ids[:2]:
                (sem / f"{fid}.txt").write_text("01_beat_align 0 4.2666666667 4.2666666667 0.1\n05_iconic_l 0 1.1 1.1 0.5\n", encoding="utf-8")
                for model, value in (("baseline", 0.1), ("foremotion", 0.05)):
                    write_motion(root / model / fid / "pred.npz", frames=128, value=value)
                    write_motion(root / model / fid / "gt.npz", frames=128)
            cli = evaluation.parse_args(["--eval-hdf5", str(database_path), "--baseline-dir", str(root / "baseline"),
                                         "--foremotion-dir", str(root / "foremotion"), "--annotations-dir", str(sem),
                                         "--input-fps", "30", "--min-fgd-windows", "2", "--bootstrap", "50",
                                         "--output-dir", str(root / "out")])
            with patch.object(evaluation, "MotionEvaluator", FakeEvaluator), redirect_stdout(StringIO()):
                output = evaluation.run(cli)
            self.assertEqual(evaluation.sha256(database_path), input_hash)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["recordings"], 2)
            with (output / "coverage.csv").open() as stream:
                coverage = list(csv.DictReader(stream))
            self.assertEqual(len(coverage), 2)
            self.assertAlmostEqual(float(coverage[0]["semantic_seconds"]), 1.1)
            with (output / "fgd_by_seed.csv").open() as stream:
                fgd = list(csv.DictReader(stream))
            self.assertEqual(len(fgd), 4)
            self.assertTrue(all(row["value"] != "" for row in fgd))
            self.assertTrue((output / "paired_comparison.csv").is_file())
            self.assertTrue((output / "semantic_vs_beat_interaction.csv").is_file())
            self.assertIn("Primary interaction", (output / "report.md").read_text())
            with self.assertRaisesRegex(ValueError, "new/empty"):
                evaluation.run(cli)
            manifest = root / "pairs.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["recording_id", "seed", "baseline_path", "foremotion_path"])
                writer.writeheader()
                for fid in ids[:2]:
                    for seed in (1, 2):
                        writer.writerow({"recording_id": fid, "seed": seed,
                                         "baseline_path": f"baseline/{fid}/pred.npz", "foremotion_path": f"foremotion/{fid}/pred.npz"})
            cli = evaluation.parse_args(["--eval-hdf5", str(database_path), "--pairs-manifest", str(manifest),
                                         "--annotations-dir", str(sem), "--input-fps", "30", "--metrics", "geometry",
                                         "--bootstrap", "50", "--output-dir", str(root / "out_two_seeds")])
            with patch.object(evaluation, "MotionEvaluator", FakeEvaluator), redirect_stdout(StringIO()):
                output = evaluation.run(cli)
            with (output / "paired_comparison.csv").open() as stream:
                comparisons = list(csv.DictReader(stream))
            self.assertTrue(all(int(row["paired_recordings"]) == 2 for row in comparisons))
            with (output / "metrics_summary.csv").open() as stream:
                summary = list(csv.DictReader(stream))
            self.assertTrue(all(int(row["seeds_with_observations"]) == 2 for row in summary))


if __name__ == "__main__":
    unittest.main(verbosity=2)
