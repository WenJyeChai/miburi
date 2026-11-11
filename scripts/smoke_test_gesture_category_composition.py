"""Analytical checks; no remote data, model weights or GPU needed.

Run: python -B -m scripts.smoke_test_gesture_category_composition
"""

from contextlib import redirect_stdout
from io import StringIO
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from scripts import gesture_category_composition as audit


def make_cache(path, records):
    with h5py.File(path, "w") as database:
        database.attrs["motion_fps"] = 25
        ids, splits = [], []
        for recording_id, split, duration in records:
            group = database.create_group(recording_id)
            group.attrs.update(file_id=recording_id, split=split, chunk_startsec=0.0, chunk_endsec=duration)
            group.create_dataset("motion", shape=(int(duration * 25), 1, 3), dtype="float32")
            ids.append(recording_id)
            splits.append(split)
        database.create_dataset("chunk_ids", data=np.array(ids, dtype="S"))
        database.create_dataset("chunk_splits", data=np.array(splits, dtype="S"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CompositionChecks(unittest.TestCase):
    def test_type_ids_and_names_not_relevance_scores(self):
        expected = ["none", "beat"] + ["deictic"] * 3 + ["iconic"] * 3 + ["metaphoric"] * 3
        for number, category in enumerate(expected):
            self.assertEqual(audit.category_from_label(str(number)), category)
        self.assertEqual(audit.category_from_label("04_deictic_h"), "deictic")
        self.assertEqual(audit.category_from_label("10_metaphoric_h"), "metaphoric")
        self.assertEqual(audit.category_from_label("metaphorical"), "metaphoric")
        self.assertEqual(audit.category_from_label("need_cut"), "need_cut")
        with self.assertRaises(ValueError):
            audit.category_from_label("0.4")
        with self.assertRaisesRegex(ValueError, "disagree"):
            audit.category_from_label("04_iconic_h")
        with self.assertRaises(ValueError):
            audit.category_from_label("mystery")

    def test_overlap_semantic_priority_duplicate_labels_and_conflicts(self):
        rows = [audit.Annotation("beat", 0, 8)] * 2 + [
            audit.Annotation("iconic", 2, 6), audit.Annotation("deictic", 4, 7),
            audit.Annotation("none", 8, 9), audit.Annotation("none", 3, 5),
            audit.Annotation("metaphoric", 10, 12),
        ]
        measured = audit.measure_intervals(rows, 0, 10)
        np.testing.assert_allclose(list(measured["allocated"].values()), [3, 2, 3, 0])
        np.testing.assert_allclose(list(measured["category_union"].values()), [8, 3, 4, 0])
        for key, expected in {
            "gesture_seconds": 8, "explicit_no_gesture_seconds": 1, "unlabeled_seconds": 1,
            "cross_category_overlap_seconds": 5, "beat_semantic_overlap_seconds": 5,
            "semantic_category_overlap_seconds": 2, "none_gesture_conflict_seconds": 2,
        }.items():
            self.assertAlmostEqual(measured[key], expected)
        fractional = audit.measure_intervals(rows, 0, 10, "fractional")
        np.testing.assert_allclose(list(fractional["allocated"].values()), [31 / 6, 7 / 6, 5 / 3, 0])
        self.assertAlmostEqual(sum(fractional["allocated"].values()), 8)

    def test_clipping_touching_endpoints_and_zero_lengths(self):
        rows = [audit.Annotation("beat", -1, 2), audit.Annotation("deictic", 2, 3),
                audit.Annotation("iconic", 3, 5), audit.Annotation("beat", 12, 13),
                audit.Annotation("beat", 3, 3)]
        result = audit.measure_intervals(rows, 1, 4)
        np.testing.assert_allclose(list(result["allocated"].values()), [1, 1, 1, 0])
        self.assertEqual(result["clipped_rows"], 2)
        self.assertEqual(result["outside_rows"], 1)
        self.assertEqual(result["zero_length_rows"], 1)
        self.assertEqual(result["cross_category_overlap_seconds"], 0)
        with self.assertRaises(ValueError):
            audit.measure_intervals([audit.Annotation("beat", 3, 2)], 0, 4)
        with self.assertRaises(ValueError):
            audit.measure_intervals([audit.Annotation("beat", 0, float("nan"))], 0, 4)

    def test_need_cut_masks_overlapping_categories_and_merges_exclusions(self):
        annotations = [
            audit.Annotation("beat", 0, 10), audit.Annotation("iconic", 2, 7),
            audit.Annotation("need_cut", 3, 6), audit.Annotation("need_cut", 5, 8),
            audit.Annotation("need_cut", 20, 21),
        ]
        for policy in audit.POLICIES:
            result = audit.measure_intervals(annotations, 0, 10, policy)
            self.assertEqual(result["excluded_seconds"], 5)
            self.assertEqual(result["need_cut_rows"], 3)
            self.assertEqual(result["gesture_seconds"], 5)
            self.assertEqual(result["category_union"]["beat"], 5)
            self.assertEqual(result["category_union"]["iconic"], 1)
            self.assertAlmostEqual(sum(result["allocated"].values()), 5)
            self.assertAlmostEqual(sum(result[field] for field in (
                "gesture_seconds", "explicit_no_gesture_seconds", "unlabeled_seconds", "excluded_seconds",
            )), 10)
        result = audit.measure_intervals(annotations, 0, 10)
        self.assertEqual(result["allocated"]["beat"], 4)
        self.assertEqual(result["allocated"]["iconic"], 1)
        clipped = audit.measure_intervals([audit.Annotation("need_cut", -5, 5)], 0, 4)
        self.assertEqual(clipped["excluded_seconds"], 4)
        self.assertEqual(clipped["clipped_rows"], 1)
        self.assertEqual(clipped["unlabeled_seconds"], 0)

    def test_all_cut_recording_has_no_fabricated_proportions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sem = root / "sem"
            sem.mkdir()
            cache = root / "eval.hdf5"
            make_cache(cache, [("15_carlos_0_1_1", "test", 4)])
            (sem / "15_carlos_0_1_1.txt").write_text(
                "01_beat_align 0 4 4 0.1\nneed_cut 0 4 4 0\n", encoding="utf-8",
            )
            args = audit.parse_args(["--eval-hdf5", str(cache), "--annotations-dir", str(sem),
                                     "--output-dir", str(root / "out"), "--no-plots"])
            with redirect_stdout(StringIO()):
                output = audit.run(args)
            row = read_csv(output / "recording_summary.csv")[0]
            self.assertEqual(float(row["excluded_seconds"]), 4)
            self.assertEqual(float(row["gesture_seconds"]), 0)
            speaker = read_csv(output / "speaker_summary.csv")[0]
            self.assertEqual(float(speaker["evaluated_seconds"]), 0)
            self.assertEqual(speaker["beat_fraction"], "")
            self.assertEqual(speaker["annotation_coverage_fraction"], "")
            self.assertTrue(all(r["ci95_low"] == "" for r in read_csv(output / "speaker_categories.csv")))

    def test_native_txt_csv_and_empty_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.txt"
            path.write_text("01_beat_align 0 8 8 0.1\n04_deictic_h 2 4 2 0.4 pointing at tree\n", encoding="utf-8")
            rows = audit.load_annotations(path)
            self.assertEqual(rows, [audit.Annotation("beat", 0, 8), audit.Annotation("deictic", 2, 4)])
            path = root / "sample.csv"
            path.write_text("types,start,end,score\n07_iconic_h,1,3,0.7\n", encoding="utf-8")
            self.assertEqual(audit.load_annotations(path), [audit.Annotation("iconic", 1, 3)])
            path.write_text("category,start_seconds,end_seconds\nbeat,0,nan\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                audit.load_annotations(path)
            path = root / "empty.txt"
            path.write_text("", encoding="utf-8")
            result = audit.measure_intervals(audit.load_annotations(path), 0, 4)
            self.assertEqual(result["gesture_seconds"], 0)
            self.assertEqual(result["unlabeled_seconds"], 4)

    def test_full_sequence_selection_and_scott_filter(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "eval.hdf5"
            make_cache(path, [("2_scott_0_1_1", "test", 4), ("1_wayne_0_1_1", "test", 8),
                              ("5_carla_0_1_1", "test", 4), ("2_scott_0_9_9", "train", 4)])
            args = audit.parse_args(["--eval-hdf5", str(path)])
            rows = audit.load_recordings(args)
            self.assertEqual([r.speaker for r in rows], ["scott", "wayne"])
            args.dataset_ratio = "scott_beatx_fulllength"
            self.assertEqual(len(audit.load_recordings(args)), 1)
            args.dataset_ratio = "full_beatx_eval"
            self.assertEqual(len(audit.load_recordings(args)), 3)
            with h5py.File(path, "a") as database:
                database["2_scott_0_1_1"].attrs["chunk_endsec"] = 10
            with self.assertRaisesRegex(ValueError, "disagrees"):
                audit.load_recordings(args)

    def test_training_chunks_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.hdf5"
            make_cache(path, [("2_scott_0_1_1_C0", "test", 4)])
            with self.assertRaisesRegex(ValueError, "training chunks"):
                audit.load_recordings(audit.parse_args(["--eval-hdf5", str(path)]))

    def test_manifest_deduplication_and_duration_bounds(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text("id,type,duration_seconds\n2_scott_0_1_1,test,4\n2_scott_0_1_1,test,4\n1_wayne_0_9_9,train,4\n", encoding="utf-8")
            args = audit.parse_args(["--manifest", str(path)])
            self.assertEqual(len(audit.load_recordings(args)), 1)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("2_scott_0_1_1,test,5\n")
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                audit.load_recordings(args)

    def test_bootstrap_uses_pooled_durations_and_is_reproducible(self):
        values = np.array([[1, 0, 0, 0], [0, 9, 0, 0]], dtype=float)
        a = audit.bootstrap_composition(values, 500, np.random.default_rng(2342))
        b = audit.bootstrap_composition(values, 500, np.random.default_rng(2342))
        self.assertEqual(a, b)
        self.assertEqual(a[0], (0, 1))
        self.assertEqual(audit.bootstrap_composition(values[:1], 500, np.random.default_rng(2342)), [(None, None)] * 4)
        self.assertEqual(audit.bootstrap_composition(np.zeros((2, 4)), 500, np.random.default_rng(2342)), [(None, None)] * 4)

    def test_missing_annotation_audit_never_imputes_zero_gestures(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "eval.hdf5"
            sem = root / "sem"
            sem.mkdir()
            make_cache(cache, [("2_scott_0_1_1", "test", 4), ("1_wayne_0_1_1", "test", 4)])
            (sem / "2_scott_0_1_1.txt").write_text("01_beat_align 0 4 4 0.1\n", encoding="utf-8")
            args = audit.parse_args(["--eval-hdf5", str(cache), "--annotations-dir", str(sem),
                                     "--no-plots", "--output-dir", str(root / "out")])
            with self.assertRaisesRegex(ValueError, "Missing annotation"):
                audit.run(args)
            args.missing_annotations = "skip"
            with redirect_stdout(StringIO()):
                output = audit.run(args)
            rows = read_csv(output / "recording_summary.csv")
            missing = next(r for r in rows if r["speaker"] == "wayne")
            self.assertEqual(missing["gesture_seconds"], "")
            summary = read_csv(output / "speaker_summary.csv")
            wayne = next(r for r in summary if r["speaker"] == "wayne")
            self.assertEqual(wayne["beat_fraction"], "")
            self.assertEqual(wayne["missing_recordings"], "1")

    def test_end_to_end_tables_plots_and_input_integrity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "eval.hdf5"
            sem = root / "sem"
            sem.mkdir()
            records = [("2_scott_0_1_1", "test", 4), ("2_scott_0_2_2", "test", 4),
                       ("1_wayne_0_1_1", "test", 4), ("1_wayne_0_2_2", "test", 4)]
            make_cache(cache, records)
            for recording_id, _, _ in records:
                semantic = "07_iconic_h" if "scott" in recording_id else "04_deictic_h"
                (sem / f"{recording_id}.txt").write_text(f"01_beat_align 0 4 4 0.1\n{semantic} 2 4 2 0.7\nneed_cut 1 2 1 0\n", encoding="utf-8")
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [cache, *sem.iterdir()]}
            args = audit.parse_args(["--eval-hdf5", str(cache), "--annotations-dir", str(sem),
                                     "--bootstrap", "100", "--permutations", "99", "--output-dir", str(root / "out")])
            with redirect_stdout(StringIO()):
                output = audit.run(args)
            for path, checksum in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
            for plot in ["speaker_composition", "annotation_coverage", "scott_vs_other_speakers"]:
                for extension in ["png", "pdf"]:
                    self.assertGreater((output / f"{plot}.{extension}").stat().st_size, 1000)
            categories = read_csv(output / "speaker_categories.csv")
            scott = {r["category"]: float(r["fraction_of_gesture_time"]) for r in categories if r["speaker"] == "scott"}
            self.assertEqual(scott, {"beat": 1 / 3, "deictic": 0.0, "iconic": 2 / 3, "metaphoric": 0.0})
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["selected_recordings"], 4)
            self.assertEqual(metadata["excluded_seconds"], 4)
            self.assertEqual(metadata["need_cut_rows"], 4)
            self.assertEqual(float(read_csv(output / "speaker_summary.csv")[0]["evaluated_seconds"]), 6)
            test = metadata["exploratory_permutation_test"]
            self.assertEqual(test["status"], "ok")
            self.assertAlmostEqual(test["between_speaker_variance_fraction"], 1)
            self.assertGreater(test["p_value"], 0)
            self.assertLessEqual(test["p_value"], 1)
            with self.assertRaisesRegex(ValueError, "not empty"):
                audit.run(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
