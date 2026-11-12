# Semantic versus beat gesture evaluation

`scripts/gesture_category_evaluation.py` evaluates **your system's generated
motion against ground truth**, separately on semantic-present and beat-only
intervals in the BEATX test recordings. Supply one saved motion directory; no
baseline model is required. The script uses existing `sem/` annotations and
local evaluator assets. It does not construct a generator, train a model or
contact Hugging Face.

This analysis characterizes the evaluated system's performance by gesture
category. Attributing a category-specific improvement to ForeMotion training
would require a separate matched model comparison.

## 1. Save the generated motions

Use the existing `scripts/test.py` command with `--save True` and
`--visualize False`. For the ForeMotion checkpoint discussed in this project:

```bash
CUDA_VISIBLE_DEVICES=2 \
NO_TORCH_COMPILE=1 \
HF_HUB_OFFLINE=1 \
OMP_NUM_THREADS=4 \
python scripts/test.py \
  --config experiments/0807_2331_gtdm3_sharedregretrvq_global_scott_densefg_bf16_4_regret_6_weight_cosine/config.yaml \
  --test_ckpt experiments/0807_2331_gtdm3_sharedregretrvq_global_scott_densefg_bf16_4_regret_6_weight_cosine/last_1000.safetensors \
  --is_train False \
  --dataset_ratio scott_beatx_fulllength \
  --beatx_cache_path datasets/data_cache/beatx_eval/database.hdf5 \
  --batch_size 1 --loader_workers 8 \
  --eval_generation_mode production --cfg_scale 1.5 --random_seed 2342 \
  --save True --visualize False --wandb False
```

The standard `test.py` initialization still requires its Moshi files to be cached
when `HF_HUB_OFFLINE=1` is used. That restriction concerns motion generation;
the category-analysis script itself has no Moshi dependency.

This produces files such as:

```text
experiments/<run>/test_1000/2_scott_0_1_1/gt.npz
experiments/<run>/test_1000/2_scott_0_1_1/pred.npz
```

`test.py` writes to `test_<checkpoint_epoch>` again on subsequent calls, so
preserve each seed's exports in a separate directory before generating another
seed. Existing saved motions can be reused without regenerating them.

## 2. Evaluate your system

Copy the new script and the existing `scripts/gesture_category_composition.py`
to the server. Keep the existing `scripts/trainers/utils/` modules there as well.
Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=4 \
python scripts/gesture_category_evaluation.py \
  --eval-hdf5 datasets/data_cache/beatx_eval/database.hdf5 \
  --motion-dir experiments/0807_2331_gtdm3_sharedregretrvq_global_scott_densefg_bf16_4_regret_6_weight_cosine/test_1000 \
  --annotations-dir datasets/beat_english_v2.0.0/sem \
  --data-dir datasets/beat_english_v2.0.0 \
  --dataset-ratio scott_beatx_fulllength \
  --split test \
  --generation-seed 2342 \
  --input-fps 25 \
  --output-dir reports/semantic_vs_beat_scott
```

All selected recordings must have `pred.npz`, `gt.npz` and an annotation file.
Missing recordings are not silently dropped. `--foremotion-dir` remains an alias
for `--motion-dir`. The default split is test only. Use `full_beatx_fulllength` for the
existing 23-speaker evaluation protocol. `--speakers` can further restrict it.

The `sem/` directory can contain training, validation and test annotations
together. The script first selects the requested split from the evaluation
HDF5, then checks `<motion-dir>/<recording_id>/pred.npz` and `gt.npz`, and reads
only `sem/<recording_id>.txt` (or a supported CSV/TSV with that exact stem).
For example, `test_1000/2_scott_0_1_1/` uses `sem/2_scott_0_1_1.txt`.
Other annotation files and extra output folders are ignored. The directory name
`test_1000` is not itself used as evidence of a recording's split. Keep
`--split test` and check `selected_recordings.csv` for the exact evaluated IDs;
the same list is recorded in metadata. You do not need to remove training
annotation files from `sem/`.

The default metrics include geometry, FGD and beat alignment. For an initial
geometry-only check, add `--metrics geometry --max-recordings 1` and use a new
output directory. Such a partial run is marked in metadata. `--device cpu`
works, although SMPL-X is faster on CUDA.

Required local assets:

- SMPL-X: `assets_dep/smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz`
- FGD encoder: `datasets/beat_english_v2.0.0/weights/AESKConv_240_100.bin`
- Beat normalization: `datasets/beat_english_v2.0.0/weights/mean_vel_smplxflame_30.npy`
- Beat audio: `<data-dir>/wave16k/<recording_id>.wav`

Use `--smplx-path`, `--fgd-weights` and `--mean-velocity` for other local asset
locations. Geometry-only mode does not require FGD weights or audio assets.

## Frame rates and alignment

**The existing `test.py` exports 25-fps arrays but writes 30 in the NPZ
`mocap_frame_rate` field.** Therefore `--input-fps 25` overrides that declaration.
The source declaration and actual chosen input rate are both recorded in
`input_manifest.csv`. For genuinely 30-fps arrays, use `--input-fps 30`.

The script checks motion duration against the selected full-sequence HDF5.
It permits at most one missing input tail frame by default, matching the
two-frame codec group truncation in `test.py`. Prediction and reference frame
counts must match exactly between prediction and reference, and across seeds; it does not time-warp,
silently truncate mismatched predictions, align gestures after generation, or
pad short sequences. Use `--max-tail-frames` only for a separately justified
export protocol.

Rotations are resampled with the existing evaluator's SLERP to 30 fps.
Annotation timestamps remain in seconds. For geometric metrics, each frame
represents the half-open time bin `[t/30,(t+1)/30)`, and exact interval overlap
provides its fractional weight. This retains sub-frame annotation durations.
Finite supplied motion is evaluated, as in the existing generation evaluator;
`pose_valid` flags do not remove frames in this protocol.

## Category definitions

The existing annotation reader handles the native BEAT labels and TXT/CSV/TSV
formats. [The dataset authors](https://pantomatrix.github.io/BEAT/) define beat
and the deictic/iconic/metaphoric categories; quality levels are pooled here.

- **Semantic-present:** union of deictic, iconic and metaphoric intervals.
- **Beat-only:** beat intervals outside that semantic union.
- Overlapping semantic labels count once. Semantic-plus-beat intervals belong
  to semantic-present; this is an explicit analysis policy, not a claim that
  these functions cannot co-occur.
- `need_cut` and unknown labels mask overlapping time. Explicit no-gesture
  and unlabeled time are excluded from both primary groups. Missing annotation
  files stop the run rather than being treated as beat or no gesture.

The evaluation masks come from the reference annotations. Full continuous
utterances are generated before any category selection.
`coverage.csv` reports seconds, contiguous semantic/beat event counts, overlap,
exclusions and FGD window coverage for every recording.

## Metric definitions

**MPJPE and PCK:** SMPL-X first 55 joints in metres are converted to millimetres.
Translation is zero and each pose's pelvis position is subtracted; global
orientation and articulation remain. There is no Procrustes or rotation
alignment. The model uses the reference body shape and `flat_hand_mean=True`,
matching the existing generation evaluator. MPJPE is mean Euclidean joint
distance; PCK is the fraction of joint errors strictly below 50, 100 or 150 mm.
Use `--pck-thresholds-mm` to declare other thresholds before inspecting results.

Results are reported for the 13 upper-body joints, 30 finger joints and their
43-joint union. Counts include exact fractional frame exposure. The additional
rotation error is the geodesic difference in degrees between local joint
rotations. These are reference-motion fidelity metrics on annotated categories.

**Category FGD:** the existing frozen BEAT encoder receives full-body 6D rotations
with the existing reference-jaw substitution. It encodes *continuous* 32-frame
windows at 30 fps, on a shared nonoverlapping grid by default. At least 80% of a
window must lie in a category. Unlabeled/no-gesture time does not count toward
that fraction. Any flagged/unknown-label overlap rejects the entire FGD window.
A window can enter at most one category. No disjoint annotation spans
are stitched together. `--window-frames`, `--window-stride` and `--window-purity`
configure this explicitly.

Window selection depends only on reference annotations. FGD is calculated
over pooled latent rows for each category/model/seed, never averaged from
individual-recording FGD. A category needs at least 20 windows by default;
otherwise its FGD is blank with `insufficient_windows`. Values with no more
feature rows than feature dimensions are flagged `computed_low_feature_count`.
Window counts and frame coverage must accompany these scores. This windowed
protocol differs from the original full-utterance FGD and does not establish
that scores from different-sized category populations have equal estimation bias.

**Category BeatAlign:** audio onsets and velocity-minimum candidates are detected
on continuous audio/motion. The existing mean-speed normalization, order-7
extrema detector, 13 upper-body joints and Gaussian sigma of 0.3 seconds are
used. For each audio onset, the score averages its nearest-motion-beat match
over those joints. No candidates means zero; no eligible audio onsets means
unavailable. Only onsets in the category are pooled. A two-second recording-edge
margin is excluded by default. Candidate motion beats retain surrounding
context, avoiding artificial beats at annotation cuts. This onset-pooled
category adaptation is not the existing whole-utterance frame-weighted score.
Candidates within flagged/unknown spans or their eight-frame derivative/extrema
neighborhood are discarded.
Ground-truth alignment is exported as a reference under `ground_truth`.

## Category scores and uncertainty

`category_metrics.csv` reports the system's MPJPE, PCK, rotation error and
BeatAlign separately for semantic-present and beat-only time, with 95%
recording-bootstrap confidence intervals. MPJPE/rotation error are lower-is-
better; PCK/BeatAlign are higher-is-better. Frames and joints are not independent
bootstrap units. Intervals are omitted when fewer than two recordings have
observations in a category. A missing category is reported as unavailable,
rather than receiving zero error. FGD is reported separately with sample counts.

With several generation seeds, numerators are averaged within each recording
before recording-bootstrap inference. Seeds do not increase the number of
independent recordings. Per-seed dataset scores and seed standard deviations
are also exported. Intervals are conditional on the selected checkpoints,
generation seeds, recordings and speaker cohort.

These metrics test motion fidelity/timing at reference-annotated semantic
events. They do not independently determine whether a generated gesture
expresses the speech meaning; a different realization may also be semantically
valid. Category difficulty and sample size differ, so a smaller score in one
category does not establish that ForeMotion training benefits that category more.

The previous paired mode remains optional: explicitly supplying `--baseline-dir`
alongside `--motion-dir` also generates paired model differences and a semantic-
versus-beat improvement interaction. Neither baseline files nor interaction
outputs are used in the single-system analysis.

## Several generation seeds

Use `--motion-manifest motions.csv` in place of `--motion-dir`:

```csv
recording_id,seed,prediction_path
2_scott_0_1_1,2342,/exports/foremotion/2342/2_scott_0_1_1/pred.npz
2_scott_0_1_1,3453,/exports/foremotion/3453/2_scott_0_1_1/pred.npz
```

Include every selected recording for every seed. Paths can be absolute or
relative to the manifest's directory. `gt.npz` is read beside each prediction;
an optional `reference_path` column overrides this. References must agree across
seeds. Duplicate seed/recording rows and incomplete recording coverage are
rejected. Legacy `--pairs-manifest` / `foremotion_path` names are accepted;
`baseline_path` and model-specific reference columns are only needed for the
optional paired comparison.

## Outputs and verification

- `report.md`: main system performance tables by category and FGD table.
- `selected_recordings.csv`: exact evaluated recording IDs, verified split and annotation paths.
- `category_metrics.csv`: per-category system scores with 95% confidence intervals.
- `metrics_by_seed.csv` / `metrics_summary.csv`: dataset scores and seed means/std.
- `per_recording_metrics.csv`: numerators and denominators for reproducibility.
- `fgd_by_seed.csv` / `fgd_windows.csv`: when requested, scores/status and exact windows.
- `coverage.csv` / `annotation_intervals.csv`: population, exposure and exclusions.
- `input_manifest.csv`: prediction/reference paths, hashes and frame-rate declarations.
- `metadata.json`: settings, definitions, script hash, timing and complete/failed status.

`paired_comparison.csv` and `semantic_vs_beat_interaction.csv` are produced only
when an optional baseline is explicitly provided.

Use a fresh output directory for each run. Partial per-recording results are
written progressively; automatic resume is not implemented.

```bash
python -B -m scripts.smoke_test_gesture_category_evaluation
```

Synthetic checks cover annotation overlap/exclusions, exact-time masks,
sub-frame events, fixed continuous windows, millimetre/PCK arithmetic,
reference/fps validation, seed pairing, known Fréchet controls, continuous
rhythm scores, recording-bootstrap category estimates/interactions and complete
single-model and paired single/multiple-seed pipeline exports. Local component checks also exercised real SMPL-X on
CUDA, the evaluator architecture with synthetic encoder weights, and its rhythm
detector. These checks do not represent real BEATX model results.
