# BEATX gesture composition by speaker

`scripts/gesture_category_composition.py` measures the time annotated as beat,
deictic, iconic and metaphoric gestures in your evaluation recordings. Run it
on the remote server containing the data; it requires no trained model or GPU.
Its purpose is to characterize the evaluation set when investigating differences
between Scott and multi-speaker ForeMotion results.

## Run on the remote server

From your existing project directory and Python environment:

```bash
python scripts/gesture_category_composition.py \
  --eval-hdf5 datasets/data_cache/beatx_eval/database.hdf5 \
  --data-dir datasets/beat_english_v2.0.0 \
  --dataset-ratio full_beatx_fulllength \
  --output-dir reports/gesture_composition_test
```

Set `--eval-hdf5` to the **same full-sequence HDF5 used for your evaluation**.
Set `--annotations-dir /path/to/sem` if annotations are outside
`<data-dir>/sem`. The selected test groups determine the recording list and
their motion durations determine the time bounds. The script reads only HDF5
metadata/shapes, not full motion arrays, and writes no input caches or sidecars.
Training chunks (`_C0`, etc.) are rejected to avoid substituting training clips
for complete evaluation recordings.

The default `full_beatx_fulllength` filter excludes Carla and Itoi, matching the
existing loader's 23-speaker protocol. `scott_beatx_fulllength` selects Scott;
`goodspk_beatx_fulllength` selects Wayne, Lawrence, Stewart and Solomon;
`full_beatx_eval` admits all speakers present in the cache. Use the ratio from
the actual evaluation, and optionally narrow it with `--speakers scott wayne`.
The script does not assume that the cache contains exactly 249 or 15 recordings;
the exported recording list is authoritative for this audit.

Dependencies: `numpy`, `h5py` and `matplotlib` from the existing environment.
`--no-plots` omits Matplotlib; manifest mode also does not require h5py.

## Annotations

The native [BEAT annotation format](https://github.com/PantoMatrix/BEAT#dataset)
has one `sem/<recording_id>.txt` file per recording. Each whitespace-delimited
row contains `types start end duration score keywords`. For example:

```text
01_beat_align 0.0 8.0 8.0 0.1
04_deictic_h 2.0 4.0 2.0 0.4 pointing at the door
```

The type ID mapping is 0 = no gesture, 1 = beat, 2--4 = deictic, 5--7 = iconic,
8--10 = metaphoric, as documented by the
[dataset authors](https://pantomatrix.github.io/BEAT/). Quality levels within a
category are combined. **The relevance `score` is not a category ID and does
not weight time.** Durations are recomputed from end minus start, in seconds.
Timestamps stay in seconds; no 30-to-25-fps conversion is applied to annotation
times. The HDF5 motion duration uses its stored motion rate (fallback 25 fps).

Per-recording `.csv` or `.tsv` files also work, with headers
`category,start_seconds,end_seconds` or `types,start,end`. Use the same recording
ID as the filename. Explicit numeric type IDs 0--10 and category names are
accepted. Unmapped labels are reported separately by default. Nonfinite
timestamps, reversed intervals, inconsistent type IDs/names, duplicate candidate
files and conflicting recording entries raise an error. Intervals
crossing recording bounds are clipped; intervals outside the bounds and
zero-duration annotations are counted in diagnostics.

`need_cut` is also accepted as a **flagged interval**, following its occurrence
in the released `15_carlos_0_1_1.txt` annotations. Its formal upstream meaning
has not been verified, so the script makes a conservative analysis choice:
exclude that time from the analysis rather than map it to a gesture or to no
gesture. Exclusions mask all overlapping labels, including beat. Overlapping
`need_cut` intervals are merged implicitly and clipped to the evaluation bounds.
They appear as `excluded_seconds` and a separate coverage-plot segment;
`need_cut_rows` counts source rows, including ones outside evaluation bounds.
The selected recording bounds and source annotation files are preserved.

The default `--unknown-labels separate` applies the same time exclusion to
other non-standard labels and measures each distinct label separately, so a
new annotation label does not interrupt the whole audit. Label strings are
trimmed and lowercased for grouping. The script makes no assumption about
what an unmapped label means. For example, two `self_touch` intervals covering
seconds 1--4 and 2--5 contribute **4 seconds**, rather than 6, to that label's
separate table. If a different extra label covers seconds 3--7, it contributes
4 seconds to its own table; their **combined exclusion is 6 seconds**. Per-label
durations may overlap across labels and must not be summed as exclusive time.

For strict label validation, pass `--unknown-labels error`; the explicitly
recognized `need_cut` flag remains supported. Both modes still reject malformed
timestamps and contradictions between a known type ID and its category name.

The project's dataset downloader retrieves motion/audio/transcript subfolders
but **does not retrieve `sem/`**. Use the semantic files from your BEAT2 release
or obtain that subfolder from the
[upstream dataset](https://huggingface.co/datasets/H-Liu1997/BEAT2/tree/main/beat_english_v2.0.0/sem).
Missing annotation files cause an error by default. `--missing-annotations skip`
allows an incomplete audit while reporting missing recordings and their time.
Missing files are never treated as zero gestures. Empty existing files have
zero labeled time and their duration is reported as unlabeled.

## Overlap and proportions

The primary composition is:

```text
category fraction = allocated category seconds / union of all four gesture categories
```

An interval sweep integrates exact timestamp boundaries. It merges repeated
annotations of the same category implicitly; a duplicate never adds time.
Each instant contributes at most one second of allocated category time per
second of recording, even when different labels overlap.

The default `--overlap-policy semantic-over-beat` treats overlapping deictic,
iconic or metaphoric labels as overriding beat. Simultaneous semantic categories
share the interval equally. This is an explicit analysis choice motivated by
the [released annotation example](https://huggingface.co/datasets/H-Liu1997/BEAT2/blob/main/beat_english_v2.0.0/sem/10_kieks_0_1_1.txt),
which includes a metaphoric interval nested inside a beat interval; it is not a
claim that every annotation conflict has a unique ground truth.

For the example above, the primary result is 6 seconds beat and 2 seconds
deictic, giving 75% and 25%. A sensitivity run with
`--overlap-policy fractional` shares overlap equally among all active categories,
giving 7 seconds beat and 1 second deictic. Use a different output directory
for this run. With either policy, total allocated time is 8 seconds.

The script also exports each category's union duration after flagged-span
exclusions and before cross-category allocation, and the time with beat/semantic, multiple semantic, or no-gesture/
gesture conflicts. **Raw category unions may sum above total gesture time; do
not use them as a 100% stacked composition.** Explicit no-gesture annotations
are counted only outside the four gesture categories. Unlabeled time is kept
separate and is not inferred to be beat, rest, or missing motion.

Speaker/cohort compositions pool duration before dividing, so a longer gesture
interval contributes more than a shorter one. `fraction_of_evaluated_time`
provides the second denominator: duration with available annotation files minus
flagged-span exclusions. This is `evaluated_seconds` in `speaker_summary.csv`;
`selected_seconds` retains the original time bounds. Coverage tables/plots expose
excluded time, unlabeled time and missing annotation files separately.

## Outputs

- `selected_recordings.csv`: exact selected IDs, speakers and time bounds.
- `recording_summary.csv`: per-recording allocated seconds, raw unions, overlap,
  coverage, extra-label row counts, source annotation hashes and missing-file status.
- `speaker_summary.csv`: counts, exposure, coverage and four category proportions.
- `speaker_categories.csv`: one row per speaker/category, with seconds,
  proportions and bootstrap confidence intervals.
- `cohort_categories.csv`: all-selected, Scott and other-speakers-pooled results.
  All-selected includes Scott; these three cohorts are not independent samples.
- `extra_labels_by_recording.csv`: each non-standard label's source-row count,
  number of positive-duration rows intersecting the evaluated time bounds, and
  union duration in seconds, per recording.
- `extra_labels_by_speaker.csv`: those counts and durations pooled by speaker
  and label, with counts of recordings containing the label and positive label time.
  Both files include headers even when no extra labels occur. Source-row counts
  include rows outside evaluation bounds; union durations use only the evaluated
  interval. `metadata.json` also contains overall totals per extra label.
- `speaker_composition.png/.pdf`: stacked proportions among labeled gesture time.
- `annotation_coverage.png/.pdf`: categorized, explicit no-gesture, unlabeled,
  non-standard-label, and missing-file time as fractions of selected evaluation duration.
- `scott_vs_other_speakers.png/.pdf`: pooled comparison, when both groups exist.
- `metadata.json` and `summary.txt`: settings, selection/script fingerprints,
  statistical assumptions and interpretation.

The default 2,000-replicate bootstrap resamples **whole recordings** within a
speaker, and reports percentile 95% confidence intervals for the pooled ratio.
Intervals are omitted with fewer than two analyzable recordings or no labeled
gesture time. Fully excluded recordings do not supply bootstrap units. Cohort
intervals resample recordings within the selected cohort and are
conditional on the speakers selected; they do not generalize to unseen speakers.
Use `--bootstrap 0` to disable intervals.

For an optional exploratory omnibus comparison, add `--permutations 2000`.
This tests between-speaker variation in the four-dimensional **per-recording**
composition, weighting recordings equally and shuffling speaker labels while
preserving group sizes. It exports a between-speaker variance fraction and a
Monte Carlo p-value with the plus-one correction. It needs at least two positive-
gesture recordings per speaker. It does not treat individual frames or seconds
as independent observations. Repeated prompts, emotions and session conditions
can violate exchangeability, so the optional test is not a controlled estimate
of a speaker effect. The main duration-pooled tables use a different weighting.

Differences in composition can motivate category-specific ForeMotion evaluations;
they alone do not establish why ForeMotion improves more in one speaker setting.
That further step requires generated-motion metrics for comparable recordings,
with speech content and conditions considered.

## Manifest alternative

If you have the exact evaluated recording list rather than the cache:

```csv
recording_id,speaker,duration_seconds
2_scott_0_1_1,scott,65.0
1_wayne_0_1_1,wayne,65.0
```

```bash
python scripts/gesture_category_composition.py \
  --manifest /path/to/evaluated_recordings.csv \
  --annotations-dir /path/to/sem \
  --output-dir reports/gesture_composition_manifest
```

`id`/`file_id` can replace `recording_id`; the speaker can be inferred from BEAT
IDs. Optional `type`/`split` columns select `test` by default. The official split
CSV is accepted, but its listed recordings can differ from recordings actually
retained in an evaluation cache. Include `end_seconds`, or `duration_seconds`,
to use known evaluation bounds. Without durations, WAV header duration from
`<data-dir>/wave16k/<id>.wav` is used, which can differ slightly from the motion
duration. Optional `start_seconds` supports explicitly trimmed manifests.
Identical duplicate entries count once; conflicting bounds raise an error.

Outputs require a new or empty directory, so separate policy runs remain
distinguishable. Omitting `--output-dir` creates a timestamped folder under
`reports/`.

## Verification

```bash
python -B -m scripts.smoke_test_gesture_category_composition
```

The synthetic checks cover native label parsing, duplicate/nested/multi-category
intervals, exact durations, clipping, split and speaker selection, rejected
training chunks, missing annotations, reproducibility, input integrity and
table/chart exports. These are not measurements of the remote dataset.
