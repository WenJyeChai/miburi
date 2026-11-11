# Three-condition future-speech comparison

This is a frozen-checkpoint experiment for the ForeMotion rebuttal. It evaluates
one trained shared-weight ForeMotion model under three speech conditions on the
same selected test recordings:

| Condition | Direct temporal speech attention |
| --- | --- |
| Causal | The existing rolling past/current window |
| Matched future | The same past/current lower boundary plus the correct future |
| Mismatched future | The same past/current lower boundary plus another recording's future audio/text |

The script includes both teacher-forced token prediction and autoregressive
generated-motion evaluation. It constructs no optimizer and requires no Moshi
download: speech embedding weights come directly from the supplied checkpoint.

## Exact intervention

The default future scope is the **training clip length**, obtained from the
training config: normally 125 aligned groups / 10 seconds. During complete-
utterance evaluation, speech is divided into nonoverlapping clips of this
length. At prediction step t, future access ends at the end of t's clip. The
last clip may be shorter. This is a clip-bounded experiment, not an unlimited-
future oracle over the whole recording. Motion histories are continuous and
do not reset at clip boundaries. `--clip-steps` changes the future scope and is
recorded in metadata.

For each target clip, a seeded donor selection chooses a speech segment from
a **different utterance by the same speaker in the same split**. Its duration
matches the target clip exactly. Audio and text are taken from the same donor
segment. There is no padding, repetition, time stretching, cross-speaker
substitution, or target/donor self-pairing. For clips longer than one group,
the donor must differ somewhere **after the first group**, so a changed current
group alone cannot masquerade as a future intervention. `donor_manifest.csv`
records the full mapping, including offsets.
At least two eligible utterances per speaker are needed. With
`--max-recordings`, the donor pool still uses the full selected split.

For each prediction, the mismatched memory is constructed from the **original
target's entire prefix through the current group**, followed by the donor
suffix through the clip end. Temporal attention retains the existing rolling
past boundary; gesture self-attention stays causal. The kinematic transformer
receives the original current audio/text group in all three conditions.

The temporal prefix is recomputed at every prediction for each condition. This
is necessary: reusing states formed under the matched future could leak that
future into the mismatched condition. Earlier query positions retain their own
clip-end boundary. Consequently, a previous clip's states cannot suddenly gain
access to the next clip's speech. The recomputation is slower than the production
streaming generator; this experiment does not measure deployment latency.

## Run

Copy `scripts/future_speech_comparison.py` and
`miburi/models/gesture_lm_future_control.py` to the corresponding paths on your
remote server. The runner reuses the existing `scripts/codec_reconstruction_audit.py`,
models, loader and metric helpers. Use the environment already used for model
evaluation, and run from the repository root.

Replace `YOUR_FOREMOTION_RUN` and the checkpoint filename with the selected
trained run. **Use that run's saved training config**, including its matching
codec paths and kinematic weight-sharing setting. The speaker map defaults to
`speaker_id_to_index.json` beside the checkpoint. Supply `--speaker-map` if it
is elsewhere. Do not rebuild speaker indices from a filtered evaluation subset.

```bash
CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=4 \
python scripts/future_speech_comparison.py \
  --config experiments/YOUR_FOREMOTION_RUN/config.yaml \
  --checkpoint experiments/YOUR_FOREMOTION_RUN/last_1000.safetensors \
  --eval-hdf5 datasets/data_cache/beatx_eval/database.hdf5 \
  --dataset-ratio scott_beatx_fulllength \
  --split test \
  --mode both \
  --generation-seeds 2342 3453 4564 \
  --cfg-scale 1.0 \
  --output-dir reports/future_speech_comparison_scott
```

For a multi-speaker checkpoint, use its corresponding config/checkpoint and
`--dataset-ratio full_beatx_fulllength`. This uses the loader's existing protocol
excluding Carla and Itoi. The default split is **test only**. A training-split
analysis must be a separate run with `--split train` and a full-sequence cache
containing those recordings. Chunked training HDF5s are rejected.

The HDF5 must contain the full speech token streams, motion, contacts,
expressions, body shape and split metadata. Incomplete trailing codec frame
groups are dropped, matching the production generator: at 25 fps and chunk
size 2 this is at most one trailing frame. The dropped count is exported.
The finite supplied motion is used as in production evaluation; `pose_valid`
flags are not used to remove individual frames. Standard BEATX test recordings
in this protocol have valid lower/face regions; unavailable lower/face token
targets are masked if present.

The generated-motion stage requires CUDA, the existing SMPL-X assets at
`assets_dep/smplx_2020/`, evaluator weights at
`datasets/beat_english_v2.0.0/weights/`, and target waveform files under the
configured BEATX data directory. These are the existing `test.py` dependencies.
`--data-dir` overrides the source audio directory; the evaluator's asset/weight
paths remain the existing repository defaults. Optional `--upper-codec`,
`--lower-codec` and `--face-codec` override codec paths, but use the codecs with
which this generator was trained. The loader checks the supplied codecs'
frozen quantizer state against the state stored in the generator checkpoint;
a mismatched codec stops the run.

For a short plumbing check first, use `--max-recordings 1 --generation-seeds 2342`.
For a much faster token-only check, use
`--mode tokens --max-recordings 1 --token-stride 25`. Token-only mode also works
with `--device cpu`; full generated metrics require CUDA. These partial runs
are marked in metadata and should not be used as final rebuttal results.

## Token metrics

Gold motion is encoded deterministically with the frozen codecs, matching the
batched canonical targets used in production validation. All conditions receive
the **same gold past gesture history and within-step RVQ prefix**. Stochastic
training targets, input dropout, face-loss reweighting, KL losses and auxiliary
objectives are disabled for scoring.

Cross-entropy is ordinary hard-target CE in **nats**, normalized over all
original head classes, including the padding output class. Padding targets are
excluded. Top-1 accuracy and top-5 accuracy use those same heads and targets.
The overall value is a valid-token micro average; region and per-codebook tables
make the 8-upper / 8-lower / 4-face weighting explicit.

By default every position with at least one future group in its clip is scored.
The final step of each clip is excluded from the primary token comparison
because it has no future intervention. All three conditions score identical
positions. `--token-stride` subsamples positions uniformly, while
`--min-future-steps` can restrict scoring to longer remaining futures. The
generated-motion stage still generates every step.
`token_intervention_coverage.csv` reports how many scored positions actually
have changed donor tokens in their remaining future. Identical tails can occur
even when an earlier part of the donor clip differs; those positions remain in
the common scoring set and are identifiable in this coverage report.

Paired CE/accuracy differences and 95% bootstrap intervals resample **complete
recordings**, keeping each recording's conditions paired. They do not treat
individual frames or codebooks as independent samples. The pooled multi-speaker
analysis is conditional on the selected recordings/speakers.

## Generated-motion metrics

Each condition generates its own complete utterance autoregressively from the
same start token and speaker condition. Histories naturally diverge after
different predictions; identical gold histories apply to token scoring only.

The default sampler uses the existing temperatures (0.9) and top-p values
(temporal 0.8, depth 0.95). A common uniform sampling variate is used across
conditions for each step/codebook to reduce sampling noise. Donor assignments
stay fixed across generation seeds. CFG is **1.0 by default** to keep this
intervention easy to interpret; use your benchmark CFG explicitly if comparing
against earlier scores. As in the existing generator, masked lower-body heads
skip classifier-free guidance. `--sampler greedy` is a separate deterministic control.

FGD, BeatAlign, L1 diversity and `facial_l2` reuse the existing `GestureMetrics`
evaluator. Its `facial_l2` is a face-vertex MSE after resampling to 30 fps and
the existing ground-truth jaw substitution; retain this protocol when comparing
with earlier benchmark scores. An additional `expression_mse` measures the raw
generated versus reference expression coefficients at 25 fps without that jaw
substitution. These metrics have different units and must not be interchanged.
BeatAlign and L1 diversity include their ground-truth reference values.

The evaluator's frozen networks are shared, while each condition has independent
metric accumulators. Per-seed dataset metrics and across-seed means/std are
reported separately. FGD is computed over the complete selected dataset for
each condition/seed, not averaged from per-recording FGD values.

## Outputs and interpretation

- `side_by_side.csv` and `report.md`: the main three-row comparison.
- `token_summary.csv`: all streams, temporal q0, depth, upper/hands, lower and face.
- `token_by_speaker.csv` and `token_by_recording_codebook.csv`: detailed token metrics.
- `token_intervention_coverage.csv`: scored positions, positions with changed
  future speech, and the summed remaining future lengths for each recording.
- `paired_token_differences.csv`: matched minus causal, and mismatched minus matched,
  with paired recording-bootstrap confidence intervals.
- `generation_metrics.csv`: dataset-level metrics for every condition/seed.
- `generation_by_recording.csv`: per-recording expression MSE and frame counts.
- `donor_manifest.csv` and `recordings.csv`: exact evaluated population and donor segments.
- `generated_tokens/<seed>/<recording>.npz`: codes for all three conditions.
- With `--save-motion`, `generated_motion/` stores generated poses, expressions,
  translations and references for subsequent rendering. It does not render videos.
- `metadata.json`: checkpoint/config hashes, speaker map, complete settings,
  versions, timing, and run status. Token totals and generated codes are written
  progressively; automatic resume is not implemented. Use a new output directory
  for each run.

Lower matched-future CE and higher accuracy relative to both controls support
content-specific use of correct future speech. Matched and mismatched futures
have identical lengths, so that contrast controls for the amount of available
future context. Similar matched/mismatched scores would weaken the claim that
the model benefits specifically from correct future content. None of these
outcomes is assumed by the script, and token improvements need not translate
into improvements on every generated-motion metric.

Only the causal condition is the deployable causal view. The two future views
are evaluation controls with privileged speech, using the same frozen weights.

## Local verification

```bash
python -B -m scripts.smoke_test_future_speech_comparison
```

The checks use tiny synthetic networks/data to verify causal/teacher equivalence,
no correct-future leakage in the mismatch view, identity-donor controls, common
sampling, donor selection, CE arithmetic, paired bootstrap, strict checkpoint
loading and full runner exports. They do not reproduce remote dataset scores.
