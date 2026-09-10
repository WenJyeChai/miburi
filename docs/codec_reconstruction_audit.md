# Current codec reconstruction notebook

Open `scripts/codec_reconstruction_audit.ipynb` with the Python environment used
for MIBURI training. It evaluates the codecs referenced by the selected gesture
LM config, without constructing the gesture transformer or speech models.

The default config is
`configs/gtdm3_teacher_c_face1_cos800_rvq_beatx_scott.yaml`. Its current checkpoints
are upper `demoexp_release_uppercodec/last_180.safetensors`, lower
`demoexp_release_lowercodec/last_440.safetensors`, and face
`demoexp_release_facecodec/last_100.safetensors`, all under `experiments/`.
Checkpoint paths and architecture settings come from the LM config; generic
codec YAML defaults do not override the production model factory.

## Run

1. In the first code cell, set `REPO_ROOT` if automatic detection cannot find the
   repository. Select the GPU with `CUDA_VISIBLE_DEVICES` before CUDA starts;
   `DEVICE="cuda:0"` means the first GPU in that visible list.
2. Set `PATH_OVERRIDES` if the HDF5 or checkpoint files live elsewhere. The
   default data selector is inherited from the LM config, currently
   `scott_beatx_lowervalid`. Use the same cache as the teacher experiment.
3. Choose `PARTS=("face",)` for a focused pass, or retain all three codecs.
   `MAX_SAMPLES_PER_SPLIT=8` gives a setup check; `None` evaluates every selected
   sample. The default splits are train and validation. Add test explicitly
   when needed.
4. Run the cells in order. Inspect the preflight paths and the result manifest,
   then the train/validation table, codebook distributions and difficult clips.
5. Run the export cell. It creates a fresh timestamped directory under
   `reports/codec_reconstruction/` with CSV tables, a JSON manifest and PNG/SVG
   figures.

The notebook needs Jupyter/IPython plus the existing training dependencies. It
does not install dependencies or download assets. Basic parameter-space metrics
need only the codec checkpoints and motion HDF5. `GEOMETRY=True` additionally
uses the existing SMPL-X asset under `assets_dep/smplx_2020/`; set it to `False`
to run the parameter and code-usage diagnostics alone. CPU is supported but
full streaming evaluation is intended for the training GPU environment.

## Measurements

- Geodesic rotation errors, including the face codec's reconstructed jaw.
- Expression RMSE, translation and velocity errors; translation reconstruction
  follows the production integration convention.
- Foot-contact RMSE over all four channels, using unrounded decoder outputs.
- Optional SMPL-X joint/vertex errors with explicit spatial and velocity units.
- Per-clip errors and pooled train/validation summaries.
- Per-stage code counts, active fraction, empirical entropy/perplexity, and
  held-out token mass absent from the audited training set.

The summary aggregates numerators and counts before taking square roots. It
does not average per-clip RMSE values. Trajectory and vertex velocity metrics
use two adjacent valid frames within one clip. The direct decoded-velocity
metric uses the production mixed finite-difference target. Changing the warmup
exclusion applies the same rule to all splits. A larger training set naturally visits more codes, so
codebook coverage should be interpreted alongside sample/token counts.

`MODE="streaming"` is the default production encode/decode path, resetting
streaming state for each clip. The alternative full-clip mode is separately
labelled and should not be silently mixed with streaming results. Quantization
is canonical and frozen; codebook usage is counted from eval-mode encodings,
not by enabling training-time EMA updates or RVQ augmentation.

## Facial metric definitions

The notebook computes direct errors at the native motion frame rate. It does
not call the old `ReconMetrics` aggregate, which overwrites the reconstructed
jaw with the target and computes facial velocity using a mixed predicted/target
frame difference. Here the predicted jaw is retained, and velocity is
`fps * (prediction[t] - prediction[t-1])` compared with the corresponding
target difference. Existing training/evaluation code is unchanged.

Geometry names containing `face_deformation_all_vertices` measure the whole
SMPL-X vertex array under face-only articulation, with the body pose neutral.
They are not a lip-only or face-region score; unchanged body vertices dilute
localized facial error. Rotation/expression metrics remain useful without an
extra face-region vertex-index asset. FGD and speech synchronization are outside
this reconstruction audit.

## Interpretation and reproducibility

The manifest records exact checkpoint hashes, selected sample IDs and settings,
plus duplicate IDs and overlapping source time intervals across selected splits.
Non-overlapping chunks from the same recording are not interval overlaps; use
the manifest's source file IDs to check recording-level separation.
Evaluation does not prove which
historical recordings the released codec was trained on. Checkpoint and codebook
state fingerprints are checked for changes during the audit.

Good reconstruction on Scott establishes that the codec can represent those
motions; it does not establish that speech predicts the tokens well. Poor
reconstruction on both train and validation supports checking the codec or
preprocessing. A large reconstruction gap instead points toward generalization
or split composition. If a new codec is trained, compare decoded reconstruction
and generated motion rather than treating transformer CE under different
tokenizers as directly comparable.

No dataset reconstruction scores are bundled with this notebook. Synthetic
checks can verify the implementation, but actual Scott scores must be produced
on the machine with the HDF5 cache.

## Validation

The seven CPU smoke checks passed, including strict loading and repeated frozen
streaming reconstruction with all three released checkpoints:

```bash
python -B -m scripts.smoke_test_codec_reconstruction_audit --release-codecs
```

All nine notebook code cells were also executed headlessly with a synthetic
HDF5 cache, the released checkpoints and the actual SMPL-X asset. Reconstruction,
code-usage tables, state checks and report exports passed. This verifies the
workflow; it does not measure Scott reconstruction quality. The notebook is
saved with empty outputs so synthetic results cannot be mistaken for real data.
