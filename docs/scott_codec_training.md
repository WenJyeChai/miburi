# Train the current codecs from scratch on Scott

`scripts/train_scott_codecs.py` launches three independent runs using the
existing causal Mimi/RVQ codec trainers. The model architecture and feature
layout match the deployed codecs:

| Part | Input per frame | RVQ | Encoder / decoder transformer |
| --- | --- | --- | --- |
| Upper | 13 upper-body + 30 hand joints in 6D: 258 values | 8 stages, 2048 entries each | 8 layers, 4 heads each |
| Lower | 9 joints in 6D + root velocity + 4 contacts: 61 values | 8 stages, 2048 entries each | 8 layers, 4 heads each |
| Face | Jaw in 6D + 100 expression coefficients: 106 values | 4 stages, 2048 entries each | 4 layers, 2 heads each |

All three retain latent width 256, code vector width 128, two convolutional
blocks and two-frame chunks at 25 FPS (12.5 code steps/second). The launcher
uses fresh model initialization and fresh RVQ state. RVQ initialization and
EMA updates use training batches; no released codec checkpoint is loaded.

The data selector is `scott_beatx_lowervalid`, using the same
`datasets/data_cache/beatx_gtdm3/database.hdf5` as the Scott gesture experiments.
The audited cache had 477 training clips and 54 validation clips; the 89 test
clips remain for final evaluation. Counts follow the supplied cache rather
than being hard-coded. Training uses only the training split. Standard codec
training does not construct a test loader. Audio and text are not model inputs.

## Run

Run in the repository on the CUDA training machine, using its existing MIBURI
environment. This Bash command trains upper, lower and face sequentially on
physical GPU 2, with separate output directories and W&B runs:

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_INIT_TIMEOUT=600 \
OMP_NUM_THREADS=4 \
python scripts/train_scott_codecs.py \
  --part all \
  --batch_size 32 \
  --loader_workers 8 \
  --epochs 2000 \
  --test_period 10 \
  --wandb True \
  --wandb_project miburi_single \
  --wandb_group codec-scott-scratch \
  --wandb_name codec-scott-scratch \
  --wandb_tags codec scott scratch rvq bs32
```

Change `--part all` to `--part upper`, `--part lower` or `--part face` to run
just one. For parallel runs, use those three commands in separate terminals
with different `CUDA_VISIBLE_DEVICES` values. This launcher uses plain Python
and a single GPU per run; do not wrap it in `torchrun`.

Append `--dry_run` to inspect commands without creating a training run or
requiring CUDA/data assets. Omit `--wandb True` for local logging only.
`--wandb_mode offline` retains offline W&B logging when W&B is enabled.
`--beatx_cache_path /path/to/database.hdf5`, `--deps_path /path/to/assets_dep/`
and `--out_path /path/to/experiments/` override local paths. Relative paths
resolve from the repository. SMPL-X is required by the existing training losses:
`<deps_path>/smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz`.

Each invocation gets a unique suffix to keep previous experiments intact.
The launcher stops if a part fails, and does not start the remaining parts.
It deliberately does not accept resume/checkpoint or architecture overrides.
To inspect the complete recipes, see:

- `configs/mimi_scott_scratch_upper.yaml`
- `configs/mimi_scott_scratch_lower.yaml`
- `configs/mimi_scott_scratch_face.yaml`

## Training recipe

The initial loss weights and optimizer schedule follow the saved released
experiment configs, including face reconstruction weight 3 and lower contact
weight 1. AdamW starts at `1e-4` with betas `(0.9, 0.95)`, zero weight decay
and gradient clipping at 1. Step decay occurs every 50 epochs: factor 0.9 for
upper, and 0.75 for lower and face. Commitment weight ramps over 25 epochs to
0.25. The defaults retain 2000 training epochs; `--epochs` overrides this cap.
These are starting recipes for the comparison, not optimized Scott schedules.

Parameters, optimizer and latents use FP32. This preserves the deployed model
architecture and inference dtype; the older saved training recipes used BF16
latents. Expect different memory/runtime costs from those historical runs.
Training retains variable-length crops and the original RVQ augmentation.
Validation always reconstructs full clips without random cropping, RVQ dropout
or quantization bypass. All validation frames are scored, including finite
frames marked `pose_valid=False`; no warmup is discarded. Nonfinite inputs or
changed sample identities stop validation rather than silently replacing clips.

## Reconstruction validation and checkpoints

Every 10 epochs and at the final epoch, each codec reconstructs validation
clips through its production two-frame streaming encode/decode path. Metrics
pool error sums and element counts across all clips before computing averages
or square roots. Parameters and RVQ EMA state do not update during evaluation.

MSE, MAE and RMSE are reported separately for rotation 6D, expression
coefficients, root velocity and contacts as applicable, without averaging
different feature units together. PSNR uses a fixed range of
2 for rotation 6D and 1 for contacts. Expressions and root velocity have no
fixed range, so they do not receive an arbitrary PSNR value. These feature
metrics are not pixel/image metrics and are not in millimetres.

`best_reconstruction.safetensors` is selected using validation rotation-6D MSE
for upper/lower and expression MSE for face. Lower checkpoint selection
therefore does not directly optimize trajectory or contact error; inspect
those separately in the final codec audit. `best_reconstruction.json` records
the selected epoch, metric and value. Periodic `last_<epoch>.safetensors`
checkpoints are also retained. Checkpoints are weights, not full optimizer and
RNG snapshots; this launcher implements fresh training, not exact resume.

The opt-in `codec_standard_eval` flag leaves the legacy validation path
available to other experiments. Standard reconstruction validation replaces
the legacy validation-loss pass; training-loss curves still use the original
loss recipe, so they are not directly comparable to raw validation MSE.

W&B and TensorBoard use `val/codec_standard/rotation6d_mse`,
`val/codec_standard/expression_mse` and the other applicable metric names.
`val/codec_standard/best_reconstruction_epoch` identifies the selected epoch.

Optional `--codec_eval_fgd True` additionally uses the existing BEATX evaluator
at `<beatx_data_path>/weights/AESKConv_240_100.bin`. It resamples rotations to
30 FPS and trims each clip to a multiple of 32 frames for the evaluator. This
is **part-isolated reconstruction FGD**: the evaluated part is reconstructed
and other body parts come from GT. The jaw reconstruction is retained for face.
The evaluator sees rotations, not expression coefficients or root translation,
so it is not a complete facial or trajectory metric. Missing evaluator assets
or failed FGD computation produce an unavailable status and warning; standard
reconstruction validation continues. FGD does not select the best checkpoint.

## Compare with the current codecs

After training, use `scripts/codec_reconstruction_audit.ipynb` with the same
sample manifest, parts, streaming mode and metric scope used for the baseline.
Keep the existing gesture-model config for architecture construction and set:

```python
PATH_OVERRIDES = {
    "upperbodycodec_ckpt": "experiments/<upper-run>/best_reconstruction.safetensors",
    "lowerbodycodec_ckpt": "experiments/<lower-run>/best_reconstruction.safetensors",
    "facecodec_ckpt": "experiments/<face-run>/best_reconstruction.safetensors",
}
```

Use validation to select checkpoints; evaluate test after selection. The audit
adds geodesic rotation error, integrated root-trajectory error, optional SMPL-X
geometry errors and RVQ usage diagnostics. See
[`codec_reconstruction_audit.md`](codec_reconstruction_audit.md).

Codec checkpoint compatibility does not make code indices interchangeable:
new RVQ codebooks change token meanings. Any later gesture-model experiment
using the new codecs must re-encode motion targets and train the gesture model
against those new tokens.

## Local checks

```bash
python -B -m scripts.smoke_test_scott_codec_training
```

The checks use temporary synthetic data and inspect the released checkpoint
shapes when available. They do not start a GPU experiment or contact W&B.
