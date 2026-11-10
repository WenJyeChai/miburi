# ForeMotion: Condition C training

This source extraction trains the Scott single-speaker ForeMotion model with
the selected batch-64, no-warmup, cosine-schedule recipe. The training entry
point is `scripts/train.py`; the configuration is
`configs/gtdm3_foremotion_c_rvq_beatx_scott.yaml`.

The training loop, model implementations, losses, data processing and selected
YAML are copied from the existing project. There is no replacement training
algorithm. `SOURCE_MANIFEST.json` records the original and packaged SHA-256
hashes. Only the trainer and dataset import registries were narrowed to the
included implementations. Supporting shared modules retain their original
names and may contain other model variants; the YAML selects Condition C.

## Training setting

| Item | Value |
| --- | --- |
| Dataset selection | Scott, `scott_beatx_lowervalid` |
| Training clips | 250 motion frames at 25 fps; 125 gesture steps |
| Student direct speech attention | Up to 50 aligned past/current groups |
| Teacher direct speech attention | Same lower boundary, plus future groups through the clip end |
| Gesture self-attention | Causal in both views, direct window 25 steps |
| Teacher | Shared weights, detached predictions; no separate teacher checkpoint |
| Temporal / kinematic layers | 4 / 2, with 2 / 1 attention heads |
| Kinematic weights | Codebook-step-specific, as in the original configuration |
| Motion codecs | Frozen released codecs: upper/hands 8, lower 8, face 4 streams |
| Codebook size | 2,048 codes; token heads also include padding |
| Precision | BF16 model; FP32 optimizer state |
| Batch size / seed | 64 / 2342 |
| Optimizer | AdamW, learning rate 1e-4, betas (0.9, 0.95), weight decay 0 |
| Gradient clipping | Norm 1.0 |
| Speech / learning-rate warmup | Both 0 epochs |
| Training budget | 10,000 training epochs |
| Learning-rate schedule | Flat through epoch 200, cosine to 1e-6 at epoch 10,000 |
| Distillation | Weight 0.6 at epoch 0, 6.0 from epoch 1; temperature 1 |
| RVQ regularization | Ramps across the first epoch; top-k 8, temperature 0.5, maximum sampling probability 0.15 and soft-target weight 0.5 |
| Face token-loss multiplier | 2.5 |
| Other loss coefficients | VAD 0.05, contrastive 0.1; original activation schedules retained |
| Checkpoint / validation interval | 20 epochs; validation starts at epoch 20 |

Epochs are zero-based. The unchanged entry point trains epochs 0 through 9999,
then performs its final checkpoint/validation at epoch 10000. The cosine end
is inherited from `epochs` (`lr_cosine_end_epoch=-1`). Changing `--epochs` also
changes that endpoint unless explicitly overridden. This is not the later
1,000-epoch, face-weight-1, cosine-end-800 experiment.

Both predictive views receive the same gesture history and sampled within-step
RVQ prefix. Only the teacher's temporal speech cross-attention gains future
access; its kinematic speech slice stays aligned to the current step. Only the
causal student is needed for generation. These masks apply to cached aligned
speech tokens, not an end-to-end live transcription system.

## Environment

Run all commands from this extracted directory. The existing project targets
Python 3.12 and CUDA PyTorch 2.9.0. Use a CUDA-capable GPU supporting the selected
BF16 configuration. The following is the existing project's CUDA 12.6 setup:

```bash
conda create -n foremotion python=3.12 -y
conda activate foremotion
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-reviewer.txt
python -m pip install --no-deps -e .
```

Use a PyTorch CUDA build compatible with your driver. The source dependency
ranges are retained; this is not a fully locked environment. W&B is disabled
by default. To enable it, install `wandb` and pass `--wandb True` plus your own
project settings. No account credentials are included.

## Assets and training cache

Data and binary checkpoints are not bundled. The existing download helpers
are included:

```bash
miburi-download-assets
miburi-download-checkpoints
miburi-download-beatx-dataset
```

Keep the included source licenses and observe the respective upstream asset
licenses. The selected codec files must be available at:

```text
experiments/demoexp_release_uppercodec/last_180.safetensors
experiments/demoexp_release_lowercodec/last_440.safetensors
experiments/demoexp_release_facecodec/last_100.safetensors
assets_dep/smplx_2020/                      # SMPL-X NEUTRAL_2020 model
```

The trainer initializes speech embeddings by loading Moshi from
`kyutai/moshiko-pytorch-bf16`, then discarding the language model. The first
run may download those weights into `assets_dep/kyutai_cache/`; this initialization
also needs sufficient memory. The gesture generator starts without a resumed
generator checkpoint.

Use the original compatible training HDF5 if already available, at
`datasets/data_cache/beatx_gtdm3/database.hdf5`. It must contain motion, valid
region labels, VAD, aligned `moshi_audiotokens` and `moshi_texttokens`, and
train/development/test membership. A motion-only cache is insufficient.

To build a cache, first provide the official BEAT2 `train_test_split.csv` at
the path below. **The included dataset downloader downloads the data subfolders
but does not download this split CSV.** Do not generate a new random split.
The dataset directory must contain `smplxflame_25/`, `wave16k/` and
`whisper_transcription/`.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python -m scripts.trainers.dataloaders.beatx.build_hdf5_beatx \
  --data_dir datasets/beat_english_v2.0.0/ \
  --file_list_path datasets/beat_english_v2.0.0/train_test_split.csv \
  --hdf5_path datasets/data_cache/beatx_gtdm3/database.hdf5 \
  --motion_fps 25 \
  --pose_length 250 \
  --process_multimodal_signals
```

This retains the builder's existing defaults, including lower-valid filtering.
The training loader selects Scott from the resulting cache. Embody3D data is
not required for this recipe, despite the unused paths retained in the YAML.
Motion codes are produced by the frozen regional codecs during training.
The exact historical cache and split manifest remain the preferred inputs
for reproducing a particular reported run.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_foremotion_c_rvq_beatx_scott.yaml
```

Change `CUDA_VISIBLE_DEVICES` to select your GPU. The command uses the complete
selected recipe; no additional hyperparameter overrides are needed. Keep one
GPU for the specified batch size. With DDP, the configured batch size is per
process, which changes the effective batch size.

To reuse externally stored inputs, pass `--beatx_cache_path`,
`--upperbodycodec_ckpt`, `--lowerbodycodec_ckpt`, `--facecodec_ckpt`, and
`--deps_path`. Directory arguments such as `--deps_path` and `--out_path` must
retain a trailing slash because the original code concatenates these paths.

Outputs are written under `experiments/<timestamp>_<project_name>/`, including
configuration, logs, TensorBoard events, a speaker map, and
`last_<epoch>.safetensors`. Checkpoints from this entry point contain model
weights, not optimizer/scheduler states; continuation is not an exact optimizer
resume. Periodic validation measures token-prediction losses; it does not run
full-utterance generation metrics such as FGD. This extraction covers training.

## Checks without data or pretrained weights

```bash
python -B scripts/train.py --config configs/gtdm3_foremotion_c_rvq_beatx_scott.yaml --help
python -B -m scripts.smoke_test_gesture_lm_condition_c
python -B -m scripts.smoke_test_condition_c_trainers
python -B -m scripts.smoke_test_rvq_depth_regularization
```

The existing checks cover the speech boundary, causal gesture attention,
shared teacher detachment, student backward pass, trainer routing, and RVQ
prefix/target behavior. They use small synthetic models and do not establish
full-dataset convergence or reproduce paper scores.
