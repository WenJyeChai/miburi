# Multispeaker training-component ablation

This is a matched 2 x 2 version of the historical Scott component study:
stochastic RVQ token regularization and shared-weight GlobalRegret distillation.
It deliberately preserves GlobalRegret's unrestricted full-clip speech teacher
view, rather than substituting the later condition-C rolling-past policy.
Gesture history stays causal; future-gesture distillation is disabled.

| Variant | Token regularization | Speech distillation | Config |
| --- | --- | --- | --- |
| baseline | Off | Off | `configs/gtdm3_ablation_baseline_beatx_allspk.yaml` |
| distill | Off | On | `configs/gtdm3_ablation_distill_beatx_allspk.yaml` |
| tokenreg | On | Off | `configs/gtdm3_ablation_tokenreg_beatx_allspk.yaml` |
| full | On | On | `configs/gtdm3_ablation_full_beatx_allspk.yaml` |

The baseline uses the original trainer. Distillation-only uses SharedRegret
without the RVQ mixin. Token-only and full use SharedRegretRVQ; token-only
sets both regret weights to zero, skipping the teacher/KL branch. All four
use the same causal model architecture and independent per-step depth weights.
No new trainer implementation is required.

## Shared protocol

- `dataset_ratio: full_beatx_lowervalid`: all BEATX speakers remaining after
  the existing lower-valid filter, using the cache's train/validation/test
  splits. This is a seen-speaker multispeaker study, not a held-out-speaker
  generalization protocol. A cache built with a Scott/good-speaker filter
  must be replaced with an all-speaker cache; changing the ratio cannot
  restore speakers omitted at cache-build time.
- The cache path is `datasets/data_cache/beatx_gtdm3/database.hdf5` in all
  configs. Override `--beatx_cache_path` identically in all commands if the
  all-speaker cache is elsewhere. The existing dataset supplies speaker IDs;
  the model already has temporal and depth speaker embeddings.
- All-speaker codec paths follow the existing all-speaker baseline:
  upper `experiments/allspk_release_uppercodec/last_490.safetensors`, lower
  `experiments/allspk_release_lowertranscodec/last_705.safetensors`, and face
  `experiments/allspk_release_facecodec/last_380.safetensors`. These must exist
  on the training host. If using another codec set, override all three paths
  identically across the four runs. Do not compare CE across different codecs.
- Fresh runs, seed 2342, 250 motion frames at 25 fps, BF16 parameters,
  batch 64 per GPU. Two GPUs give a global batch of 128. Keep device count
  and per-GPU batch fixed across the matrix.
- Historical full-component optimization recipe: AdamW, LR 1e-4,
  weight decay 0, StepLR factor 0.8 every 2000 epochs, 10000 training epochs.
  This is a reproducibility starting point, not an optimized multispeaker
  budget. Any budget/schedule changes must be applied identically to all runs.
- Face weight 2.5, VAD with face logits, contrastive/body/memory settings,
  checkpoint/validation interval 20, and model dimensions are identical.
- With distillation: temperature 1, weight 0.6 at epoch 0, reaching 6 at
  epoch 1, all 20 heads. Without it: both initial and maximum weights are 0.
- With token regularization: top-k 8, temperature 0.5, sampling probability
  0.15, soft-target weight 0.5, starting at epoch 0 with a one-epoch ramp.
  Without it: both sampling probability and soft-target weight are 0.
  q0 supervision stays canonical hard CE in every run. Temporal and depth
  paths remain trainable; this is not the frozen-temporal training phase.

The historical Scott runs used different warmups, strengths and batch setups.
This matrix holds those settings fixed to isolate the two components within
the multispeaker setting; it is not an exact replay of every old run.

## Training commands

Run from the repository root in the existing Linux/CUDA training environment
with the data, codec checkpoints and dependencies installed. Authenticate W&B
before starting. These are four separate fresh runs; execute sequentially on
the same two GPUs, or use disjoint GPU pairs for concurrent runs.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export WANDB_INIT_TIMEOUT=600

# Neither component.
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/gtdm3_ablation_baseline_beatx_allspk.yaml --wandb True

# Distillation only.
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/gtdm3_ablation_distill_beatx_allspk.yaml --wandb True

# Token regularization only.
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/gtdm3_ablation_tokenreg_beatx_allspk.yaml --wandb True

# Both components.
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/gtdm3_ablation_full_beatx_allspk.yaml --wandb True
```

W&B project: `miburi_multi`; group: `allspk-training-components-global`.
Each variant has a distinct local project name and W&B run name. Torchrun
enables DDP automatically in `scripts/train.py`. For a single GPU, use
`CUDA_VISIBLE_DEVICES=0 python scripts/train.py ...` and override the W&B
name to reflect the actual setup. Keep all four runs on that same setup.

## Evaluation and validation

Compare the unweighted canonical hard-label `val/ce_loss` at matched epochs
and at each run's validation-selected best checkpoint. Training CE differs
between token-reg and plain runs, so training CE is not the common ranking
metric. Run generated-motion evaluation separately with the same sampling
settings, seeds and held-out sequences. `test_period` schedules checkpoint
saving and validation, not autoregressive motion benchmarks.

Config checks can run without data. In the full training environment, also run:

```bash
python -m scripts.smoke_test_rvq_depth_regularization
python -m scripts.smoke_test_gesture_lm_shared_regret
python -m scripts.smoke_test_condition_c_trainers
```

The last suite also checks shared stochastic-RVQ objective behavior. Local
validation of this addition checked YAML syntax, accepted config keys,
trainer registration and the four-way configuration differences. The local
Windows Python environment lacks `loguru` and `configargparse`, so full CLI
parsing and the model smoke suites could not be completed there. Dataset/CUDA
training has not been launched.
