# Condition C: face loss and shorter cosine decay

These experiments test reducing `face_loss_weight` from 2.5 to 1.0 and
shortening the cosine **learning-rate** schedule from an endpoint of 10,000
to 800. AdamW `weight_decay` remains 0.0; it is a separate optimizer setting.
The existing scheduler and face-loss implementation already support these
settings, so this change adds experiment configurations without changing
training-loop or model behavior.

Use the original codebook-specific depth weights
(`gestureformer_depformer_weights_per_step: True`) as the architecture baseline:
they achieved better validation CE than shared depth weights in the September 9
comparison. Both ForeMotion C and the directly supervised teacher C have the
same three ablations. Their existing gradient routing, RVQ augmentation,
VAD/contrastive auxiliaries, seed, data and attention masks are retained.

## Experiment matrix

| Variant | Face CE weight | Cosine start | Cosine end | Training budget |
| --- | ---: | ---: | ---: | ---: |
| Existing original baseline | 2.5 | 200 | 10000 | Existing history |
| `face1` | 1.0 | 200 | 10000 | 1000 epochs |
| `cos800` | 2.5 | 200 | 800 | 1000 epochs |
| `face1_cos800` | 1.0 | 200 | 800 | 1000 epochs |

All new configurations use `lr_base: 1e-4`, `lr_min: 1e-6`,
`warmup_epochs: 0` and `pretrain_warmup_epochs: 0`. They start fresh
(`is_continue: False`) and have distinct output project names.

The face-only control explicitly retains `lr_cosine_end_epoch: 10000`.
Leaving the endpoint at its default of -1 would resolve it to the new
1000-epoch budget, accidentally changing the learning rate in that control.
Existing baseline and shared-depth configurations are unchanged.

The combined runs below test both changes together. Use the individual
controls afterward, or alongside them if resources permit, to attribute an
improvement to face weighting, the schedule, or their interaction.

## Learning-rate schedule

The shorter schedule holds at 1e-4 through scheduler epoch 200, follows a
cosine down to 1e-6 at 800, and stays at that minimum afterward. This gives
approximately 200 further training epochs at the floor. The original schedule
was still near its initial learning rate when validation CE deteriorated.

| Scheduler epoch | Original schedule | Shorter schedule |
| ---: | ---: | ---: |
| 200 | 1.00000e-4 | 1.00000e-4 |
| 400 | 9.98983e-5 | 7.52500e-5 |
| 500 | 9.97713e-5 | 5.05000e-5 |
| 600 | 9.95936e-5 | 2.57500e-5 |
| 800 | 9.90872e-5 | 1.00000e-6 |
| 1000 | 9.83811e-5 | 1.00000e-6 |

The existing fresh-run training loop advances the scheduler after each epoch.
Consequently, optimization at epoch label e > 0 uses the schedule for e - 1:
the first decrease is used at 202 and the exact floor from 801. With
`epochs: 1000`, training uses labels 0 through 999 and label 1000 performs
final checkpoint saving and validation. These existing conventions are
preserved for the comparison. `decay_epochs` does not control `cosine_delay`.

This is a targeted schedule to test, not an established optimum or a guarantee
against overfitting. Keep the best validation checkpoint as well as the final
checkpoint when evaluating the experiment.

## Combined runs

Run from the repository root in the Linux/CUDA training environment. Both
commands use GPU 2; run sequentially or change one GPU selection for concurrent
runs. The YAML files contain the settings; key flags are repeated here to make
the commands easy to inspect alongside previous experiments.

ForeMotion C: causal student supervision plus the detached shared teacher's KL.

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_INIT_TIMEOUT=600 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_foremotion_c_face1_cos800_rvq_beatx_scott.yaml \
  --gestureformer_depformer_weights_per_step True \
  --batch_size 64 \
  --loader_workers 8 \
  --pretrain_warmup_epochs 0 \
  --regret_start_epoch 0 \
  --regret_ramp_epochs 1 \
  --regret_weight 6 \
  --regret_initial_weight 0.6 \
  --kinematic_rvq_start_epoch 0 \
  --kinematic_rvq_ramp_epochs 1 \
  --dense_future_gesture_weight 0 \
  --face_loss_weight 1.0 \
  --test_period 20 \
  --epochs 1000 \
  --lr_policy cosine_delay \
  --lr_base 1e-4 \
  --lr_min 1e-6 \
  --warmup_epochs 0 \
  --lr_cosine_start_epoch 200 \
  --lr_cosine_end_epoch 800 \
  --weight_decay 0 \
  --wandb True \
  --wandb_project miburi_single \
  --wandb_group future-only-c-face-cosine-scott \
  --wandb_name foremotion-c-face1-cos800-rvq-scott-bs64 \
  --wandb_tags foremotion condition-c future-only shared-teacher per-step-depth face1 cosine800 scott rvq bs64
```

Directly supervised teacher C: teacher loss receives gradients, with no
student forward or regret loss.

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_INIT_TIMEOUT=600 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_teacher_c_face1_cos800_rvq_beatx_scott.yaml \
  --gestureformer_depformer_weights_per_step True \
  --batch_size 64 \
  --loader_workers 8 \
  --pretrain_warmup_epochs 0 \
  --regret_weight 0 \
  --regret_initial_weight 0 \
  --kinematic_rvq_start_epoch 0 \
  --kinematic_rvq_ramp_epochs 1 \
  --dense_future_gesture_weight 0 \
  --face_loss_weight 1.0 \
  --test_period 20 \
  --epochs 1000 \
  --lr_policy cosine_delay \
  --lr_base 1e-4 \
  --lr_min 1e-6 \
  --warmup_epochs 0 \
  --lr_cosine_start_epoch 200 \
  --lr_cosine_end_epoch 800 \
  --weight_decay 0 \
  --wandb True \
  --wandb_project miburi_single \
  --wandb_group future-only-c-face-cosine-scott \
  --wandb_name teacher-c-face1-cos800-supervised-rvq-scott-bs64 \
  --wandb_tags teacher-only condition-c future-only supervised-teacher per-step-depth face1 cosine800 scott rvq bs64
```

## Individual controls

These commands load all training settings from their respective YAMLs,
including batch size 64, loader workers 8 and the appropriate regret weights.
Do not copy the combined run's face or cosine overrides into these controls.

ForeMotion C, face weight only:

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_INIT_TIMEOUT=600 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_foremotion_c_face1_rvq_beatx_scott.yaml \
  --wandb True \
  --wandb_project miburi_single \
  --wandb_group future-only-c-face-cosine-scott \
  --wandb_name foremotion-c-face1-rvq-scott-bs64 \
  --wandb_tags foremotion condition-c future-only shared-teacher per-step-depth face1 cosine10000 scott rvq bs64
```

ForeMotion C, shorter cosine only:

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_INIT_TIMEOUT=600 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_foremotion_c_cos800_rvq_beatx_scott.yaml \
  --wandb True \
  --wandb_project miburi_single \
  --wandb_group future-only-c-face-cosine-scott \
  --wandb_name foremotion-c-cos800-rvq-scott-bs64 \
  --wandb_tags foremotion condition-c future-only shared-teacher per-step-depth face2.5 cosine800 scott rvq bs64
```

Supervised teacher C, face weight only:

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_INIT_TIMEOUT=600 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_teacher_c_face1_rvq_beatx_scott.yaml \
  --wandb True \
  --wandb_project miburi_single \
  --wandb_group future-only-c-face-cosine-scott \
  --wandb_name teacher-c-face1-supervised-rvq-scott-bs64 \
  --wandb_tags teacher-only condition-c future-only supervised-teacher per-step-depth face1 cosine10000 scott rvq bs64
```

Supervised teacher C, shorter cosine only:

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_INIT_TIMEOUT=600 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_teacher_c_cos800_rvq_beatx_scott.yaml \
  --wandb True \
  --wandb_project miburi_single \
  --wandb_group future-only-c-face-cosine-scott \
  --wandb_name teacher-c-cos800-supervised-rvq-scott-bs64 \
  --wandb_tags teacher-only condition-c future-only supervised-teacher per-step-depth face2.5 cosine800 scott rvq bs64
```

## Compare the results

Use unweighted `val/ce_loss`, `val/ce_face`, `val/ce_upper` and `val/ce_lower`,
plus `val/temporal_q0_ce` and its accuracy. Compare each role with its own
original baseline at matched epoch labels through 980 and at the best validation
checkpoint within that common horizon. Report the new final checkpoint at 1000
separately: it follows training through 999, whereas the historical baseline's
epoch 1000 also included that epoch's training. The original W&B runs are
`0st12z6a` (ForeMotion C) and `8hir7gzc` (supervised teacher C).

For depth specifically, use the unweighted sum
`val/kinematic_upper_hard_ce + val/kinematic_lower_hard_ce + val/kinematic_face_hard_ce`.
These contributions use division by 20; multiply their sum by 20/19 to obtain
mean CE across the 19 depth heads. The upper contribution excludes q0.

Do not treat a lower weighted `val/kinematic_hard_ce` or training total as
evidence of improvement across face weights: reducing the face coefficient
mechanically lowers those totals even for identical predictions. Unweighted
per-part metrics retain the same definitions. Body contributions contain eight
heads each and face four; for mean CE per head, multiply `val/ce_upper` and
`val/ce_lower` by 20/8 and `val/ce_face` by 20/4.

The face change scales its hard/soft token objective. VAD still consumes face
logits and propagates through them; this does not test detaching that path.
Auxiliary and KL coefficients remain fixed, so reducing face CE also changes
their relative contribution to the total training objective.

Select the `last_<epoch>.safetensors` corresponding to the best validation
epoch; the current trainer tracks best metrics but does not write a dedicated
best checkpoint. `test_period: 20` controls checkpointing and validation here.
Complete-utterance generated-motion evaluation is separate; see the
[condition-C evaluation guide](condition_c_experiments.md#interpret-and-evaluate).

## Local verification

All six configurations and all six documented commands passed the actual
argument parser. Effective settings differ from their original baselines only
in experiment identifiers, training budget, explicit cosine endpoint and the
intended face weight. Six 1000-epoch simulations using the real PyTorch
optimizer and repository scheduler verified the flat start, cosine values and
clamping at the floor. Synthetic checks through both C trainers' RVQ objectives
verified the face-gradient scaling and preserved body/q0 CE gradients, padding
behavior and unweighted part diagnostics. These checks do not measure
convergence or motion quality; full dataset/CUDA training was not launched.
