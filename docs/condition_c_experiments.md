# Condition C: future-only speech privilege

At each temporal query, the condition-C teacher sees the same direct rolling
past-speech window as the causal student, plus every current/future speech token
in the supplied sequence. With the released settings this is a 50-token
past/current speech window at 12.5 Hz, approximately four seconds. Each temporal
cross-attention layer retains the student's original strict lower boundary;
only the upper (future) boundary is removed. Gesture self-attention remains
causal. Earlier information can still be carried indirectly in gesture history.

The two configurations below deliberately use the same data, seed, model sizes,
stochastic RVQ settings, face weighting, VAD/contrastive auxiliaries, modality
dropout, and learning-rate schedule. RVQ uses canonical temporal inputs and q0
targets, with stochastic depth prefixes and the existing hard/soft depth loss.

| Run | Main gradient-enabled forward | Additional forward | Supervision |
| --- | --- | --- | --- |
| ForeMotion C | Causal student | Detached C teacher using the same parameters | Student token/RVQ loss + auxiliaries + forward KL |
| Supervised teacher C | C teacher | None | Teacher token/RVQ loss + auxiliaries |

ForeMotion's teacher has no direct loss gradient but shares all updated weights
with the student. The teacher-only trainer does not inherit the regret trainer:
there is no student pass, KL loss, or EMA update. Its temporal and depth
parameters are trained through the C forward. Both teacher views use the same
mask implementation. Existing GlobalRegret, word/sentence, and unrestricted
offline experiments retain their original policies.

## Train

Run each command from the repository root in the existing Linux/CUDA training
environment. Both commands select GPU 2; run them sequentially, or change one
GPU selection when running concurrently. Output project names and W&B run names
differ so the two experiments can be identified independently.

ForeMotion under C:

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_INIT_TIMEOUT=600 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_foremotion_c_rvq_beatx_scott.yaml \
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
  --test_period 20 \
  --lr_policy cosine_delay \
  --lr_cosine_start_epoch 200 \
  --wandb True \
  --wandb_project miburi_single \
  --wandb_group future-only-c-scott \
  --wandb_name foremotion-c-rvq-scott-bs64 \
  --wandb_tags foremotion condition-c future-only shared-teacher scott rvq bs64
```

Directly supervised teacher under C:

```bash
CUDA_VISIBLE_DEVICES=2 \
WANDB_INIT_TIMEOUT=600 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_teacher_c_rvq_beatx_scott.yaml \
  --batch_size 64 \
  --loader_workers 8 \
  --pretrain_warmup_epochs 0 \
  --regret_weight 0 \
  --regret_initial_weight 0 \
  --kinematic_rvq_start_epoch 0 \
  --kinematic_rvq_ramp_epochs 1 \
  --dense_future_gesture_weight 0 \
  --test_period 20 \
  --lr_policy cosine_delay \
  --lr_cosine_start_epoch 200 \
  --wandb True \
  --wandb_project miburi_single \
  --wandb_group future-only-c-scott \
  --wandb_name teacher-c-supervised-rvq-scott-bs64 \
  --wandb_tags teacher-only condition-c future-only supervised-teacher scott rvq bs64
```

The ForeMotion KL weight is 0.6 at epoch 0 and reaches 6 at epoch 1. Both runs
ramp RVQ augmentation over the first epoch. Teacher-only regret weights are
explicitly zero because that trainer has no regret objective.

## Interpret and evaluate

Training retains `pose_length: 250` at 25 fps: the privileged future is the
remainder of each supplied ten-second training clip. This change does not join
clips or give either teacher the rest of a source utterance outside its clip.
For complete-utterance testing, use the full-sequence HDF5 described in the
repository README and provide the complete speech sequence to generation.
The C generator caches that complete memory and applies the rolling lower
boundary at each generated gesture step.

Compare these three inference views on the same held-out sequences, checkpoint
epochs, sampling settings, and seeds:

1. ForeMotion checkpoint using its causal student generator.
2. The same ForeMotion checkpoint using the condition-C teacher generator.
3. The directly supervised teacher checkpoint using the condition-C generator.

For views 2 and 3, use `scripts/test.py` with
`--config configs/gtdm3_teacher_c_rvq_beatx_scott.yaml` and the desired
`--test_ckpt`, plus the evaluation-cache and dataset-selection overrides.
The C model adds no trainable parameters or checkpoint keys, so view 2 loads
the shared ForeMotion weights through the teacher configuration. Using
ForeMotion's own configuration for testing selects its causal student.
The existing test script writes results beside its checkpoint. When evaluating
the same checkpoint in two views, place identical checkpoint copies in separate
view directories to retain both sets of saved results.

Training/validation CE is teacher-forced token prediction; it is not a
free-running motion-quality measurement. `--test_period 20` schedules checkpoint
saving and validation in the existing training script. Generated-motion
evaluation still runs separately through `scripts/test.py`.

## Implementation and checks

- `miburi/models/gesture_lm_condition_c.py`: common C attention mask, detached
  shared teacher view, and gradient-enabled teacher model.
- `scripts/trainers/uflgtdm3_condition_c_trainer.py`: separate experiment trainers.
- `StochasticRVQTrainingMixin`: reused RVQ objective without importing regret
  behavior into teacher-only training.
- `scripts/smoke_test_gesture_lm_condition_c.py`: attention/gradient/streaming checks.
- `scripts/smoke_test_condition_c_trainers.py`: trainer dispatch, objective, and
  gradient-routing checks.

Run the smoke checks in the repository's Python environment:

```bash
python -m scripts.smoke_test_gesture_lm_condition_c
python -m scripts.smoke_test_condition_c_trainers
```

Validation for this implementation: 12 condition-C model checks, 8 trainer
checks, and 22 existing SharedRegret/offline/RVQ/linguistic checks passed on CPU.
Both documented training commands also passed argument parsing and trainer
dispatch checks. Training and full-utterance CUDA benchmarks were not launched;
the training HDF5 data is not present in this workspace.

The new checks establish temporal batch/streaming parity and identical complete
generation to the existing offline generator when the speech masks coincide.
They also exposed an existing depth-generator difference from batched
teacher-forced depth predictions, reproduced in the unrestricted offline
baseline. That shared depth behavior is unchanged here and should be investigated
before using token-prediction accuracy to explain generated-motion performance.
