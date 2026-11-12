# Past-as-future training control

This tests whether informative future speech helps the causal student beyond
token regularization and distillation from repeated historical content. It is
a new **training** experiment, separate from `future_speech_comparison.py`'s
fixed-checkpoint intervention. Both use the existing shared-weight causal GTDM3
architecture. No extra teacher weights or new data cache are required.

## Exact setting

At target group `t` in a clip of `L` groups, preserve the teacher's true speech
prefix `0..t`. Let `F = L-t-1` be the number of future groups. Replace future
slots `t+1..L-1` with the chronological historical segment `t+1-F..t`, ending
with the **current group, inclusive**. Audio and text move together; all speech
codebooks and within-group substeps stay aligned. The memory length and speaker
are unchanged. No speech after `t` supplies the replacement, no wrapping or
padding is added, and no historical segment is tiled to fill a longer future.

For example, for six groups at target 2:

```text
Original speech:       A B C | D E F
Past-as-future teacher: A B C | A B C
Matched-future teacher: A B C | D E F
```

The donor comes from the current clip of the same recording and speaker. Its
offset is fixed by the target position, not randomly selected. Prediction
positions are sampled reproducibly across epochs. This directly tests novel
future content versus replaying observed speech; it does not remove the
repetition/positional inconsistency inherent in a historical-content control.

The teacher keeps the historical GlobalRegret full-clip speech mask, including
its original past access. Gesture history stays causal. At each selected
target, its complete gesture prefix is recomputed under that target's speech
memory. No hidden states produced with the real future are reused. The teacher
is gradient-free, but shares the student's updated weights. Depth conditioning
always uses the **original current** audio/text and the **same stochastic RVQ
prefix** used by the student.

## Paired controls and sampling

| Config | Teacher future |
| --- | --- |
| `configs/gtdm3_futurecontext_matched_beatx_allspk.yaml` | Correct future |
| `configs/gtdm3_futurecontext_past_as_future_beatx_allspk.yaml` | Equal-length historical speech |

Both use `UpperFaceLowerGTDM3FutureContextRegretRVQ` and differ only in
`regret_future_context` and experiment names. `regret_context_targets: 4`
samples up to four distinct eligible positions per length-group in each batch.
Sampling uses a separate deterministic seed from the training seed, split,
epoch/iteration and sample IDs, excluding the intervention mode. It does not
consume the student's random-number stream. The two runs receive the same
targets given the same batch ordering. Validation targets are stable across
epochs for the same samples and batch grouping.

Eligibility requires `t+1 >= L-t-1` and a nonempty future. For a 125-group
ten-second clip this is zero-based groups **62 through 123**. Student CE,
RVQ and auxiliary objectives still cover all valid positions. Only KL and
its teacher diagnostics use selected positions. KL is the per-head average
over selected valid targets, retaining the existing temperature and alpha;
it is not multiplied by the fraction of positions sampled.

Every selected position needs a separate temporal-prefix teacher forward, so
these runs cost more than the original dense teacher pass. Four is a starting
compute budget. Override `--regret_context_targets` identically for both runs
if changing it. A value at least 62 covers all eligible positions of a
125-group clip, at substantially higher cost.

Trailing q0 padding is trimmed before teacher construction and target selection;
noncontiguous q0 masks are rejected. Lower/face masks and body-part-dropout
prefixes are preserved. Clips with no eligible target contribute no KL but
retain normal student training. Fixed aligned training clips and one temporal
q0 head are required. Future-gesture distillation must remain disabled.

**Use the new matched config as the main comparison.** An already trained
`gtdm3_ablation_full_beatx_allspk.yaml` run uses dense KL at all positions and
is not a matched control for this experiment. The existing tokenreg-only run
is a useful no-distillation reference when data/codecs/batch/optimizer settings
are otherwise identical.

## Run

Copy the new model helper, trainer and configs, plus updated
`scripts/trainers/__init__.py` and `scripts/trainers/utils/config.py`, to the
same paths on the Linux/CUDA training host. Run from the repository root.
These configs use the all-speaker dataset and codecs from the training-component
matrix. Ensure the configured cache contains all speakers; override paths
identically in both commands when needed. Both are fresh single-GPU batch-64
runs, with offline W&B logging and distinct experiment names.

Matched-future control:

```bash
CUDA_VISIBLE_DEVICES=2 \
HF_HUB_OFFLINE=1 \
NO_TORCH_COMPILE=1 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_futurecontext_matched_beatx_allspk.yaml \
  --batch_size 64 \
  --loader_workers 8 \
  --pretrain_warmup_epochs 0 \
  --regret_future_context matched \
  --regret_context_targets 4 \
  --regret_start_epoch 0 \
  --regret_ramp_epochs 1 \
  --regret_weight 6 \
  --regret_initial_weight 0.6 \
  --regret_include_depth_levels True \
  --kinematic_rvq_start_epoch 0 \
  --kinematic_rvq_ramp_epochs 1 \
  --dense_future_gesture_weight 0 \
  --epochs 10000 \
  --test_period 20 \
  --lr_policy step \
  --lr_base 1e-4 \
  --decay_epochs 2000 \
  --decay_rate 0.8 \
  --wandb True \
  --wandb_mode offline \
  --wandb_project miburi_multi \
  --wandb_group allspk-future-content-control \
  --wandb_name matched-future-allspk-bs64 \
  --wandb_tags allspk globalregret matched-future sampled-kl rvq bs64
```

Past-as-future training:

```bash
CUDA_VISIBLE_DEVICES=2 \
HF_HUB_OFFLINE=1 \
NO_TORCH_COMPILE=1 \
OMP_NUM_THREADS=4 \
python scripts/train.py \
  --config configs/gtdm3_futurecontext_past_as_future_beatx_allspk.yaml \
  --batch_size 64 \
  --loader_workers 8 \
  --pretrain_warmup_epochs 0 \
  --regret_future_context past_as_future \
  --regret_context_targets 4 \
  --regret_start_epoch 0 \
  --regret_ramp_epochs 1 \
  --regret_weight 6 \
  --regret_initial_weight 0.6 \
  --regret_include_depth_levels True \
  --kinematic_rvq_start_epoch 0 \
  --kinematic_rvq_ramp_epochs 1 \
  --dense_future_gesture_weight 0 \
  --epochs 10000 \
  --test_period 20 \
  --lr_policy step \
  --lr_base 1e-4 \
  --decay_epochs 2000 \
  --decay_rate 0.8 \
  --wandb True \
  --wandb_mode offline \
  --wandb_project miburi_multi \
  --wandb_group allspk-future-content-control \
  --wandb_name past-as-future-allspk-bs64 \
  --wandb_tags allspk globalregret past-as-future sampled-kl rvq bs64
```

Use sequential runs on GPU 2, or separate GPUs with the same per-run setup.
The complete token-reg strengths (top-k 8, temperature 0.5, sampling 0.15,
soft mixture 0.5), face weight 2.5, seed 2342, architecture and codec paths
are shared in both YAMLs. Evaluate each trained checkpoint's **causal student**
on the same held-out sequences and generation seeds. Compare validation-selected
checkpoints and generated-motion metrics, not training CE alone.

## Diagnostics and checks

The new trainer keeps the existing raw/weighted KL, teacher/student CE,
accuracy, entropy and teacher-advantage metrics. It also logs:

- `context_eligible_fraction`: eligible q0 positions / valid q0 positions.
- `context_selected_fraction`: selected q0 positions / valid q0 positions.
- `context_changed_future_fraction`: fraction of future speech substeps where
  the **proposed historical replacement** differs from the original in at
  least one audio/text codebook. This diagnostic is computed in both controls,
  even when matched mode actually feeds the true future. Identical silence or
  repeated-token tails are retained, not selectively excluded from either run.

```bash
NO_TORCH_COMPILE=1 python -B -m scripts.smoke_test_past_as_future_training
```

Focused CPU tests cover the exact splice, eligibility and RNG isolation,
matched-teacher parity with the original GlobalRegret implementation, true-future
leakage prevention through multiple temporal layers, original current-speech
depth conditioning, stochastic prefixes, mask restoration after exceptions,
paired sampling with padded clips, student temporal/depth gradients, and real
config parsing. Existing shared-regret, condition-C trainer and fixed-checkpoint
future-speech smoke suites also pass. No dataset training or CUDA throughput
benchmark has been launched locally.
