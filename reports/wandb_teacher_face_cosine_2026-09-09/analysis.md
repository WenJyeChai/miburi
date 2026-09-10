# Completed teacher C: face weight 1 and cosine endpoint 800

The combined change improves the teacher's best unweighted validation CE and
reduces its later deterioration, but overfitting remains. The best overall
checkpoint is epoch 300, with epoch 280 almost tied. The final model at 1000
is substantially worse than those early checkpoints.

## Sources and checks

Read-only W&B snapshot retrieved on September 9, 2026, approximately 17:20
China time. Exact timestamps are in `retrieval.json`. Complete histories were
retrieved without missing logged steps, using disjoint ranges of at most 500
steps. API null placeholders were removed without forward filling.

| Run | W&B | State | History rows | Validation rows |
| --- | --- | --- | ---: | ---: |
| New teacher: face 1, cosine 800 | [lqd1cmjh](https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/lqd1cmjh) | finished | 2051 | 50 |
| Original teacher: face 2.5, cosine 10000 | [8hir7gzc](https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/8hir7gzc) | finished | 3923 | 95 |
| Shared-depth teacher: face 2.5, cosine 10000 | [2ww4lkuq](https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/2ww4lkuq) | finished | 2920 | 71 |

The new run logged face weight 1.0, original per-step depth weights, 1000
epochs, cosine start 200/end 800, base LR 1e-4 and floor 1e-6. AdamW weight
decay is 0. Both warmup settings are 0. All three runs use seed 2342, batch
64, 477 training samples, 54 validation samples and 89 test samples. The new
and original teachers each have 320,812,552 trainable parameters; the shared
teacher has 255,342,088.

The new versus original comparison changes only face weight, cosine endpoint,
training budget and experiment metadata/defaults made explicit. Logged codec,
RVQ, auxiliary, data and other behavioral settings match. The shared-depth
comparison additionally changes architecture, so it does not isolate the two
optimization changes. Configs do not establish identical remote source-code
revisions.

The final new-run training log is epoch 999. Epoch 1000 is final validation,
and its `epoch_train/*` values describe the retained preceding training
meters. Historical baseline epoch 1000 includes that epoch's training.
Therefore strict matched-epoch comparisons below use 980. No nonfinite metric
values or generated-motion `eval/*` metrics appear in these histories.

## Overall result

| Unweighted validation CE | Original teacher | Shared-depth teacher | New teacher |
| --- | ---: | ---: | ---: |
| Best through epoch 980 | 5.962052 at 220 | 6.144206 at 280 | **5.827263 at 300** |
| At matched epoch 980 | 7.798431 | 8.194585 | **6.734123** |

The new best improves on the original by **2.26%**, and on the shared-depth
teacher by **5.16%**. At epoch 980, it improves on the original by **13.65%**
and on the shared-depth teacher by **17.82%**. These are unweighted
`val/ce_loss` comparisons, so the improvements are not caused mechanically by
reducing the displayed loss coefficient.

The new final CE is **6.735901 at 1000**, **15.59% above its own best**.
Epoch 280 has CE 5.827765, only 0.000502 above epoch 300. Treat 280–300 as the
best observed region rather than assigning significance to that tiny
difference in one seeded run.

![Matched teacher curves](C:/Users/WenJChai/Desktop/miburi/reports/wandb_teacher_face_cosine_2026-09-09/teacher_comparison.png)

## Face and body behavior

These values are the existing unweighted `val/ce_part` contributions: head
losses summed within each body part and divided by 20. Upper and lower each
contain eight codebooks, while face contains four. For mean CE per codebook,
multiply upper/lower by 2.5 and face by 5, as done in the chart.

| New teacher part | Minimum | Minimum epoch | Final at 1000 | Increase above own minimum |
| --- | ---: | ---: | ---: | ---: |
| Face | 1.372010 | 220 | 1.796760 | **30.96%** |
| Lower body + translation | 1.852356 | 280 | 2.222809 | **20.00%** |
| Upper body + hands | 2.582305 | 300 | 2.716331 | **5.19%** |

Face remains the first part to deteriorate, followed by lower and upper body.
The original teacher's face minimum was 1.350813 at epoch 140. The new face
minimum occurs later, but is slightly worse numerically. Its benefit is
primarily reduced later deterioration rather than a better best-ever face CE.

At matched epoch 980, new versus original:

- Face CE is **25.28% lower**: 2.403952 to 1.796178.
- Lower-body CE is **15.29% lower**: 2.622852 to 2.221692.
- Upper-body CE is **2.00% lower**: 2.771627 to 2.716253.

Face accounts for 57.1% of the overall CE improvement at 980, lower body
37.7%, and upper body 5.2%. Thus the improvement extends beyond facial tokens.

## Temporal versus depth behavior

The new temporal q0 CE continues improving after depth and body-part validation
have begun to deteriorate. It reaches a rounded minimum of **3.90625** at 780
and stays at that logged value through 1000. However, at epoch 980 the original
teacher achieves **3.78125**, so the new q0 CE is **3.31% worse**. Accuracy is
24.19% for the new model versus 24.81% for the original at 980, a decrease of
0.62 percentage points. The new best q0 accuracy is 24.28% at 940.

This is a tradeoff: a better full-token validation objective accompanies a
slightly weaker late q0 predictor. It does not prove that temporal features as
a whole are worse: q0 is one upper-body base code, and depth losses also train
the temporal representation.

The approximate unweighted mean depth CE derived from total CE and the rounded
q0 diagnostic is 5.894 around epoch 260, rising to 6.885 at 1000. Direct
per-part depth diagnostics give approximately 5.896 around 260–300 and 6.875
at 1000; rounding and accumulation explain the small difference. These are
consistent with substantial remaining depth overfitting.

Within the new run, from epoch 300 to 980, training hard depth CE falls from
**4.285714 to 2.841518**, while validation hard depth CE rises from
**5.593750 to 6.531250**. These logged values use division by 20. The graph
rescales them by 20/19 to mean CE across depth heads. Training uses stochastic
RVQ prefixes and validation uses canonical prefixes, so the direction of
divergence is informative but the absolute gap is not a controlled estimate
using identical input distributions.

![New-run depth gap and learning rate](C:/Users/WenJChai/Desktop/miburi/reports/wandb_teacher_face_cosine_2026-09-09/teacher_training_schedule.png)

## What the cosine schedule accomplished

Every logged new-run learning-rate value matches the intended schedule under
the existing end-of-epoch scheduler convention:

| Training epoch label | Logged LR |
| ---: | ---: |
| 200 | 1.00000e-4 |
| 300 | 9.34973e-5 |
| 400 | 7.54741e-5 |
| 600 | 2.59748e-5 |
| 800 | 1.00068e-6 |
| 801 onward | 1.00000e-6 |

Validation CE changes only from **6.715914 at 800 to 6.735901 at 1000**,
an increase of **0.30%**. The declining slope and near plateau align with the
LR falling to its floor. This is consistent with the schedule slowing further
deterioration after the model has already overfit; it does not restore the
early best checkpoint. Extending this exact run at the same floor has little
support from the observed final 200 epochs as a way to recover that checkpoint.

At the best overall epoch 300, LR is still about 93.5% of its initial value.
Thus even this shorter schedule applies most of its reduction after the early
validation minimum. Earlier substantial decay is a plausible future test, but
the q0 tradeoff shows why accelerating a global schedule may also limit useful
temporal learning. A separate temporal/depth optimizer schedule is another
hypothesis, not a change validated by this run.

## Recommended next action

1. Evaluate `last_300.safetensors` as the best observed overall teacher
   checkpoint, with epoch 280 as a nearly tied alternative. Compare generated
   motion on the held-out utterances; these token histories alone do not
   establish lip synchronization or gesture quality.
2. Use the already prepared face-only and cosine-only controls to identify
   which change improves the best CE and which mainly controls late
   deterioration. Both factors changed here, so they cannot be separated from
   this one run.
3. Preserve this combined run as the strongest teacher baseline by the
   observed unweighted CE among these three runs, while retaining q0 as an
   explicit tradeoff metric. Confirm any small best-checkpoint gain with
   additional seeds before treating it as robust.

`val/kinematic_hard_ce` is weighted by the configured face coefficient in the
older runs and unweighted in the new run. Comparing that aggregate directly
across these experiments would exaggerate the gain. Face token accuracy and
prediction entropy were not logged: q0 entropy and RVQ target entropy do not
substitute for those measurements. No claim about face calibration or
face-token accuracy can be established from this snapshot.

No model, configuration or training state was changed during this review.
Raw histories, inventory, retrieval metadata, `comparison.json`, plots and the
reproducible `analyze.py` are saved in this directory.
