# Shared depth weights: W&B update, 9 September 2026

The shared-depth experiment did not solve overfitting. Best validation loss
arrived modestly later, but was worse, and both shared runs were worse than
their original counterparts at matched late epochs. The strongest conclusion
is specific to this seed, dataset and optimization setup; it does not prove
weight sharing can never help.

The snapshot was taken at 06:49:59 UTC (14:49:59 China time), with history
retrieval completed at 14:50:22 China time. All four runs report `finished`.
This status does not explain why they ended before the configured 10,000 epochs.

| Experiment | Original run | Shared-depth run |
| --- | --- | --- |
| ForeMotion C | [foremotion-c-rvq-scott-bs64](https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/0st12z6a) | [foremotion-c-shareddepth-rvq-scott-bs64](https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/7so4vnve) |
| Supervised teacher C | [teacher-c-supervised-rvq-scott-bs64](https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/8hir7gzc) | [teacher-c-shareddepth-supervised-rvq-scott-bs64](https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/2ww4lkuq) |

The shared runs end at training epochs 1072 and 1424, with last validation at
1060 and 1420. All 12,123 logged history rows were retrieved with no missing
steps. There are 53/71 shared-run and 75/95 original-run validation checkpoints.
Absent metrics returned as null by the API were omitted, never forward-filled.
No nonfinite scalar metrics or generated-motion `eval/*` measurements were found.

## The ablation actually ran

Both new runs log `gestureformer_depformer_weights_per_step=False` and
255,342,088 trainable parameters, versus 320,812,552 originally. The reduction
of 65,470,464 matches the expected change in the depth core from 69,374,464 to
3,904,000 parameters. The temporal core remains 7,808,000; per-codebook inputs
and output heads remain separate.

Paired logged configurations differ only in the sharing flag and experiment
identifiers. Seed 2342, batch size 64, 477 training clips, 54 validation clips,
RVQ augmentation, regret settings, face/VAD objectives and LR settings match.
The old configs did not expose the flag, but their parameter count and loader
default correspond to separate weights. Logged configuration agreement does
not independently establish identical remote source revisions or file contents.

## Best checkpoints and matched comparisons

All values below are canonical, unweighted validation CE over all 20 tokens.

| Experiment | Original best | Shared best | Shared best penalty | Matched epoch | Original CE | Shared CE | Shared penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ForeMotion C | 5.8642 at 280 | 6.0956 at 320 | +3.95% | 1060 | 6.3648 | 6.9623 | +9.39% |
| Teacher C | 5.9621 at 220 | 6.1442 at 280 | +3.06% | 1420 | 8.9416 | 10.0554 | +12.46% |

The 40–60 epoch shift in the overall minimum is a modest delay, not a recovery
of generalization. Comparing the shared teacher at 1420 against the original
teacher's final value at 1900 would misleadingly make the two look similar.
Matched epochs show the shared teacher is already worse.

![Matched validation comparison](C:/Users/WenJChai/Desktop/miburi/reports/wandb_shared_depth_2026-09-09/shared_depth_comparison.png)

The depth mean in the figure is approximately `(20 * val/ce_loss -
val/temporal_q0_ce) / 19`. The q0 metric is rounded in BF16; therefore use broad
minimum regions when interpreting the derived curve. Direct per-body depth
metrics also support the same ordering but have their own BF16 accumulation.

## The overfitting evidence remains

From each shared run's best overall validation checkpoint to its final
validation checkpoint, the face-weighted hard depth CE has opposing trends:

| Shared run | Epochs | Training depth hard CE | Validation depth hard CE |
| --- | --- | --- | --- |
| ForeMotion C | 320 to 1060 | 6.888 to 3.761 | 8.063 to 10.375 |
| Teacher C | 280 to 1420 | 7.040 to 2.951 | 8.125 to 15.063 |

![Training and validation depth trends](C:/Users/WenJChai/Desktop/miburi/reports/wandb_shared_depth_2026-09-09/shared_depth_train_validation.png)

The direction of change supports overfitting. The absolute training-validation
gap is not a clean quantity: training uses stochastic RVQ prefixes/targets and
dropout; validation uses canonical codes. The main training CE also mixes
hard/soft depth targets and weights face by 2.5, whereas main validation CE is
an unweighted hard-label mean. These losses should not be compared as if they
were identical objectives.

The shared core fits training more slowly than the original, but held-out
performance still worsens. Reduced ability to fit training and persistent
overfitting can coexist: sharing may lose useful codebook specialization while
remaining trainable parameters still fit training-specific patterns.

Temporal q0 has a different trajectory. Shared ForeMotion q0 CE reaches
3.71875 around epoch 800 and is 3.75 at 1060, only 0.84% above its minimum;
the large overall increase is mainly elsewhere. Shared teacher q0 reaches
3.765625 at 800, then rises to 4.25 at 1420 (+12.86%). Its best q0 CE is slightly
better than the original teacher's 3.78125, but the later matched value is worse.
q0 is the first upper-body code, not a complete global motion representation.
Depth losses also update temporal hidden states, so these curves do not localize
overfitting exclusively to a particular parameter module.

## Face predictions account for most of the extra loss

At the matched epochs, the increase of shared over original validation CE
decomposes as follows. The numbers are contributions to the same 20-head mean,
not per-body mean CE values.

| Experiment | Upper contribution | Lower contribution | Face contribution | Face fraction of total gap |
| --- | ---: | ---: | ---: | ---: |
| ForeMotion, 1060 | +0.07386 | +0.13757 | +0.38612 | 64.6% |
| Teacher, 1420 | +0.01770 | +0.31662 | +0.77949 | 70.0% |

Within shared ForeMotion, face CE contribution rises from 1.4396 at the best
overall checkpoint to 2.4040 at 1060. It accounts for more than the net total
increase because lower-body performance improves over that same interval.
Within the shared teacher, face accounts for 52.2% and lower body 35.8% of
deterioration from its best overall checkpoint to 1420.

This points to face conditioning/objectives and competition among body groups
as useful hypotheses. It does not prove that `face_loss_weight=2.5`, face-logit
VAD guidance, or weight sharing itself causes the error. Separate codebooks
encode upper body/hands, lower body/translation, and face, so complete sharing
imposes a substantial common-transformation constraint across different tasks.

## Remaining capacity and optimization

The depth core shrank by 94.4%, but the entire trainable model shrank by only
20.4%. The new model still has 255.34M trainable parameters, with the large
pretrained audio/text embedding tables unfrozen. Default table dimensions imply
approximately 198.2M parameters in those tables; the exact attribution is an
architecture-derived estimate, not a W&B per-module measurement. Parameter
count alone does not establish effective capacity: not every vocabulary row
is necessarily used in the Scott data.

Both runs also log `weight_decay=0`. AdamW therefore does not supply nonzero
weight decay automatically. The core Transformer dropout remains 0.01 in the
local loader defaults. Existing memory/body dropout and RVQ augmentation are
additional regularizers, so the runs should not be described as unregularized.

The strongest immediately testable optimization issue is the cosine endpoint.
Both runs use start 200, end -1, and `epochs=10000`; the scheduler resolves the
endpoint to 10,000. `decay_epochs=2000` is not the cosine endpoint. At epoch
600 the scheduled LR is approximately 9.959e-5, and at 1000 it is 9.838e-5,
compared with the initial 1e-4. The last logged LRs are 9.808e-5 for shared
ForeMotion and 9.624e-5 for the shared teacher: 98.1% and 96.2% of the starting LR.

Local code references:

- [Cosine scheduler](C:/Users/WenJChai/Desktop/miburi/scripts/trainers/utils/scheduler_factory.py:61)
- [Trainable speech embedding construction](C:/Users/WenJChai/Desktop/miburi/miburi/models/gesture_lm.py:80)
- [Transformer defaults](C:/Users/WenJChai/Desktop/miburi/miburi/models/loaders.py:241)
- [AdamW weight decay](C:/Users/WenJChai/Desktop/miburi/scripts/trainers/utils/optim_factory.py:120)
- [Training loader](C:/Users/WenJChai/Desktop/miburi/scripts/trainers/baseglm_trainer.py:157)

The current local single-GPU training loader shuffles. An earlier statement
that the same last 29 samples are always excluded should not be used as a
current explanation. With 477 samples, batch 64 and drop_last=True, each epoch
has seven batches, but shuffled ordering changes which 29 are omitted. Remote
historical source revisions are not established by this configuration snapshot.

## Future-aware teacher advantage remains small

At shared ForeMotion's best checkpoint (320), the detached teacher is
essentially tied with the causal student: q0 CE 4.324080 versus 4.323873, and
depth CE 6.189037 versus 6.188849. At epoch 1060, teacher depth CE is 7.119901
versus student 7.131664, an advantage of 0.011763 (0.165%); depth accuracy is
8.371% versus 8.234%. q0 CE advantage is only 0.000431.

These paired metrics use the same trained weights under different speech
visibility. They show a small measured future-speech advantage at token level,
not an established improvement in full-utterance generated motion. ForeMotion
remaining better than the directly supervised teacher can also reflect the KL
regularization effect; these experiments do not isolate that mechanism.

## Recommended next experiment

Keep the original ForeMotion C architecture as the stronger validation baseline
and retain sharing as an informative ablation. Do not declare complete weight
sharing the new preferred architecture on these results alone.

The next single-variable optimization test should use a much shorter cosine
endpoint, for example `lr_cosine_start_epoch=200`, `lr_cosine_end_epoch=800`,
and the same `lr_min=1e-6`. This is a proposed starting point, not an established
optimum. At epoch 600 this schedule would be about 2.575e-5 instead of 9.959e-5.
Evaluate at matched epochs through approximately 1000 and compare the best
validation checkpoints. Merely moving the start earlier while leaving the end
at 10,000 does not meaningfully address this near-flat schedule.

If further ablations are needed, test nonzero weight decay, frozen speech
embeddings, or face-objective changes separately. Grouped depth sharing by body
region is a possible later compromise, but these curves do not justify another
architecture change before checking optimization. Early stopping/checkpoint
selection can avoid deploying a deteriorated model; it does not itself improve
the best achievable validation quality.

No model or training configuration was changed for this analysis. Raw histories,
configuration snapshots, comparison JSON and reproducible plotting/retrieval
scripts are saved alongside this report.
