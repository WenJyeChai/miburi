# Existing body-part CE metrics

The refreshed W&B summaries have the same finished run status and logged step
counts as the complete saved histories. The following results use existing
`val/ce_upper`, `val/ce_lower` and `val/ce_face` measurements.

Sources: [shared-depth ForeMotion C](https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/7so4vnve)
and [shared-depth supervised teacher C](https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/2ww4lkuq).

| Run | Part | Best logged CE | Epoch of best | Latest CE | Increase from best |
| --- | --- | ---: | ---: | ---: | ---: |
| ForeMotion C, latest 1060 | Upper | 2.5578 | 740 | 2.6039 | 1.8% |
| ForeMotion C, latest 1060 | Lower | 1.8715 | 800 | 1.9545 | 4.4% |
| ForeMotion C, latest 1060 | Face | 1.3704 | 220 | 2.4040 | 75.4% |
| Teacher C, latest 1420 | Upper | 2.5783 | 440 | 3.0787 | 19.4% |
| Teacher C, latest 1420 | Lower | 1.9693 | 500 | 3.4891 | 77.2% |
| Teacher C, latest 1420 | Face | 1.3749 | 180 | 3.4876 | 153.7% |

![Body-part validation deterioration](C:/Users/WenJChai/Desktop/miburi/reports/wandb_shared_depth_2026-09-09/body_part_overfitting.png)

The figure normalizes each curve by its own minimum to compare deterioration;
the table retains the exact metric scale used by W&B. The minimum epochs are
sampled every 20 epochs, not exact estimates of when overfitting starts.

For shared-depth ForeMotion, the early aggregate problem is predominantly face
prediction. Upper and lower body continue improving much longer and only mildly
deteriorate by the final checkpoint. For the supervised teacher, face is again
first, but lower body and upper body subsequently deteriorate too.

`val/ce_upper` includes q0 plus seven upper-body depth codebooks. There is also
an existing `val/kinematic_upper_hard_ce` metric which excludes q0: in shared
ForeMotion it increases from 2.359375 to 2.421875, only 2.65%. Its BF16 minimum
is tied over epochs 380–720. Thus mild upper deterioration is not merely an
artifact of including q0. The supervised teacher's corresponding increase is
20.53%, consistent with the main upper-body curve.

The per-body CE values are sums divided by 20: eight upper, eight lower, four
face codebooks. They are contributions to the overall mean; do not compare
their raw magnitudes as if each averages the same number of codebooks. To get
per-codebook means, multiply upper/lower by 2.5 and face by 5. These validation
metrics do not contain the training objective's 2.5 face weighting. Their
percentage increases are unaffected by this fixed normalization.

Training has corresponding per-part metrics, but `epoch_train/ce_part` mixes
hard and soft stochastic-RVQ losses, unlike canonical hard validation. The
`kinematic_*_hard_ce` family avoids that mixture but training still uses
stochastic prefixes and targets. Opposing trends support generalization loss;
raw train-validation gaps should not be treated as identical-objective gaps.

The group-level breakdown is already available and informative. What is still
absent is the separate CE for every individual codebook, such as lower base
q8 versus its residuals, or face base q16 versus its residuals. No new training
or metric implementation was needed for this body-part analysis.
