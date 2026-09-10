"""Reproduce the teacher face-weight/cosine analysis from complete W&B histories."""
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEW, ORIGINAL, SHARED = "lqd1cmjh", "8hir7gzc", "2ww4lkuq"
IDS = (ORIGINAL, SHARED, NEW)
RUNS = {r["name"]: r for r in json.loads((ROOT / "runs.json").read_text())}
HISTORY = {rid: json.loads((ROOT / f"{rid}_history.json").read_text()) for rid in IDS}
VAL = {rid: {r["trainer/epoch"]: r for r in rows if "val/ce_loss" in r}
       for rid, rows in HISTORY.items()}
CONFIG = {rid: {k: v["value"] for k, v in json.loads(RUNS[rid]["config"]).items()}
          for rid in IDS}
KEYS = ["val/ce_loss", "val/ce_upper", "val/ce_lower", "val/ce_face",
        "val/temporal_q0_ce", "val/temporal_q0_acc", "derived/depth_mean_ce"]
for vals in VAL.values():
    for row in vals.values():
        # q0 is rounded in the logged diagnostic; this derived mean is approximate.
        row["derived/depth_mean_ce"] = (20 * row["val/ce_loss"] - row["val/temporal_q0_ce"]) / 19

def metrics(vals):
    output = {}
    for key in KEYS:
        best_epoch = (max if key.endswith("_acc") else min)(vals, key=lambda e: vals[e][key])
        best, last = vals[best_epoch][key], vals[max(vals)][key]
        output[key] = {"best_epoch": best_epoch, "best_value": best,
                       "tied_best_epochs": [e for e in vals if vals[e][key] == best],
                       "last_epoch": max(vals), "last_value": last,
                       "last_change_from_best_percent": 100 * (last / best - 1)}
    return output

result = {"runs": {}, "comparisons_at_980": {}, "config_differences": {}}
for rid in IDS:
    summary = json.loads(RUNS[rid]["summaryMetrics"])
    lr_rows = [r for r in HISTORY[rid] if "train/learning_rate" in r]
    result["runs"][rid] = {
        "name": RUNS[rid]["displayName"], "state": RUNS[rid]["state"],
        "validation_count": len(VAL[rid]), "metrics": metrics(VAL[rid]),
        "best_through_980": metrics({e: r for e, r in VAL[rid].items() if e <= 980}),
        "last_logged_training_epoch": max(r["trainer/epoch"] for r in lr_rows),
        "final_lr": lr_rows[-1]["train/learning_rate"],
        "trainable_parameters": summary["trainable_parameters"],
        "train_samples": summary["train_samples"], "val_samples": summary["val_samples"],
        "test_samples": summary["test_samples"],
        "eval_metrics": sorted({k for r in HISTORY[rid] for k in r if k.startswith("eval/")}),
        "nonfinite_metric_count": sum(isinstance(v, float) and not math.isfinite(v)
                                      for r in HISTORY[rid] for v in r.values()),
    }
    if rid == NEW:
        continue
    pair = {}
    for key in KEYS:
        old, new = VAL[rid][980][key], VAL[NEW][980][key]
        pair[key] = {"baseline": old, "new": new, "delta": new - old,
                     "change_percent": 100 * (new / old - 1)}
    old_best = min(r["val/ce_loss"] for e, r in VAL[rid].items() if e <= 980)
    new_best = min(r["val/ce_loss"] for e, r in VAL[NEW].items() if e <= 980)
    pair["best_total_ce_change_percent"] = 100 * (new_best / old_best - 1)
    delta = pair["val/ce_loss"]["delta"]
    pair["body_fraction_of_matched_gain"] = {
        part: pair[f"val/ce_{part}"]["delta"] / delta for part in ("upper", "lower", "face")}
    result["comparisons_at_980"][rid] = pair
    result["config_differences"][rid] = {
        k: [CONFIG[rid].get(k), CONFIG[NEW].get(k)]
        for k in sorted(set(CONFIG[rid]) | set(CONFIG[NEW]))
        if k != "_wandb" and CONFIG[rid].get(k) != CONFIG[NEW].get(k)}

new_vals = VAL[NEW]
best_epoch = min(new_vals, key=lambda e: new_vals[e]["val/ce_loss"])
result["new_run_detail"] = {
    "best_total_epoch": best_epoch,
    "best_total_row": new_vals[best_epoch],
    "last_row": new_vals[max(new_vals)],
    "plateau_800_to_1000": {
        key: {"at_800": new_vals[800][key], "at_1000": new_vals[1000][key],
              "change_percent": 100 * (new_vals[1000][key] / new_vals[800][key] - 1)}
        for key in KEYS},
    "train_val_at_best_and_980": {
        str(e): {k: v for k, v in new_vals[e].items()
                 if k.startswith("epoch_train/") or k in KEYS
                 or k.startswith("val/kinematic_")}
        for e in (best_epoch, 980)},
}
expected_lr_errors = []
for row in HISTORY[NEW]:
    if "train/learning_rate" not in row:
        continue
    epoch = row["trainer/epoch"]
    scheduler_epoch = max(0, epoch - 1)
    progress = min(1., max(0., (scheduler_epoch - 200) / 600))
    expected = 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * progress))
    if not math.isclose(row["train/learning_rate"], expected, rel_tol=1e-10):
        expected_lr_errors.append(epoch)
result["new_run_detail"]["lr_schedule_mismatch_epochs"] = expected_lr_errors
assert not expected_lr_errors
(ROOT / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "#fcfcfa",
                     "axes.facecolor": "#fcfcfa"})
COLORS = {ORIGINAL: "#b45536", SHARED: "#9a9eab", NEW: "#167e85"}
LABELS = {ORIGINAL: "Original teacher: face 2.5, cosine 10000",
          SHARED: "Shared-depth teacher: face 2.5, cosine 10000",
          NEW: "New teacher: face 1.0, cosine 800"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True)
for ax, (key, title, mult) in zip(axes.flat, [
    ("val/ce_loss", "Overall validation CE", 1),
    ("val/ce_face", "Face: mean CE per codebook", 5),
    ("val/ce_lower", "Lower body: mean CE per codebook", 2.5),
    ("val/temporal_q0_ce", "Temporal q0 validation CE", 1),
]):
    for rid in (SHARED, ORIGINAL, NEW):
        epochs = sorted(e for e in VAL[rid] if e <= 980)
        vals = [VAL[rid][e][key] * mult for e in epochs]
        ax.plot(epochs, vals, color=COLORS[rid], lw=2.3 if rid == NEW else 1.7,
                ls="--" if rid == SHARED else "-", label=LABELS[rid])
        if rid == NEW:
            idx = min(range(len(vals)), key=vals.__getitem__)
            ax.scatter(epochs[idx], vals[idx], color=COLORS[rid], s=25, zorder=4)
    ax.axvline(800, color="#999999", lw=.7, ls=":")
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlim(0, 1000)
    ax.grid(axis="y", color="#dedede", lw=.6)
axes[0, 0].set_ylim(5.65, 8.35)
axes[0, 1].set_ylim(6.5, 15.2)
axes[1, 0].set_ylim(4.5, 6.8)
axes[1, 1].set_ylim(3.55, 7.8)
axes[0, 0].annotate("Best: 5.827 at epoch 300", (300, new_vals[300]["val/ce_loss"]),
                    xytext=(350, 5.99), fontsize=9, color=COLORS[NEW],
                    arrowprops={"arrowstyle": "-", "color": COLORS[NEW], "lw": .8})
for ax in axes[1]:
    ax.set_xlabel("Epoch")
fig.suptitle("Teacher improves, but overfitting remains", fontsize=19, weight="bold",
             x=.075, ha="left", y=.982)
fig.text(.075, .939, "Complete teacher runs | unsmoothed validation | matched epoch labels through 980 | lower is better",
         fontsize=10, color="#555555")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(.065, .923),
           ncol=1, frameon=False, fontsize=9)
fig.text(.075, .022, "Face and lower-body values are unweighted per-codebook means. Dots: new-run minima. Vertical line: cosine endpoint 800.",
         fontsize=9, color="#555555")
fig.subplots_adjust(left=.075, right=.975, top=.795, bottom=.085, hspace=.25, wspace=.20)
fig.savefig(ROOT / "teacher_comparison.png", dpi=170)
fig.savefig(ROOT / "teacher_comparison.svg")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
epochs = sorted(e for e in new_vals if e <= 980)
for prefix, label, color, ls in (("epoch_train", "Training hard depth CE", "#bd773b", "--"),
                                 ("val", "Validation hard depth CE", "#167e85", "-")):
    axes[0].plot(epochs, [new_vals[e][f"{prefix}/kinematic_hard_ce"] * 20 / 19 for e in epochs],
                 label=label, color=color, ls=ls, lw=2)
axes[0].set_title("New run: depth training improves as validation worsens", loc="left", fontsize=11)
axes[0].set_ylabel("Mean hard CE across 19 depth heads")
axes[0].legend(frameon=False, fontsize=9)
for rid in (ORIGINAL, NEW):
    rows = [r for r in HISTORY[rid] if "train/learning_rate" in r and r["trainer/epoch"] <= 999]
    axes[1].semilogy([r["trainer/epoch"] for r in rows], [r["train/learning_rate"] for r in rows],
                     color=COLORS[rid], label="Original teacher" if rid == ORIGINAL else "New teacher", lw=2)
axes[1].set_title("Logged learning rate follows the shorter schedule", loc="left", fontsize=11)
axes[1].set_ylabel("Learning rate (log scale)")
axes[1].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.grid(axis="y", color="#dedede", lw=.6)
    ax.axvline(800, color="#999999", lw=.7, ls=":")
fig.text(.07, .025, "New face weight is 1.0, so its depth diagnostic is unweighted. Training uses stochastic RVQ prefixes; validation uses canonical prefixes.",
         fontsize=8.5, color="#555555")
fig.subplots_adjust(left=.07, right=.975, top=.90, bottom=.18, wspace=.27)
fig.savefig(ROOT / "teacher_training_schedule.png", dpi=170)
fig.savefig(ROOT / "teacher_training_schedule.svg")
plt.close(fig)

print(json.dumps({"runs": {rid: record["metrics"] for rid, record in result["runs"].items()},
                  "comparisons_at_980": result["comparisons_at_980"],
                  "plateau": result["new_run_detail"]["plateau_800_to_1000"]}, indent=2))
