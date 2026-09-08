"""Reproduce the condition-C comparison from downloaded W&B JSON histories."""

import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = {
    "foremotion": ("0st12z6a", "ForeMotion C: causal student", "#156b8a"),
    "teacher": ("8hir7gzc", "Supervised teacher C", "#c0543f"),
}
histories = {
    name: json.loads((ROOT / f"{run_id}_history.json").read_text())
    for name, (run_id, _, _) in RUNS.items()
}
validations = {
    name: {r["trainer/epoch"]: r for r in rows if "val/ce_loss" in r}
    for name, rows in histories.items()
}
matched_epoch = max(set(validations["foremotion"]) & set(validations["teacher"]))
comparison_keys = [
    "val/ce_loss", "val/ce_upper", "val/ce_lower", "val/ce_face",
    "val/temporal_q0_ce", "val/temporal_q0_acc", "val/temporal_q0_top5_acc",
    "val/kinematic_hard_ce",
]
result = {"matched_epoch": matched_epoch, "runs": {}}
for name, rows in histories.items():
    vals = validations[name]
    best_epoch = min(vals, key=lambda e: vals[e]["val/ce_loss"])
    latest_epoch = max(vals)
    q0_best = min(vals, key=lambda e: vals[e]["val/temporal_q0_ce"])
    acc_best = max(vals, key=lambda e: vals[e]["val/temporal_q0_acc"])
    result["runs"][name] = {
        "best_q0_ce_epoch": q0_best,
        "best_q0_ce": vals[q0_best]["val/temporal_q0_ce"],
        "best_q0_accuracy_epoch": acc_best,
        "best_q0_accuracy": vals[acc_best]["val/temporal_q0_acc"],
        "latest_q0_ce": vals[latest_epoch]["val/temporal_q0_ce"],
        "latest_q0_accuracy": vals[latest_epoch]["val/temporal_q0_acc"],
        "history_rows": len(rows),
        "latest_training_epoch": max(r.get("trainer/epoch", 0) for r in rows),
        "latest_validation_epoch": latest_epoch,
        "best_validation_epoch": best_epoch,
        "best_validation_ce": vals[best_epoch]["val/ce_loss"],
        "latest_validation_ce": vals[latest_epoch]["val/ce_loss"],
        "matched": {k: vals[matched_epoch][k] for k in comparison_keys},
        "median_epoch_seconds_after_100": statistics.median(
            r["epoch/duration_seconds"] for r in rows
            if "epoch/duration_seconds" in r and r.get("trainer/epoch", 0) >= 100
        ),
        "median_reserved_gpu_gb_after_100": statistics.median(
            r["system/gpu_memory_reserved_gb"] for r in rows
            if "system/gpu_memory_reserved_gb" in r and r.get("trainer/epoch", 0) >= 100
        ),
        "latest_learning_rate": next(
            r["train/learning_rate"] for r in reversed(rows) if "train/learning_rate" in r
        ),
        "eval_keys": sorted({k for r in rows for k in r if k.startswith("eval/")}),
    }
f = validations["foremotion"][matched_epoch]
result["shared_teacher_at_matched_epoch"] = {
    k: v for k, v in f.items() if k.startswith("val/regret_")
}
result["teacher_q0_ce_wins"] = sum(
    r["val/regret_teacher_q0_ce"] < r["val/regret_student_q0_ce"]
    for r in validations["foremotion"].values()
)
result["teacher_depth_ce_wins"] = sum(
    r["val/regret_depth_teacher_ce"] < r["val/regret_depth_student_ce"]
    for r in validations["foremotion"].values()
)
result["foremotion_validation_count"] = len(validations["foremotion"])
(ROOT / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
    "figure.facecolor": "#fcfcfa", "axes.facecolor": "#fcfcfa",
})
fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
for name, (_, label, color) in RUNS.items():
    vals = validations[name]
    xs = sorted(vals)
    for ax, key in zip(axes.flat[:3], [
        "val/ce_loss", "val/temporal_q0_ce", "val/ce_face",
    ]):
        ax.plot(xs, [vals[e][key] for e in xs], color=color, lw=2.1, label=label)
    best = result["runs"][name]["best_validation_epoch"]
    best_ce = vals[best]["val/ce_loss"]
    axes[0, 0].scatter(best, best_ce, color=color, s=48, zorder=5)

axes[0, 0].set_title("All-token validation CE: lower is better")
axes[0, 0].set_ylabel("Canonical hard-label CE")
axes[0, 0].set_ylim(5.5, max(r["val/ce_loss"] for vals in validations.values() for r in vals.values()) + 0.35)
axes[0, 0].legend(frameon=False, loc="upper left", fontsize=9)
axes[0, 0].axvline(matched_epoch, color="#777777", ls=":", lw=1)
axes[0, 1].set_title("Temporal q0 now worsens in teacher-only")
axes[0, 1].set_ylabel("q0 validation CE")
axes[1, 0].set_title("Face validation loss worsens first")
axes[1, 0].set_ylabel("Face CE contribution (as logged)")

vals = validations["foremotion"]
xs = sorted(vals)
for key, label, color in [
    ("val/regret_teacher_advantage", "q0 targets", "#604292"),
    ("val/regret_depth_teacher_advantage", "Depth targets", "#65924b"),
]:
    ys = [100 * vals[e][key] for e in xs]
    axes[1, 1].plot(xs, ys, label=label, color=color, lw=2)
    axes[1, 1].annotate(f"{ys[-1]:.2f}%", (xs[-1], ys[-1]), xytext=(7, 0),
                        textcoords="offset points", color=color, fontsize=9)
axes[1, 1].axhline(50, color="#777777", ls="--", lw=1)
axes[1, 1].set_title("Shared teacher: higher target probability")
axes[1, 1].set_ylabel("Fraction of validation targets (%)")
axes[1, 1].set_ylim(40, 58)
axes[1, 1].set_xlim(0, matched_epoch + 230)
axes[1, 1].legend(frameon=False, loc="lower right", fontsize=9)
for ax in axes.flat:
    ax.set_xlabel("Training epoch")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
fig.suptitle("Condition C: completed-run update", fontsize=19, weight="bold", x=0.07, ha="left")
fig.text(0.07, 0.929, f"W&B snapshot: 8 September 2026, 20:26 China time | matched epoch {matched_epoch} | unsmoothed validation values",
         fontsize=10, color="#505050")
fig.text(0.07, 0.025, "Same seed, data and RVQ settings. These are teacher-forced token metrics; no generated-motion evaluation is logged.",
         fontsize=9, color="#505050")
fig.subplots_adjust(left=0.075, right=0.965, top=0.87, bottom=0.105, hspace=0.36, wspace=0.25)
fig.savefig(ROOT / "condition_c_comparison.png", dpi=170)
fig.savefig(ROOT / "condition_c_comparison.svg")
print(json.dumps(result, indent=2))
