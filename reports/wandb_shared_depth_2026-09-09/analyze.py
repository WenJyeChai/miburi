"""Reproduce the matched-epoch shared-depth comparison from the saved snapshot."""
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = {"foremotion": ("0st12z6a", "7so4vnve"), "teacher": ("8hir7gzc", "2ww4lkuq")}
RUNS = {r["name"]: r for r in json.loads((ROOT / "runs.json").read_text())}
HISTORY = {rid: json.loads((ROOT / f"{rid}_history.json").read_text())
           for pair in PAIRS.values() for rid in pair}
VAL = {rid: {r["trainer/epoch"]: dict(r) for r in rows if "val/ce_loss" in r}
       for rid, rows in HISTORY.items()}
TRAIN = {rid: {r["trainer/epoch"]: r for r in rows if "epoch_train/ce_loss" in r}
         for rid, rows in HISTORY.items()}
for vals in VAL.values():
    for row in vals.values():
        # q0 is BF16 rounded, so this is an approximate unweighted mean of
        # the other 19 heads, not the face-weighted kinematic_hard_ce metric.
        row["derived/depth_mean_ce"] = (20 * row["val/ce_loss"] - row["val/temporal_q0_ce"]) / 19

KEYS = ["val/ce_loss", "val/temporal_q0_ce", "val/temporal_q0_acc",
        "derived/depth_mean_ce", "val/kinematic_hard_ce",
        "val/ce_upper", "val/ce_lower", "val/ce_face"]
result = {"runs": {}, "pairs": {}}
for rid, vals in VAL.items():
    latest = max(vals)
    overall_best = min(vals, key=lambda e: vals[e]["val/ce_loss"])
    config = {k: v["value"] for k, v in json.loads(RUNS[rid]["config"]).items()}
    summary = json.loads(RUNS[rid]["summaryMetrics"])
    metrics = {}
    for key in KEYS:
        minimize = not key.endswith("_acc")
        best = (min if minimize else max)(vals, key=lambda e: vals[e][key])
        metrics[key] = {"best_epoch": best, "best_value": vals[best][key],
                        "tied_best_epochs": [e for e in vals if vals[e][key] == vals[best][key]],
                        "latest_value": vals[latest][key],
                        "change_from_best_percent": 100 * (vals[latest][key] / vals[best][key] - 1)}
    lr_rows = [r for r in HISTORY[rid] if "train/learning_rate" in r]
    record = {
        "name": RUNS[rid]["displayName"], "state": RUNS[rid]["state"],
        "latest_validation_epoch": latest, "latest_training_epoch": summary["trainer/epoch"],
        "validation_count": len(vals), "metrics": metrics,
        "train_depth_hard_ce_at_best_total": TRAIN[rid][overall_best]["epoch_train/kinematic_hard_ce"],
        "train_depth_hard_ce_at_latest_validation": TRAIN[rid][latest]["epoch_train/kinematic_hard_ce"],
        "val_depth_hard_ce_at_best_total": vals[overall_best]["val/kinematic_hard_ce"],
        "val_depth_hard_ce_at_latest_validation": vals[latest]["val/kinematic_hard_ce"],
        "latest_lr": lr_rows[-1]["train/learning_rate"],
        "parameters": summary["parameters"], "trainable_parameters": summary["trainable_parameters"],
        "train_samples": summary["train_samples"], "val_samples": summary["val_samples"],
        "eval_keys": sorted({k for r in HISTORY[rid] for k in r if k.startswith("eval/")}),
        "nonfinite_metrics": sum(isinstance(v, float) and not math.isfinite(v)
                                 for row in HISTORY[rid] for v in row.values()),
        "median_epoch_seconds_after_100": statistics.median(
            r["epoch/duration_seconds"] for r in HISTORY[rid]
            if "epoch/duration_seconds" in r and r["trainer/epoch"] >= 100),
        "config": {k: config.get(k) for k in ["gestureformer_depformer_weights_per_step",
            "random_seed", "batch_size", "weight_decay", "textaudio_emb_freeze",
            "lr_policy", "lr_base", "lr_cosine_start_epoch", "lr_cosine_end_epoch",
            "lr_min", "epochs", "face_loss_weight", "vad_use_face_logits"]},
    }
    deterioration = vals[latest]["val/ce_loss"] - vals[overall_best]["val/ce_loss"]
    record["body_deterioration_from_best_total"] = {
        part: {"ce_contribution_change": vals[latest][f"val/ce_{part}"] - vals[overall_best][f"val/ce_{part}"],
               "fraction_of_net_total_increase": (vals[latest][f"val/ce_{part}"] - vals[overall_best][f"val/ce_{part}"]) / deterioration}
        for part in ("upper", "lower", "face")}
    if "val/regret_depth_teacher_ce" in vals[latest]:
        row = vals[latest]
        record["shared_teacher_latest"] = {k: v for k, v in row.items() if k.startswith("val/regret_")}
    result["runs"][rid] = record

for name, (original, shared) in PAIRS.items():
    epoch = max(set(VAL[original]) & set(VAL[shared]))
    pair = {"epoch": epoch, "original_id": original, "shared_id": shared, "metrics": {}}
    for key in KEYS:
        a, b = VAL[original][epoch][key], VAL[shared][epoch][key]
        pair["metrics"][key] = {"original": a, "shared": b, "delta": b - a,
                                "change_percent": 100 * (b / a - 1)}
    total_delta = pair["metrics"]["val/ce_loss"]["delta"]
    pair["body_share_of_matched_total_gap"] = {
        part: pair["metrics"][f"val/ce_{part}"]["delta"] / total_delta
        for part in ("upper", "lower", "face")}
    best_a = result["runs"][original]["metrics"]["val/ce_loss"]["best_value"]
    best_b = result["runs"][shared]["metrics"]["val/ce_loss"]["best_value"]
    pair["best_total_ce_change_percent"] = 100 * (best_b / best_a - 1)
    cfg_a = {k:v["value"] for k,v in json.loads(RUNS[original]["config"]).items()}
    cfg_b = {k:v["value"] for k,v in json.loads(RUNS[shared]["config"]).items()}
    pair["config_differences"] = {k:[cfg_a.get(k),cfg_b.get(k)] for k in sorted(set(cfg_a)|set(cfg_b))
                                  if cfg_a.get(k)!=cfg_b.get(k) and k!="_wandb"}
    result["pairs"][name] = pair

(ROOT / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "axes.titleweight": "bold",
                     "figure.facecolor": "#fcfcfa", "axes.facecolor": "#fcfcfa"})
colors = ("#1d6885", "#c55835")
fig, axes = plt.subplots(3, 2, figsize=(12, 10.8), sharex="col")
for col, (name, pair) in enumerate(PAIRS.items()):
    end = result["pairs"][name]["epoch"]
    for index, rid in enumerate(pair):
        epochs = sorted(e for e in VAL[rid] if e <= end)
        for row, key in enumerate(("val/ce_loss", "derived/depth_mean_ce", "val/temporal_q0_ce")):
            values = [VAL[rid][e][key] for e in epochs]
            ax = axes[row, col]
            label = "Original: separate depth weights" if index == 0 else "New: shared depth weights"
            ax.plot(epochs, values, lw=2, color=colors[index], label=label)
            best = min(range(len(values)), key=values.__getitem__)
            ax.scatter(epochs[best], values[best], color=colors[index], s=25, zorder=5)
            label_y = (8 if index else -9) if row == 2 else 0
            ax.annotate(f"{values[-1]:.3f}", (epochs[-1], values[-1]), xytext=(6, label_y),
                        textcoords="offset points", color=colors[index], va="center", fontsize=9)
            ax.set_xlim(0, end * 1.12)
            ax.grid(axis="y", color="#d9d9d9", lw=.6)
    axes[0,col].set_title("ForeMotion C: causal student" if name == "foremotion" else "Directly supervised teacher C")
    axes[2,col].set_xlabel("Epoch (both curves clipped to the shared run's last validation)")
    axes[0,col].set_ylim(5.6, 10.5)
    axes[1,col].set_ylim(5.6, 10.9)
    axes[2,col].set_ylim(3.4, 8.1)
for row, ylabel in enumerate(("All 20 tokens: validation CE", "19 depth tokens: mean validation CE", "Temporal q0: validation CE")):
    axes[row,0].set_ylabel(ylabel)
fig.suptitle("Shared depth weights did not remove overfitting", fontsize=19, weight="bold", x=.07, ha="left", y=.982)
fig.text(.07, .946, "W&B snapshot: 9 September 2026, 14:50 China time | same seed and logged training settings | lower is better", fontsize=10, color="#555555")
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.06,.936), ncol=2, frameon=False)
fig.text(.07,.023,"Unsmoothed teacher-forced validation. Dots mark minima within the displayed range. Depth mean is derived from total CE and rounded q0 CE.",fontsize=8.8,color="#555555")
fig.subplots_adjust(left=.09, right=.963, top=.874, bottom=.085, hspace=.18,wspace=.20)
fig.savefig(ROOT / "shared_depth_comparison.png", dpi=160)
fig.savefig(ROOT / "shared_depth_comparison.svg")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12,4.6))
for ax, (name, (_,rid)) in zip(axes, PAIRS.items()):
    epochs = sorted(VAL[rid])
    ax.plot(epochs,[TRAIN[rid][e]["epoch_train/kinematic_hard_ce"] for e in epochs], color="#1d6885", label="Training hard-CE component", lw=2)
    ax.plot(epochs,[VAL[rid][e]["val/kinematic_hard_ce"] for e in epochs], color="#c55835",label="Validation hard CE",lw=2)
    ax.set_title("Shared-depth ForeMotion C" if name=="foremotion" else "Shared-depth teacher C")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Face-weighted depth hard CE")
    ax.set_ylim(2,16)
    ax.grid(axis="y",color="#dddddd",lw=.6)
    ax.legend(frameon=False,fontsize=9)
fig.suptitle("Training keeps improving while validation deteriorates", fontsize=17, weight="bold",x=.07,ha="left",y=.98)
fig.text(.07,.022,"Compare trends, not the absolute gap: training uses stochastic RVQ prefixes/targets and dropout; validation uses canonical codes.",fontsize=9,color="#555555")
fig.subplots_adjust(top=.80,bottom=.15,left=.07,right=.98,wspace=.20)
fig.savefig(ROOT / "shared_depth_train_validation.png",dpi=160)
plt.close(fig)

print(json.dumps({"pairs": result["pairs"], "runs": {rid: {k:v for k,v in row.items() if k not in ("metrics","shared_teacher_latest")} for rid,row in result["runs"].items()}}, indent=2))
