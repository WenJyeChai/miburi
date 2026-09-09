"""Inspect existing W&B per-body validation metrics without retraining."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {"7so4vnve": "Shared-depth ForeMotion C", "2ww4lkuq": "Shared-depth teacher C"}
PARTS = (("upper", "Upper body + hands", 8, "#216c91"),
         ("lower", "Lower body + translation", 8, "#529345"),
         ("face", "Face", 4, "#c45438"))
result = {}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "axes.spines.top":False, "axes.spines.right":False,
                     "axes.titleweight":"bold", "figure.facecolor":"#fcfcfa",
                     "axes.facecolor":"#fcfcfa"})
fig, axes = plt.subplots(1,2,figsize=(12,5.2),sharey=True)
for ax,(rid,label) in zip(axes,RUNS.items()):
    rows=json.loads((ROOT/f"{rid}_history.json").read_text())
    vals={r["trainer/epoch"]:r for r in rows if "val/ce_loss" in r}
    end=max(vals)
    result[rid]={"name":label,"latest_epoch":end,"parts":{}}
    upper_key="val/kinematic_upper_hard_ce"
    upper_best=min(vals,key=lambda e:vals[e][upper_key])
    result[rid]["upper_depth_only"]={
        "metric":upper_key,"best_logged_ce":vals[upper_best][upper_key],
        "tied_best_epochs":[e for e in vals if vals[e][upper_key]==vals[upper_best][upper_key]],
        "latest_logged_ce":vals[end][upper_key],
        "increase_percent":100*(vals[end][upper_key]/vals[upper_best][upper_key]-1),
    }
    for index,(part,title,count,color) in enumerate(PARTS):
        key=f"val/ce_{part}"
        best_epoch=min(vals,key=lambda e:vals[e][key])
        best=vals[best_epoch][key]
        latest=vals[end][key]
        epochs=sorted(vals)
        normalized=[vals[e][key]/best for e in epochs]
        ax.plot(epochs,normalized,color=color,lw=2,label=title)
        ax.scatter(best_epoch,1,color=color,s=32,zorder=5)
        # Upper/lower finish close together in ForeMotion; separate labels.
        offset=(-13 if part=="upper" else 6) if rid=="7so4vnve" and part!="face" else 0
        ax.annotate(f"+{100*(latest/best-1):.1f}%",(end,latest/best),
                    xytext=(7,offset),textcoords="offset points",color=color,va="center",fontsize=10)
        result[rid]["parts"][part]={"metric":key,"codebooks":count,"best_epoch":best_epoch,
            "best_logged_ce":best,"latest_logged_ce":latest,"increase_percent":100*(latest/best-1),
            "best_mean_per_codebook_ce":best*20/count,"latest_mean_per_codebook_ce":latest*20/count}
    ax.axhline(1,color="#999999",ls=":",lw=1)
    ax.set_title(label,pad=12)
    ax.set_xlabel("Epoch")
    ax.set_xlim(0,end*1.18)
    ax.set_ylim(.87,2.72)
    ax.grid(axis="y",color="#dddddd",lw=.6)
    minima=result[rid]["parts"]
    ax.text(0,-.22,f"Minimum epochs: upper {minima['upper']['best_epoch']} | lower {minima['lower']['best_epoch']} | face {minima['face']['best_epoch']}",
            transform=ax.transAxes,fontsize=9,color="#555555")
axes[0].set_ylabel("Validation CE / that body's own minimum")
handles,labels=axes[0].get_legend_handles_labels()
fig.suptitle("Face validation deteriorates first and most strongly",x=.07,ha="left",fontsize=18,weight="bold",y=.98)
fig.legend(handles,labels,loc="upper left",bbox_to_anchor=(.06,.91),frameon=False,ncol=3,fontsize=10)
fig.text(.07,.018,"Existing val/ce_upper, val/ce_lower and val/ce_face; no smoothing. 1.0 = each curve's own best. Upper includes temporal q0.",fontsize=9,color="#555555")
fig.subplots_adjust(top=.77,bottom=.22,left=.07,right=.98,wspace=.14)
fig.savefig(ROOT/"body_part_overfitting.png",dpi=170)
fig.savefig(ROOT/"body_part_overfitting.svg")
(ROOT/"body_part_comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
