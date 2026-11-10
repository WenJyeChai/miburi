"""Reproduce the codec-swap comparison from the read-only W&B snapshot."""
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE_AUDIT = Path('C:/Users/WenJChai/Downloads/codec_audit_20260910_072514_524748')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAIRS = [('ForeMotion', '59fnmnjg', 'u2zs5s3p'), ('Teacher C', 'lqd1cmjh', 'rmyldw6a')]
CODECS = {'upper': ('wg7n0bze', 990), 'lower': ('akld8gc6', 970), 'face': ('j4taovlj', 1000)}
RUNS = {r['name']: r for r in json.loads((ROOT / 'runs.json').read_text())}
IDS = [rid for _, old, new in PAIRS for rid in (old, new)] + [v[0] for v in CODECS.values()]
HISTORY = {rid: json.loads((ROOT / f'{rid}_history.json').read_text()) for rid in IDS}
CONFIG = {rid: {k: v.get('value') if isinstance(v, dict) else v
                for k, v in json.loads(RUNS[rid]['config']).items()} for rid in IDS}
VAL = {rid: {int(r['trainer/epoch']): r for r in HISTORY[rid] if 'val/ce_loss' in r}
       for _, a, b in PAIRS for rid in (a, b)}
URL = 'https://wandb.ai/wenjye00-hong-kong-university-of-science-and-technology/miburi_single/runs/'

def change(old, new):
    return dict(old=old, new=new, difference=new-old, percent=100*(new/old-1))

def best(rid, metric='val/ce_loss'):
    return min(VAL[rid].values(), key=lambda r: r[metric])

result = {'pairs': {}, 'codecs': {}, 'reconstruction_comparison': {}, 'integrity': {}}
for rid in IDS:
    rows = HISTORY[rid]
    assert [r['_step'] for r in rows] == list(range(len(rows)))
    result['integrity'][rid] = dict(rows=len(rows), state=RUNS[rid]['state'],
        validation_rows=sum(any(k.startswith('val/') for k in r) for r in rows),
        eval_metrics=sorted({k for r in rows for k in r if k.startswith('eval/')}))
for label, old, new in PAIRS:
    assert set(VAL[old]) == set(VAL[new]) == set(range(20, 1001, 20))
    a, b = VAL[old][1000], VAL[new][1000]
    total_delta = b['val/ce_loss'] - a['val/ce_loss']
    details = dict(old_id=old, new_id=new,
        config_differences={k: [CONFIG[old].get(k), CONFIG[new].get(k)]
          for k in sorted(set(CONFIG[old]) | set(CONFIG[new]))
          if k != '_wandb' and CONFIG[old].get(k) != CONFIG[new].get(k)},
        final={k: change(a[k], b[k]) for k in [
            'epoch_train/ce_loss', 'val/ce_loss', 'epoch_train/temporal_q0_ce',
            'val/temporal_q0_ce', 'val/temporal_q0_acc', 'epoch_train/kinematic_hard_ce',
            'val/kinematic_hard_ce', 'epoch_train/rvq_target_entropy',
            'epoch_train/rvq_sampled_nonnearest', 'epoch_train/rvq_changed_from_deterministic']},
        best=change(best(old)['val/ce_loss'], best(new)['val/ce_loss']),
        best_epochs=[best(old)['trainer/epoch'], best(new)['trainer/epoch']],
        deterioration_percent=[100*(VAL[rid][1000]['val/ce_loss']/best(rid)['val/ce_loss']-1)
                               for rid in (old, new)],
        body_parts={part: change(a[f'val/ce_{part}'], b[f'val/ce_{part}']) |
            {'fraction_of_total_increase': (b[f'val/ce_{part}']-a[f'val/ce_{part}'])/total_delta}
            for part in ('upper','lower','face')},
        diagnostic_rows={rid: {str(e): VAL[rid][e] for e in sorted({20,100,int(best(rid)['trainer/epoch']),980,1000})}
                         for rid in (old,new)})
    assert math.isclose(sum(v['difference'] for v in details['body_parts'].values()), total_delta, abs_tol=1e-5)
    result['pairs'][label] = details
    for rid in (old,new):
        assert CONFIG[rid]['is_continue'] is False
        summary = json.loads(RUNS[rid]['summaryMetrics'])
        assert (summary['train_samples'],summary['val_samples'],summary['test_samples']) == (477,54,89)
        assert summary['trainable_parameters'] == 320812552
        # Verify matched learning-rate histories, not merely matching config labels.
    rates = [{int(r['trainer/epoch']):r['train/learning_rate'] for r in HISTORY[rid]
              if 'train/learning_rate' in r} for rid in (old,new)]
    assert rates[0] == rates[1]

for part,(rid,selected_epoch) in CODECS.items():
    vals = {int(r['trainer/epoch']):r for r in HISTORY[rid] if 'val/codec_standard/clip_count' in r}
    row = vals[selected_epoch]
    assert row['val/codec_standard/clip_count'] == 54
    assert row['val/codec_standard/frame_count'] == 13500
    assert vals[max(vals)]['val/codec_standard/best_reconstruction_epoch'] == selected_epoch
    result['codecs'][part] = dict(run=rid,state=RUNS[rid]['state'],selected_epoch=selected_epoch,
        configured_epochs=CONFIG[rid]['epochs'], last_logged_epoch=max(r.get('trainer/epoch',-1) for r in HISTORY[rid]),
        last_validation_epoch=max(vals), metrics={k.removeprefix('val/codec_standard/'):v for k,v in row.items()
                                               if k.startswith('val/codec_standard/')})

with (BASELINE_AUDIT/'reconstruction_summary.csv').open(newline='',encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
matches = [('Expression RMSE','face','expression_rmse','expression_rmse'),
           ('Contact RMSE','lower','contact_rmse','contact_rmse'),
           ('Direct root-velocity RMSE (m/s)','lower','translation_velocity_direct_rmse_m_s','translation_velocity_rmse')]
for label,part,old_key,new_key in matches:
    row, = [r for r in rows if (r['split'],r['part'],r['metric_scope'],r['metric']) == ('val',part,'all_frames',old_key)]
    assert int(row['n_samples'])==54 and row['mode']=='streaming'
    result['reconstruction_comparison'][label] = change(float(row['value']), result['codecs'][part]['metrics'][new_key])

(ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
old_color,new_color='#267a84','#ba503a'
fig,axes=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
for col,(label,old,new) in enumerate(PAIRS):
    ax=axes[0,col]
    for rid,color,codec in [(old,old_color,'Released codecs'),(new,new_color,'Scott codecs')]:
        epochs=sorted(VAL[rid]); rs=[VAL[rid][e] for e in epochs]
        ax.plot(epochs,[r['val/ce_loss'] for r in rs],color=color,lw=2,label=codec+' validation')
        ax.plot(epochs,[r['epoch_train/ce_loss'] for r in rs],color=color,lw=1.5,ls='--',label=codec+' training')
        r=best(rid);ax.scatter(r['trainer/epoch'],r['val/ce_loss'],color=color,s=27,zorder=5)
    ax.set(title=label+': total CE',ylabel='Logged CE',xlabel='Epoch',xlim=(0,1000))
    ax.legend(fontsize=8,loc='lower left');ax.grid(alpha=.18)
ax=axes[1,0]
parts=['Upper + hands','Lower + root/contact','Face']; x=list(range(3)); width=.35
for offset,(label,old,new) in zip([-.5,.5],PAIRS):
    deltas=[result['pairs'][label]['body_parts'][p]['difference'] for p in ('upper','lower','face')]
    bars=ax.bar([i+offset*width for i in x],deltas,width,label=label,color='#407e9a' if label=='ForeMotion' else '#96605d')
    ax.bar_label(bars,fmt='%+.3f',fontsize=9,padding=3)
ax.axhline(0,color='#333333',lw=.8);ax.set_xticks(x,parts,fontsize=9)
ax.set(title='Most of the validation increase is lower-body CE',ylabel='New minus old CE contribution, epoch 1000')
ax.legend(fontsize=9);ax.grid(axis='y',alpha=.18);ax.set_ylim(-.12,.86)
ax=axes[1,1]
for rid,color,codec in [(PAIRS[0][1],old_color,'Released codecs'),(PAIRS[0][2],new_color,'Scott codecs')]:
    epochs=sorted(VAL[rid]);rs=[VAL[rid][e] for e in epochs]
    ax.plot(epochs,[r['val/temporal_q0_ce'] for r in rs],color=color,lw=2,label=codec+' validation')
    ax.plot(epochs,[r['epoch_train/temporal_q0_ce'] for r in rs],color=color,lw=1.5,ls='--',label=codec+' training')
ax.set(title='ForeMotion q0: training gain does not transfer',xlabel='Epoch',ylabel='q0 CE',xlim=(0,1000))
ax.grid(alpha=.18)
fig.suptitle('Codec retraining comparison',fontsize=16,fontweight='bold')
fig.supxlabel('Training uses stochastic depth targets; validation uses canonical targets. Code vocabularies differ across codecs.',fontsize=9)
fig.savefig(ROOT/'comparison.png',dpi=170)
fig.savefig(ROOT/'comparison.svg')
plt.close(fig)

lines=['# Scott codec retraining: what changed in the results','',
'The new codecs produce worse logged gesture-model validation CE. Overall training CE is also worse; the improvements are in selected metrics, especially temporal q0 and facial training losses. Lower-body tokens account for most of the validation increase. CE across independently trained vocabularies does not directly rank decoded gesture quality.','',
'## Runs and comparability','',
'Complete W&B histories were retrieved on 2026-09-10, with no missing logged steps. Each gesture run has 50 validation points, ends at epoch 1000, and starts fresh. All four have 320,812,552 trainable parameters, batch 64, seed 2342, 477 training / 54 validation / 89 test clips, face weight 1, independent depth weights, and matching cosine learning-rate histories (start 200, end 800). The behavioral config difference is the three codec checkpoints; two newly registered codec-evaluation flags are false and inactive. Remote source revisions and cache contents have not been independently hashed.','',
'| Model | Released-codec run | Scott-codec run |','| --- | --- | --- |']
for label,old,new in PAIRS:
    lines.append(f'| {label} | [{RUNS[old]["displayName"]}]({URL+old}) | [{RUNS[new]["displayName"]}]({URL+new}) |')
lines += ['', 'Epoch 1000 training meters summarize the preceding final training epoch (999) for all four runs. Training CE mixes hard and soft depth targets with stochastic RVQ prefixes; validation CE scores canonical hard targets. The absolute train–validation difference is therefore not a controlled generalization-gap estimate.','',
'## Overall CE','', '| Model | Final training CE, old → new | Best validation CE, old → new | Final validation CE, old → new |',
'| --- | ---: | ---: | ---: |']
for label,old,new in PAIRS:
    d=result['pairs'][label];t=d['final']['epoch_train/ce_loss'];v=d['final']['val/ce_loss'];b=d['best'];e=d['best_epochs']
    lines.append(f'| {label} | {t["old"]:.4f} → {t["new"]:.4f} ({t["percent"]:+.1f}%) | {b["old"]:.4f} @ {e[0]} → {b["new"]:.4f} @ {e[1]} ({b["percent"]:+.1f}%) | {v["old"]:.4f} → {v["new"]:.4f} ({v["percent"]:+.1f}%) |')
lines += ['', '![Comparison](comparison.png)','',
'## Where the validation difference comes from','',
'The part scores below are contributions to total CE (summed head losses divided by 20). They sum to overall CE. Upper/lower each contain eight heads and face four; these are not equal-size per-part averages.','',
'| Model | Upper contribution change | Lower contribution change | Face contribution change | Lower share of total increase |',
'| --- | ---: | ---: | ---: | ---: |']
for label,d in result['pairs'].items():
    parts=d['body_parts'];lines.append(f'| {label} | {parts["upper"]["difference"]:+.4f} | {parts["lower"]["difference"]:+.4f} | {parts["face"]["difference"]:+.4f} | {parts["lower"]["fraction_of_total_increase"]:.1%} |')
lines += ['', 'ForeMotion face validation is almost unchanged; the new teacher face validation is slightly better at epoch 1000. A face-only explanation does not fit this comparison.','',
'## Better q0 training is a limited improvement','', '| Model | Training q0 CE, old → new | Validation q0 CE, old → new | Validation q0 accuracy, old → new |',
'| --- | ---: | ---: | ---: |']
for label,d in result['pairs'].items():
    a,b,c=[d['final'][k] for k in ('epoch_train/temporal_q0_ce','val/temporal_q0_ce','val/temporal_q0_acc')]
    lines.append(f'| {label} | {a["old"]:.3f} → {a["new"]:.3f} | {b["old"]:.3f} → {b["new"]:.3f} | {c["old"]:.2%} → {c["new"]:.2%} |')
lines += ['', 'q0 is the first upper-body code, not a summary of all temporal features. The new model fits those training labels more tightly, but this does not transfer to held-out labels. Its q0 validation still improves during training; these observations alone do not prove earlier temporal overfitting.','',
'## Codec reconstruction versus the previous ground-truth audit','',
'The selected checkpoints are upper epoch 990, lower 970, face 1000. These match the best logged selection metric for each codec through epoch 1000, so the filenames `last_*` do not mean an arbitrary worse checkpoint was chosen. W&B marks the three long codec runs crashed around epochs 1001–1005 despite a configured 2000-epoch budget; the state alone does not distinguish a manual stop from a runtime failure.','',
'The previous export and new codec validation both use 54 clips / 13,500 frames, complete-clip streaming, all frames, and no warmup removal. The following feature metrics have matching definitions. Exact remote clip IDs and cache hashes should be checked in a paired rerun before treating this as a fully controlled benchmark.','',
'| Reconstruction metric | Released codec | Selected Scott codec | Relative change |', '| --- | ---: | ---: | ---: |']
for label,d in result['reconstruction_comparison'].items():
    lines.append(f'| {label} | {d["old"]:.6f} | {d["new"]:.6f} | {d["percent"]:+.1f}% |')
lines += ['', f'Previous measurements: `{BASELINE_AUDIT.as_posix()}/reconstruction_summary.csv`, filtered to validation, all_frames and streaming. New measurements: the selected epochs in the [lower codec run]({URL+CODECS["lower"][0]}) and [face codec run]({URL+CODECS["face"][0]}).',
'', 'Do not compare old geodesic rotation error or MPJPE with new raw rotation-6D MSE. No matching new geometry metrics are logged, and none of the four gesture runs logs generated-motion eval metrics.','',
'## Interpretation','',
'1. **The learned targets changed.** An independently trained RVQ divides motion into different categories and residuals. Same vocabulary size does not imply the same entropy or predictability from speech/history. Smaller q0 training CE can coexist with worse held-out CE. Token CE is not a cross-codec motion-quality score.','',
'2. **The Scott codecs have not demonstrated better reconstruction.** Contact and expression errors are worse in the available comparable measurements. Their reconstruction curves generally keep improving toward epoch 1000; this is not evidence that the codec itself overfit earlier. Scratch training on 477 clips replaces a codec trained on the broader full-BEATX data distribution. The new run also uses FP32 latents whereas the saved release training recipes used BF16.','',
'3. **The lower-code representation is the first suspect.** Most added CE comes from lower-body tokens in both ForeMotion and supervised teacher, which share the new codecs but differ in regret supervision. Harder/unfamiliar lower token sequences or codec-specific fitting to the training set are plausible; code usage, per-stage entropy and token transitions are needed to distinguish them. Contact reconstruction error does not by itself prove the cause of the lower-token CE increase.','',
'4. **A simple stronger-overfitting claim is incomplete.** The new best CE is already higher. From best to final, ForeMotion deteriorates {:.2f}% with old codecs versus {:.2f}% with new; teacher {:.2f}% versus {:.2f}%. Teacher depth still shows within-run overfitting, but the new post-best deterioration is smaller. Earlier stopping alone cannot recover the previous best CE.'.format(*result['pairs']['ForeMotion']['deterioration_percent'],*result['pairs']['Teacher C']['deterioration_percent']),
'', '5. **The measured RVQ perturbation is similar.** Average top-k target entropy is about 1.610 versus 1.614 nats, sampled non-nearest fraction about 8.4% versus 8.2%, and changed-code fraction about 25.9% versus 26.1%. This is local candidate entropy, not global code-usage entropy. These diagnostics do not support a large accidental increase in augmentation strength.','',
'The logged yaw flag differs between codec and gesture configs, but current BaseCausalCodecTrainer and BaseGLMTrainer do not pass it to UNIFIEDDataset; its constructor default is false. The cache builder already aligns each chunk. Thus the config flag alone is not evidence of a preprocessing mismatch. Likewise motion tokens are encoded from motion online, so rebuilding the existing motion/speech cache is not the missing step.','',
'## Next evaluation','',
'First run the same reconstruction audit on both codec sets with exactly the same validation manifest, then test after selecting the codec configuration. Include geodesic error, MPJPE, integrated root error, contact/expression error, and per-stage usage/entropy/unseen-token mass. A canonical teacher-forced pass on both train and validation would make the within-codec token generalization gap interpretable.','',
'For the first downstream ablation, keep the original upper/face codecs and replace only lower. Compare it with the existing original-codec baseline and the all-new run. This isolates the largest contributor without replacing all vocabularies simultaneously. Evaluate autoregressively generated motion from each run’s validation-selected checkpoint with the same FGD/geometry/protocol, rather than judging by CE alone. If single-speaker adaptation is still desired, fine-tuning released codecs is a separate, useful comparison to training them from scratch.','',
'No training, checkpoint or configuration files were changed during this analysis.']
(ROOT/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k: {kk:vv for kk,vv in v.items() if kk in ('final','best','best_epochs','deterioration_percent','body_parts')}
                  for k,v in result['pairs'].items()},indent=2))
print(json.dumps(result['reconstruction_comparison'],indent=2))
