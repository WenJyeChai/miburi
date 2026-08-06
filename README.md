# MIBURI

Implementation for **MIBURI: Towards Expressive Interactive Gesture Synthesis** (CVPR 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2603.03282-b31b1b.svg)](https://arxiv.org/abs/2603.03282)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://vcai.mpi-inf.mpg.de/projects/MIBURI/)

![MIBURI Demo](assets/MIBURI_TEASER.png)

MIBURI is a real-time, dialogue-driven full-body gesture + facial-expression synthesis system. A frozen Moshi 7B LM supplies text + audio embeddings that condition a custom Gesture LM (`GTemporalDepthModel3`), which autoregressively emits multi-codebook gesture tokens. Three streaming RVQ-style codecs (upper+hands / lower+trans / face+exp) decode those tokens into SMPL-X motion at 25 fps, rendered live in a Viser viewer alongside Moshi's spoken reply.

---
### Citation

```bibtex
@InProceedings{mughal2026miburi,
	title = {MIBURI: Towards Expressive Interactive Gesture Synthesis},
	author = {M. Hamza Mughal and Rishabh Dabral and Vera Demberg and Christian Theobalt},
	booktitle={Computer Vision and Pattern Recognition (CVPR)},
	year={2026}
}
```

---

## Installation

MIBURI targets Python 3.12. The installation is two steps: pick the right **PyTorch** build for your CUDA driver, then install the package itself with all the other dependencies.

```bash
conda create --name miburi python=3.12 -y
conda activate miburi
```

### 1. Install PyTorch (CUDA-specific)

PyTorch is intentionally **not** declared as a project dependency — you pick the build that matches your CUDA driver:

```bash
# CUDA 13 (Hopper / Blackwell)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130

# CUDA 12 (Ampere / Ada / etc.)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
```

### 2. Install MIBURI + dependencies

```bash
pip install .             # editable: pip install -e .
```

`pip install .` reads [`pyproject.toml`](pyproject.toml) and installs everything else (Moshi runtime, SMPL-X, Viser, the renderer stack, etc.). A convenience [`requirements.txt`](requirements.txt) mirrors the same list for installing dependencies without the package itself.

### 3. Download release assets + checkpoints

The MIBURI demo needs two things that aren't in this Git repo: a small bundle of **demo assets** (SMPL-X model + Mixamo characters + Moshi UI frontend, ~200 MB) and the **trained checkpoints** (~1.7 GB). Both ship from Hugging Face:

```bash
miburi-download-assets         # ~200 MB into assets_dep/
miburi-download-checkpoints    # ~1.7 GB into experiments/
```

Both commands are no-ops on re-run once the files are cached locally.

> ⚠️ **Licensing.** By downloading these you agree to honor the upstream licenses:
> - **SMPL-X** (`assets_dep/smplx_2020/`) — Max Planck Society, non-commercial use only. See [the SMPL-X model license](https://smpl-x.is.tue.mpg.de/modellicense.html).
> - **Moshi 7B / Mimi** (fetched automatically from [`kyutai/moshiko-pytorch-bf16`](https://huggingface.co/kyutai/moshiko-pytorch-bf16) on first demo run) — CC-BY 4.0 (Kyutai).
> - **Mixamo characters** (`assets_dep/mixamo_characters_release/`) — Adobe [Mixamo terms of use](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html).
> - **MIBURI weights** (`experiments/*release*/`) — CC-BY-NC-4.0; please cite the paper when using.

<details>
<summary><strong>Details on release assets</strong></summary>

Three subtrees pulled from [`m-hamza-mughal/miburi-release-assets`](https://huggingface.co/m-hamza-mughal/miburi-release-assets):

| Path | Role | Size |
| --- | --- | ---: |
| `assets_dep/demo-static/` | Embedded Moshi audio UI (frontend assets the demo server statically serves) | ~1 MB |
| `assets_dep/mixamo_characters_release/` | 1 Mixamo character `.npz` bundle (`y_bot`) for `--mixamo-character y_bot` | ~1 MB |
| `assets_dep/smplx_2020/` | SMPL-X NEUTRAL 2020 model `.npz` used for vertex decoding | ~115 MB |

Override with `miburi-download-assets --local-dir <other> --repo-id <namespace>/<repo>`.

</details>

<details>
<summary><strong>Details on checkpoints</strong></summary>

8 directories pulled from [`m-hamza-mughal/miburi-release-checkpoints`](https://huggingface.co/m-hamza-mughal/miburi-release-checkpoints) — 4 codecs + 1 Gesture LM for each of the two release tracks (the "good-speaker" demo subset, and the 23-speaker "allspk" variant):

| Released dir | Role | Size |
| --- | --- | ---: |
| `demoexp_release_uppercodec/` | Upper body + hands codec (BEATX, 8 codebooks) | 106 MB |
| `demoexp_release_lowercodec/` | Lower body + trans-velocity codec | 103 MB |
| `demoexp_release_facecodec/` | Face / expression codec | 59 MB |
| `demoexp_release_gtdm3_goodspk/` | Gesture LM (GTDM3), 4-speaker BEATX subset | 590 MB |
| `allspk_release_uppercodec/` | Upper codec, all-speaker variant | 106 MB |
| `allspk_release_lowertranscodec/` | Lower codec, all-speaker variant | 103 MB |
| `allspk_release_facecodec/` | Face codec, all-speaker variant | 60 MB |
| `allspk_release_gtdm3_exp/` | Gesture LM (GTDM3), all 23 BEATX speakers | 590 MB |

Override with `miburi-download-checkpoints --local-dir <other> --repo-id <namespace>/<repo>`.

</details>

---

## Running the demo

The realtime server (`miburi/gest-server.py`) opens an aiohttp dashboard at `http://<host>:8998` with an embedded Moshi audio UI and a Viser motion viewer side by side.

### Convenience launcher: `miburi-demo`

[`scripts/miburi-demo`](scripts/miburi-demo) cds into the repo, sources conda, activates the env, and runs `gest-server` with the canonical config. Demo-time flags are forwarded straight through.

Install it once as a shell command (user-local, no sudo). From inside the repo root:

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/scripts/miburi-demo" ~/.local/bin/miburi-demo
```

If `~/.local/bin` is not already on your `PATH`, add this once to `~/.bashrc` / `~/.zshrc`:

```bash
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH";; esac
```

Verify (open a new shell or `source ~/.bashrc` first):

```bash
which miburi-demo                                        # prints the symlink path
miburi-demo --help                                       # shows gest-server's args
```

You can perform same steps for 23-spk experiment `scripts/miburi-demo-allspk`

### Usage

```bash
miburi-demo                                              # default demo
miburi-demo --minimal-audio-ui                           # collapse the audio pane (only mic visible)
miburi-demo --mixamo-character y_bot                     # render with a Mixamo character bundle
miburi-demo --minimal-audio-ui --mixamo-character y_bot
```

If your env is named something other than `miburi` (e.g. `miburi_cu12`), override per call: `MIBURI_ENV=miburi_cu12 miburi-demo`.

### Direct invocation (no launcher)

```bash
python -m miburi.gest-server \
    --glm-config experiments/demoexp_release_gtdm3_goodspk/config.yaml \
    --glm-cfg-coef 1.3
```

---

## Training

### Step 0 — Download the BEATX dataset (optional, training only)

If you don't already have a BEAT2-English checkout on disk, fetch the MIBURI-flavoured fork (BEAT2 + the MIBURI/RAG-Gesture annotations the pipeline needs) into `datasets/beat_english_v2.0.0/`:

```bash
miburi-download-beatx-dataset                    # ~16.6 GB into datasets/beat_english_v2.0.0/
```

This pulls 4 subdirs from [`m-hamza-mughal/beat2-additional-annotations`](https://huggingface.co/datasets/m-hamza-mughal/beat2-additional-annotations) (a fork of [H-Liu1997/BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2)): `smplxflame_25/` (25 fps motion, MIBURI), `whisper_transcription/` (transcripts with casing+punctuation, MIBURI), `wave16k/` (audio, BEAT2 inherited), `weights/` (CNN motion autoencoder weights for the FGD metric, BEAT2 inherited).

The **demo path does not need this** — it's only required for training MIBURI from scratch or building the eval HDF5 cache for `scripts/test.py`.

### Step 1 — Build per-dataset HDF5 caches

MIBURI ships HDF5 builders for two source datasets: **BEATX (BEAT2 English)** and **embody3d_dyadic**. The training-time loader is **unified** — it reads one or both HDF5s and mixes them according to `dataset_ratio` in the config. So the workflow is "build whichever datasets you have, then run training with a unified config."

**BEATX** — point `--data_dir` at a BEAT2-English checkout that has `smplxflame_25/`, `wave16k/`, `whisper_transcription/` subdirs and a `train_test_split.csv`:

```bash
python -m scripts.trainers.dataloaders.beatx.build_hdf5_beatx \
    --data_dir datasets/beat_english_v2.0.0/ \
    --file_list_path datasets/beat_english_v2.0.0/train_test_split.csv \
    --hdf5_path datasets/data_cache/beatx_train/database.hdf5 \
    --motion_fps 25 \
    --pose_length 250
```

**embody3d_dyadic** — point `--data_dir` at the Embody3D AIAgent checkout (per-speaker subdirs containing `audio_separated/`, motion `.npz`s, etc.):

```bash
python -m scripts.trainers.dataloaders.embody3d_dyadic.build_hdf5_embody3d \
    --data_dir datasets/embody_3d_aiagent/ \
    --hdf5_path datasets/data_cache/embody3d_train/database.hdf5 \
    --motion_fps 25 \
    --pose_length 250
```

The unified loader picks these HDF5s up automatically when a training config sets `dataset: unified` at the matching `dataset_ratio` (e.g. `full_beatx`, `33embody_66beatx`, `goodspk_beatx_lowervalid`). The trainer reads the HDF5 paths from `cache_path` in the config — point that at the directory holding the per-dataset `database.hdf5` files.

#### Evaluation HDF5 (full-sequence variant)

`scripts/test.py` operates on **full sequences**, not the fixed-length `pose_length` chunks that training uses. Build a separate HDF5 with `--full_sequence` (and `--process_multimodal_signals`, which extracts the audio / transcript signals the metric loops consume) and point your eval-time config's `cache_path` at this directory instead:

```bash
# BEATX evaluation HDF5:
python -m scripts.trainers.dataloaders.beatx.build_hdf5_beatx \
    --data_dir datasets/beat_english_v2.0.0/ \
    --file_list_path datasets/beat_english_v2.0.0/train_test_split.csv \
    --hdf5_path datasets/data_cache/beatx_eval/database.hdf5 \
    --full_sequence \
    --motion_fps 25 \
    --pose_length 250 \
    --process_multimodal_signals

# Embody3D evaluation HDF5 (same idea, no --file_list_path needed):
python -m scripts.trainers.dataloaders.embody3d_dyadic.build_hdf5_embody3d \
    --data_dir datasets/embody_3d_aiagent/ \
    --hdf5_path datasets/data_cache/embody3d_eval/database.hdf5 \
    --full_sequence \
    --motion_fps 25 \
    --pose_length 250 \
    --process_multimodal_signals
```

`--full_sequence` keeps each source file as one HDF5 entry (no `_C<chunk>` suffixes), `--process_multimodal_signals` pre-computes the audio + text alignment payloads, and `--pose_length 250` only acts as a minimum-length filter in this mode rather than a chunk size. The released demo checkpoints' YAMLs ship with `dataset_ratio` values ending in `_eval` / `_fulllength` (e.g. `full_beatx_eval`, `full_beatx_fulllength`) — those expect the eval HDF5 layout.

### Step 2 — Train

Single-GPU baseline:

```bash
python scripts/train.py --config configs/<your-config>.yaml
```

Multi-GPU via `torchrun` (the standalone DDP launcher MIBURI's trainers expect):

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train.py --config configs/<your-config>.yaml --batch_size 4 --loader_workers 32 --ddp True
```

#### VQ-VAE codec training

Three independent codecs operate on disjoint SMPL-X joint subsets. Each has its own canonical config under [`configs/`](configs/):

```bash
# Upper body + hands (43 joints, 8 codebooks):
python scripts/train.py --config configs/mimi_causalrvq_upperhands_25_smplxbeatxembody3d_fchunksize2.yaml

# Lower body + trans-velocity (9 joints + 3 translation, 8 codebooks):
python scripts/train.py --config configs/mimi_causalrvq_lowertransvel_25_smplxbeatx_fchunksize2.yaml

# Face expressions + jaw/eye pose (1 joint + 100 expression coeffs, 4 codebooks):
python scripts/train.py --config configs/mimi_causalrvq_faceexp_25_smplxbeatx_fchunksize2.yaml
```

The three codecs must be trained first because the Gesture LM (next step) is conditioned on a frozen copy of their RVQ codebooks.

#### Gesture LM (GTDM3) training

After all three codecs are trained, point the LM training at their checkpoints. The codec paths are set inside the GTDM3 config (`upperbodycodec_ckpt`, `lowerbodycodec_ckpt`, `facecodec_ckpt`):

```bash
# BEATX-only small variant (the released demo checkpoint's training config):
python scripts/train.py --config configs/gtdm3_audiotext_ufl_25_beatx_small.yaml

# All-speaker BEATX variant:
python scripts/train.py --config configs/gtdm3_audiotext_ufl_25_beatx_allspk.yaml

# embody3d + BEATX mix:
python scripts/train.py --config configs/gtdm3_audiotext_ufl_25_embody3dbeatx_allspk.yaml
```

For multi-GPU, wrap any of these in the same `torchrun --standalone --nnodes=1 --nproc-per-node=N` line as above.

#### Experimental future-teacher sequence (T0/T1/T2)

The causal GTDM3 trainer is the T0 control. The existing masked full-condition
teacher is T1: it hides the target plus a 400 ms guard, then exposes future
tokens from the intact full-clip codec encoding.

T2 uses the same teacher architecture, audio/text condition, and selected-frame
objective, but replaces the revealed upper/lower/face future window with an
independent encoding of the complete remaining globally preprocessed raw-motion
suffix:

```bash
# Four-speaker good-speaker pilot: build the reset-code cache first.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python scripts/build_reset_future_cache.py \
    --config configs/gtdm3_resetfuture_fullcondition_oneq0_beatx_small.yaml

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python scripts/train.py \
    --config configs/gtdm3_resetfuture_fullcondition_oneq0_beatx_small.yaml \
    --wandb True --wandb_mode online \
    --wandb_project miburi --wandb_group t2-reset-future-goodspk
```

The good-speaker cache uses all 120 target positions that retain at least one
future token after the five-token guard in a 10-second clip. Each target stores
the full remaining suffix, from 120 tokens for the earliest target down to one
token for the latest. The builder length-buckets and right-pads independent
suffixes within each non-streaming causal-codec batch, then packs only the valid
`uint16` RVQ indices into sharded HDF5 files. Training chooses one manifest
target per clip per epoch and cycles all 120 without rerunning the codecs, so the training batch
size and epoch length remain 128 clips and match the original good-speaker
configuration. The all-speaker sparse-pilot config remains available as
`configs/gtdm3_resetfuture_fullcondition_oneq0_beatx_allspk.yaml`.

For the exhaustive all-speaker experiment matched to the released all-speaker
codecs and optimization schedule, first build its separate codec-fingerprinted
120-target cache, then train the privileged teacher:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python scripts/build_reset_future_cache.py \
    --config configs/gtdm3_resetfuture_fullcondition_oneq0_beatx_allspk_t120.yaml

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python scripts/train.py \
    --config configs/gtdm3_resetfuture_fullcondition_oneq0_beatx_allspk_t120.yaml \
    --wandb True --wandb_mode online \
    --wandb_project miburi --wandb_group t2-reset-future-allspk-t120
```

The causal all-speaker student reuses the same reset cache. Supply the newly
trained teacher checkpoint explicitly so a stale good-speaker or sparse-pilot
teacher cannot be selected accidentally:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python scripts/train.py \
    --config configs/gtdm3_resetregret_fullsuffix_t120_audiotext_ufl_25_beatx_allspk.yaml \
    --regret_teacher_ckpt experiments/<allspk-t120-teacher-run>/last_<epoch>.safetensors \
    --wandb True --wandb_mode online \
    --wandb_project miburi --wandb_group student-resetregret-allspk-t120
```

This exhaustive cache is distinct from both
`reset_future_goodspk_fullsuffix_t120` and the all-speaker `t8` sparse pilot.
The chunked `beatx_gtdm3/database.hdf5` source is reused; only the reset-code
cache must be rebuilt when the dataset selector or gesture codecs change.

The first reset token remains visible. Set `--reset_prefix_drop_tokens 1` for
the immediate boundary-artifact ablation; this does not require rebuilding the
cache. A learned reset-segment embedding is intentionally disabled in this
first architecture-matched comparison.

T1 and T2 now expose the same remaining-suffix boundary. Their privileged
motion differs only in whether those future codes come from the intact
full-clip codec state (T1) or a freshly reset codec state (T2).

Before launching a checkpoint run, execute the focused information-boundary
checks:

```bash
python scripts/smoke_test_gesture_lm_reset_future.py
```

These verify past/hidden-interval invariance, future sensitivity, codec-state
isolation, frame/token boundary alignment, lower-velocity boundary handling,
and replacement of all three body-part streams.

To distill the full-suffix good-speaker T2 checkpoint into the ordinary causal
student, use the matching full-suffix reset-regret config. The frozen teacher
is queried with the same variable-length `t120` reset suffixes used during
teacher training; the student never receives reset codes and retains the
released causal MIBURI inputs and losses:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python scripts/train.py \
    --config configs/gtdm3_resetregret_fullsuffix_t120_audiotext_ufl_25_beatx_small.yaml \
    --wandb True --wandb_mode online \
    --wandb_project miburi --wandb_group student-resetregret-fullsuffix-t120
```

The historical fixed-window experiment remains available as
`configs/gtdm3_resetregret_h16_t105_audiotext_ufl_25_beatx_small.yaml`; pair it
only with its original `h16_t105` teacher/cache. In either setting, teacher
checkpoint and reset cache must use the same five-token guard and suffix mode.
The student trains with causal CE from scratch; q0 forward-KL distillation
starts at epoch 50 and ramps from alpha 0.1 to 1.0 over 15 epochs.

#### Shared-weight GlobalRegret (no separate teacher run)

`UpperFaceLowerGTDM3SharedRegretTrainer` implements the dual-view mechanism
literally: the causal student and its future-aware teacher are the *same*
weights, differing only in the temporal transformer's audio/text
cross-attention mask (relaxed to bidirectional for one extra, gradient-free
forward pass per step). There is no `--regret_teacher_ckpt`, no
reset-future cache, and no prior teacher-pretraining run to manage. It also
distills every RVQ level rather than q0 alone, since the kinematic/depth
transformer is shared weights too and picks up a teacher-view distribution
for free by conditioning on the teacher's temporal hidden state instead of
the student's:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python scripts/train.py \
    --config configs/gtdm3_sharedregret_global_audiotext_ufl_25_beatx_small.yaml \
    --wandb True --wandb_mode online \
    --wandb_project miburi --wandb_group student-sharedregret-global
```

Only GlobalRegret (full-clip bidirectional cross-attention) is implemented;
LocalRegret's one-step lookahead is left as follow-up work since it needs an
actual attention-bias primitive rather than the `causal` boolean toggle this
trainer uses. Set `--regret_include_depth_levels False` to restrict
distillation to q0 only, matching the scope of the frozen-teacher trainer
above. `scripts/smoke_test_gesture_lm_shared_regret.py` checks the
cross-attention mask is toggled and restored correctly, that the teacher
forward pass is gradient-free, and that the resulting loss backpropagates
into both the temporal and depth transformers.

##### + the kinematic transformer's stochastic-RVQ training strategy

`UpperFaceLowerGTDM3SharedRegretRVQTrainer` additionally reuses the
kinematic/depth training strategy from `UpperFaceLowerGTDM3FrozenTemporalRVQTrainer`
(coherent stochastic RVQ input prefixes plus a hard/soft-CE mixture against a
distance-aware top-k target distribution for the depth levels) -- but without
freezing the temporal transformer. q0 stays an ordinary trainable head with
plain hard CE; only k=1..K-1 get the stochastic-RVQ treatment. Because the
depth transformer's input prefix is stochastic during training, the
GlobalRegret teacher's depth branch is conditioned on that same sampled
prefix (not the canonical codes) rather than the temporal branch's plain
`input_codes`, so the regret KL reflects future context only, not which RVQ
prefix variant was used; validation is unaffected since RVQ sampling is
train-only there too.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python scripts/train.py \
    --config configs/gtdm3_sharedregretrvq_global_audiotext_ufl_25_beatx_small.yaml \
    --wandb True --wandb_mode online \
    --wandb_project miburi --wandb_group student-sharedregretrvq-global
```

Reuses the existing `kinematic_rvq_*` args unchanged; no `--frozen_temporal_ckpt`
is needed or accepted. `scripts/smoke_test_gesture_lm_shared_regret.py` also
covers the composition-specific piece: that `depth_input_codes` only changes
the teacher's depth branch (never the temporal branch), and that a mismatched
shape is rejected.

#### Experimental three-q0 global-C2F Gesture LM

The opt-in C2F variant is separate from the released GTDM3 model and config.
It predicts upper/lower/face q0 tokens from three temporal heads, then predicts
the remaining 17 tokens in an interleaved global coarse-to-fine order. Like
original Miburi, its temporal backbone embeds the complete previous 20-token
frame. It supports paired detached temporal/kinematic self-forcing plus
SentiAvatar-style soft recovery.

```bash
python scripts/train.py \
    --config configs/gtdm3_c2f3q0_audiotext_ufl_25_beatx_allspk.yaml
```

The config initializes the Gesture LM from scratch (`is_continue: False`) but
still loads the frozen pretrained gesture codecs and initializes its audio/text
lookup embeddings from Moshi/Mimi. Its self-forcing schedule is expressed as a
fraction of `epochs` when `c2f_self_forcing_start_epoch` is negative. The
experimental config instead uses an absolute epoch-100 start and a 50-epoch
ramp so changing the total run horizon cannot silently move the curriculum.

For deterministic rollout evaluation matching the detached greedy
self-forcing policy, run `scripts/test.py` against a full-sequence evaluation
HDF5 with `--eval_generation_mode greedy_cfg1`. The default
`--eval_generation_mode production` retains temporal top-p 0.8, kinematic
top-p 0.95, temperature 0.9, and `--cfg_scale`.

True temporal rollout is sequential across all gesture frames, so self-forced
training batches are expected to be materially slower than teacher-forced
batches. W&B exposes overall and per-part `c2f_temporal_*_q0_acc` metrics for
the active self-forced batches.

#### Weights & Biases logging

W&B logging is optional and disabled by default. Install the tracking extra
and authenticate once:

```bash
pip install -e ".[tracking]"
wandb login
```

Enable it on any codec or GTDM3 training command with `--wandb True`.
Only global rank 0 creates a W&B run, including under DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/train.py \
    --config configs/gtdm3_audiotext_ufl_25_beatx_small.yaml \
    --ddp True --batch_size 4 --loader_workers 4 \
    --wandb True --wandb_project miburi \
    --wandb_group gtdm3-goodspk \
    --wandb_tags gtdm3 beatx 4xa800
```

Training runs log the configured loss meters, learning rate, global step,
epoch, data/train time, throughput, GPU reserved memory, global batch size,
parameter counts, and checkpoint metadata. Validation meters are reduced
across all DDP ranks before logging. Codec validation also logs MPJPE.
TensorBoard logging remains enabled.

Useful tracking options:

- `--wandb_name <name>` — override the timestamped experiment name.
- `--wandb_entity <team-or-user>` — select a W&B entity.
- `--wandb_mode offline` — keep the run local for later `wandb sync`.
- `--wandb_run_id <id> --wandb_resume allow` — continue an existing W&B run.

Standalone `scripts/test.py` runs accept the same W&B options and log
evaluation metrics such as FGD, MPJPE, facial errors, beat alignment,
diversity, inference time, and real-time factor where applicable.

---

## Evaluation

`scripts/test.py` runs a trainer's `test()` method against a saved checkpoint. It dispatches to `args.trainer + "Trainer"`, so the trainer field in the YAML decides which evaluation loop runs.

Common pattern:

```bash
python scripts/test.py  \
    --config experiments/<experiment-dir>/config.yaml \ --dataset_ratio full_beatx_fulllength \
    --cfg_scale 2.3 --batch_size 1 --beatx_cache_path datasets/data_cache/beatx_eval/database.hdf5
```

Useful overrides (all defined in [`scripts/trainers/utils/config.py`](scripts/trainers/utils/config.py)):

| Flag | Default | Effect |
| --- | --- | --- |
| `--visualize True` | `False` | Also render the per-sample side-by-side GT/Pred mp4 (face codec uses head-only camera framing). Off by default keeps the run metrics-only and fast. |
| `--save True` | `False` | Also write per-sample `gt.npz` / `pred.npz` (+ `upper_tokens.npz` for the GLM). Off by default so a metrics-only run leaves disk untouched. |
| `--max_batches N` | `None` | Process only the first N batches. Useful for spot-checking a new visualization or rerunning metrics on a subset. |
| `--beatx_cache_path <dir>` | from YAML | Point at the **eval** HDF5 cache built in [Step 1](#evaluation-hdf5-full-sequence-variant). The YAML carries the training cache path; override on the CLI so you don't have to maintain two YAMLs. |
| `--embody3d_cache_path <dir>` | from YAML | Same role for the Embody3D dataset. |

Typical eval invocation pointing at the full-sequence cache:

```bash
python scripts/test.py \
    --config experiments/demoexp_release_gtdm3_goodspk/config.yaml \
    --test_ckpt experiments/demoexp_release_gtdm3_goodspk/last_6500.safetensors \
    --beatx_cache_path datasets/data_cache/beatx_eval/
```

### Released demo checkpoints

8 directories under [`experiments/`](experiments/) (fetched by `miburi-download-checkpoints`, see [Install Step 3](#3-download-release-checkpoints)):

| Experiment | Trainer | Latest ckpt |
| --- | --- | --- |
| `demoexp_release_uppercodec/` | `UpperBodyCausalCodec` | `last_180.safetensors` |
| `demoexp_release_lowercodec/` | `LowerBodyCausalCodec` | `last_440.safetensors` |
| `demoexp_release_facecodec/`  | `FaceExpCausalCodec`   | `last_100.safetensors` |
| `demoexp_release_gtdm3_goodspk/` | `UpperFaceLowerGTDM3` | `last_6500.safetensors` |
| `allspk_release_uppercodec/` | `UpperBodyCausalCodec` | `last_490.safetensors` |
| `allspk_release_lowertranscodec/` | `LowerBodyCausalCodec` | `last_705.safetensors` |
| `allspk_release_facecodec/`  | `FaceExpCausalCodec`   | `last_380.safetensors` |
| `allspk_release_gtdm3_exp/` | `UpperFaceLowerGTDM3` | `last_1720.safetensors` |

### Codec evaluation (per-sample GT vs Pred + ReconMetrics)

```bash
# Upper body + hands codec:
python scripts/test.py \
    --config experiments/demoexp_release_uppercodec/config.yaml \
    --test_ckpt experiments/demoexp_release_uppercodec/last_180.safetensors

# Lower body + trans-velocity codec:
python scripts/test.py \
    --config experiments/demoexp_release_lowercodec/config.yaml \
    --test_ckpt experiments/demoexp_release_lowercodec/last_440.safetensors

# Face codec (renders with a tight head-only camera when --visualize True):
python scripts/test.py \
    --config experiments/demoexp_release_facecodec/config.yaml \
    --test_ckpt experiments/demoexp_release_facecodec/last_100.safetensors

# Add visualization + small batch limit for spot-checks:
python scripts/test.py \
    --config experiments/demoexp_release_uppercodec/config.yaml \
    --test_ckpt experiments/demoexp_release_uppercodec/last_180.safetensors \
    --visualize True --max_batches 5
```

After the loop you'll get a `ReconMetrics` block in the logs: `fgd score`, `Reconstruction MPJPE`, `Facial L2`, `Facial L-Vel`.

### Gesture LM (GTDM3) evaluation

```bash
# good-speaker variant (the one the default demo uses):
python scripts/test.py \
    --config experiments/demoexp_release_gtdm3_goodspk/config.yaml \
    --test_ckpt experiments/demoexp_release_gtdm3_goodspk/last_6500.safetensors

# all-speaker variant (23 BEATX speakers):
python scripts/test.py \
    --config experiments/allspk_release_gtdm3_exp/config.yaml \
    --test_ckpt experiments/allspk_release_gtdm3_exp/last_1720.safetensors

# Spot-check a handful of samples with side-by-side video:
python scripts/test.py \
    --config experiments/demoexp_release_gtdm3_goodspk/config.yaml \
    --test_ckpt experiments/demoexp_release_gtdm3_goodspk/last_6500.safetensors \
    --visualize True --max_batches 5
```

`GestureMetrics` logs `Facial L2`, `Facial L-Vel`, `fgd score`, `align score`, `gt align score`, `L1div score`, `GT L1div score`. The metric handles both BEATX (`<beatx_data_path>/wave16k/<file_id>.wav`) and embody3d_dyadic (`<embody3d_path>/<smpid>/<spkid>/audio_separated/<smpid>.wav`) audio path layouts; the branch is taken automatically based on the `file_id` shape.

---

## License

MIBURI — source code, model architectures, training configs, and released checkpoint weights — is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International). See [`LICENSE`](LICENSE) for the full text. Research and academic use is welcome; commercial use is not permitted.

Portions of the codebase derive from third-party projects under their own licenses, which the CC BY-NC 4.0 terms do **not** override:

- **Kyutai Moshi / Mimi** runtime code is MIT-licensed; the runtime weights are fetched at demo time from [`kyutai/moshiko-pytorch-bf16`](https://huggingface.co/kyutai/moshiko-pytorch-bf16) and [`kyutai/moshi-artifacts`](https://huggingface.co/kyutai/moshi-artifacts) under CC-BY 4.0 (Kyutai).
- **Meta AudioCraft** — MIT-licensed; see [`LICENSE.audiocraft`](LICENSE.audiocraft).
- **SMPL-X model** ([`assets_dep/smplx_2020/`](assets_dep/smplx_2020/), distributed via [`m-hamza-mughal/miburi-release-assets`](https://huggingface.co/m-hamza-mughal/miburi-release-assets)) — Max Planck Society; non-commercial only. See the [SMPL-X model license](https://smpl-x.is.tue.mpg.de/modellicense.html).
- **Adobe Mixamo characters** ([`assets_dep/mixamo_characters_release/`](assets_dep/mixamo_characters_release/)) — Adobe [Mixamo terms of use](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html).

