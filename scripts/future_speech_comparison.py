"""Fixed-checkpoint causal / matched-future / mismatched-future evaluation.

Run from the repository root; see docs/future_speech_comparison.md. No model
training or optimizer is constructed. Real metrics require remote data/assets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from scripts import codec_reconstruction_audit as codec_audit
from miburi.models.gesture_lm_future_control import (
    CONDITIONS, generate_current_codes, speech_views, teacher_forced_logits, temporal_prediction,
)


SCOPES = {"all": range(20), "temporal_q0": range(1), "depth": range(1, 20),
          "upper_hands": range(8), "lower": range(8, 16), "face": range(16, 20)}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_donor_plan(infos, targets, clip_steps, seed):
    """Different utterance, same speaker, same split, exact length; no tiling."""
    rng = random.Random(seed)
    plan = []
    for target in targets:
        for start in range(0, target["steps"], clip_steps):
            end = min(target["steps"], start + clip_steps)
            length = end - start
            choices = [i for i in infos if i["speaker"] == target["speaker"]
                       and i["recording_id"] != target["recording_id"] and i["steps"] >= length]
            rng.shuffle(choices)
            selected = None
            for donor in choices:
                starts = list(range(0, donor["steps"] - length + 1, clip_steps))
                rng.shuffle(starts)
                for donor_start in starts:
                    a = target["audio"][:, start:end]
                    b = donor["audio"][:, donor_start:donor_start + length]
                    x = target["text"][:, start:end]
                    y = donor["text"][:, donor_start:donor_start + length]
                    # The current/first group is never replaced. Require a
                    # difference in the part that can actually become future.
                    if length == 1 or not (torch.equal(a[:, 1:], b[:, 1:]) and torch.equal(x[:, 1:], y[:, 1:])):
                        selected = (donor, donor_start)
                        break
                if selected:
                    break
            if selected is None:
                raise ValueError(f"No distinct same-speaker donor for {target['recording_id']} at {start}")
            donor, donor_start = selected
            plan.append({"recording_id": target["recording_id"], "speaker": target["speaker"],
                         "clip_start_step": start, "clip_end_step": end,
                         "donor_recording_id": donor["recording_id"], "donor_start_step": donor_start,
                         "donor_end_step": donor_start + length})
    return plan


def load_generator(args, codecs, checkpoint, device):
    """Initialize embedding shapes from this checkpoint, avoiding Moshi downloads."""
    from safetensors.torch import load_file
    from miburi.models import GTemporalDepthModel3, loaders

    weights = codec_audit._strip_checkpoint_prefixes(load_file(str(checkpoint), device="cpu"))
    text = weights["text_procemb.weight"]
    audio = [weights[f"audio_procemb.{i}.weight"] for i in range(8)]
    kwargs = copy.deepcopy(loaders.get_gesturelm_kwargs())
    kwargs["depformer_weights_per_step"] = args.gestureformer_depformer_weights_per_step
    layers = torch.nn.ModuleList([
        copy.deepcopy(layer) for region in ("upper", "lower", "face")
        for layer in codecs[region].quantizer.vq.layers
    ])
    model = GTemporalDepthModel3(
        num_heads=args.gestureformer_heads, num_layers=args.gestureformer_layers,
        depformer_heads=args.gestureformer_depformer_heads, depformer_layers=args.gestureformer_depformer_layers,
        query2mem_scale=1, num_temp_classifiers=args.num_temp_classifiers,
        text_procemb=text, audio_procemb=audio, gesture_codec_layers=layers,
        vad_guidance=args.vad_guidance, vad_use_face_logits=args.vad_use_face_logits,
        body_parts=3, bp_dist=None, textaudio_emb_freeze=args.textaudio_emb_freeze,
        dtype=getattr(torch, args.param_dtype), **kwargs,
    )
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False).eval().to(device=device, dtype=getattr(torch, args.param_dtype))
    level = 0
    for region in ("upper", "lower", "face"):
        for source_layer in codecs[region].quantizer.vq.layers:
            stored = model.gesture_codec_layers[level].state_dict()
            for key, value in source_layer.state_dict().items():
                expected = value.to(device=stored[key].device, dtype=stored[key].dtype)
                if not torch.equal(expected, stored[key]):
                    raise ValueError(f"{region} codec does not match the frozen quantizer stored in the generator (level {level}, {key})")
            level += 1
    if model.n_q != 20 or model.query2mem_scale != 1:
        raise ValueError("This experiment expects the existing 20-stream, 12.5-Hz setup")
    return model


def load_data(cli, config):
    module, _, _ = codec_audit._data_modules(ROOT)
    dataset = module.UNIFIEDDataset(config, cli.split, only_motion=False,
                                    dataset_ratio=config.dataset_ratio, debug=False,
                                    varying_frame_length=False, ret_rawaudio=False,
                                    align_first_frame_yaw=config.align_first_frame_yaw)
    infos = []
    for index, ref in enumerate(dataset._chunk_refs):
        if ref.dataset_name != "BEATX" or "_C" in ref.chunk_id:
            raise ValueError("Use a full-sequence BEATX cache; training chunks are not accepted")
        group = dataset._get_h5(ref.hdf5_path)[ref.chunk_id]
        fid = codec_audit._data_modules(ROOT)[0]._decode_if_bytes(group.attrs["file_id"])
        speaker = fid.split("_")[1].casefold()
        if cli.speakers and speaker not in cli.speakers:
            continue
        if float(group.attrs.get("chunk_startsec", 0)) != 0:
            raise ValueError("Only complete utterances starting at zero are supported")
        frames = int(group["motion"].shape[0])
        steps = frames // config.frame_chunk_size
        audio = torch.as_tensor(group["moshi_audiotokens"][:]).long().reshape(8, -1)
        text = torch.as_tensor(group["moshi_texttokens"][:]).long().reshape(1, -1)
        if steps < 2 or min(audio.shape[-1], text.shape[-1]) < steps:
            raise ValueError(f"Short or misaligned speech/motion at {fid}")
        infos.append({"recording_id": fid, "speaker": speaker, "dataset_index": index,
                      "steps": steps, "frames": steps * config.frame_chunk_size,
                      "trailing_frames_dropped": frames % config.frame_chunk_size,
                      "audio": audio[:, :steps], "text": text[:, :steps]})
    if len({i["recording_id"] for i in infos}) != len(infos):
        raise ValueError("Duplicate recording IDs in the selected evaluation cache")
    if not infos:
        raise ValueError("No recordings match this split and speaker selection")
    return dataset, sorted(infos, key=lambda i: i["recording_id"])


def get_sample(dataset, info):
    sample = dataset[info["dataset_index"]]
    if sample["filechunk_id"] != dataset._chunk_refs[info["dataset_index"]].chunk_id:
        raise ValueError("Dataset substituted a different recording; evaluation refuses replacement")
    frames = info["frames"]
    result = dict(sample)
    for key in ("motion", "motion_upper", "motion_hands", "motion_lower", "motion_face",
                "transl", "contact", "expressions", "beta", "joint_positions", "vad_bits"):
        if torch.is_tensor(result.get(key)):
            result[key] = result[key][:frames]
    for key in ("motion", "transl", "expressions", "beta"):
        if not torch.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite {key} in {info['recording_id']}")
    return result


def count_tokens(logits, target, valid):
    """Unweighted hard CE in nats, over original V+1 classes including pad."""
    if logits.shape[:2] != (3, 20) or target.shape != (20,) or valid.shape != (20,):
        raise ValueError("Unexpected token-metric dimensions")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite prediction logits")
    gold = target[None].expand(3, -1)
    nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), gold.reshape(-1), reduction="none").reshape(3, 20)
    correct = logits.argmax(-1) == gold
    top5 = (logits.topk(min(5, logits.shape[-1]), -1).indices == gold[..., None]).any(-1)
    mask = valid[None].expand(3, -1)
    return torch.stack((nll * mask, correct * mask, top5 * mask, mask), -1).double().cpu().numpy()


def token_rows(totals, info):
    rows = []
    for view, name in enumerate(CONDITIONS):
        for codebook in range(20):
            nll, correct, top5, count = totals[view, codebook]
            rows.append({"recording_id": info["recording_id"], "speaker": info["speaker"],
                         "condition": name, "codebook": codebook, "nll_sum": float(nll),
                         "correct": int(correct), "top5_correct": int(top5), "tokens": int(count)})
    return rows


def aggregate_tokens(rows, by_speaker=False):
    result = []
    speakers = sorted({r["speaker"] for r in rows}) if by_speaker else [None]
    for speaker in speakers:
        for condition in CONDITIONS:
            for scope, indices in SCOPES.items():
                group = [r for r in rows if r["condition"] == condition and r["codebook"] in indices
                         and (speaker is None or r["speaker"] == speaker)]
                count = sum(r["tokens"] for r in group)
                nll = sum(r["nll_sum"] for r in group)
                result.append({**({"speaker": speaker} if by_speaker else {}),
                               "condition": condition, "scope": scope, "tokens": count,
                               "ce_nats": nll / count if count else None,
                               "accuracy": sum(r["correct"] for r in group) / count if count else None,
                               "top5_accuracy": sum(r["top5_correct"] for r in group) / count if count else None})
    return result


def paired_differences(rows, bootstrap, seed):
    """Paired whole-recording resampling; never bootstrap individual tokens."""
    result = []
    ids = sorted({r["recording_id"] for r in rows})
    rng = np.random.default_rng(seed)
    for scope, indices in SCOPES.items():
        grouped = defaultdict(lambda: np.zeros(3))
        for row in rows:
            if row["codebook"] in indices:
                grouped[row["condition"], row["recording_id"]] += [row["nll_sum"], row["correct"], row["tokens"]]
        values = {}
        for condition in CONDITIONS:
            values[condition] = np.array([grouped[condition, fid] for fid in ids], dtype=float)
        for first, second in (("matched_future", "causal"), ("mismatched_future", "matched_future")):
            a, b = values[first], values[second]
            if not np.array_equal(a[:, 2], b[:, 2]):
                raise ValueError("Conditions did not score identical targets")
            for metric, column in (("ce_nats", 0), ("accuracy", 1)):
                count = a[:, 2].sum()
                difference = (a[:, column].sum() - b[:, column].sum()) / count if count else None
                lower = upper = None
                usable = np.flatnonzero(a[:, 2] > 0)
                if bootstrap and len(usable) >= 2:
                    picks = rng.choice(usable, size=(bootstrap, len(usable)), replace=True)
                    distribution = (a[picks, column].sum(1) - b[picks, column].sum(1)) / a[picks, 2].sum(1)
                    lower, upper = np.quantile(distribution, [0.025, 0.975]).tolist()
                result.append({"scope": scope, "contrast": f"{first} - {second}", "metric": metric,
                               "difference": difference, "ci95_low": lower, "ci95_high": upper})
    return result


def metric_instances(config):
    """Share frozen evaluator networks; isolate each condition's accumulators."""
    name = "_future_comparison_metric_utils"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / "scripts/trainers/utils")]
        sys.modules[name] = package
    cls = importlib.import_module(name + ".metrics").GestureMetrics
    base = cls(config)
    base.eval_copy.requires_grad_(False).eval()
    base.smplx.requires_grad_(False).eval()
    result = {}
    for condition in CONDITIONS:
        instance = copy.copy(base)
        for key in ("alignmenter", "gt_alignmenter", "l1_calculator", "gt_l1_calculator"):
            setattr(instance, key, copy.deepcopy(getattr(base, key)))
        instance.reset()
        result[condition] = instance
    return result


@torch.no_grad()
def decode_motion(codecs, codes, sample, dataset, config):
    """Original regional decoding, 6D rotations and mixed-difference velocity."""
    _, rc, kinematics = codec_audit._data_modules(ROOT)
    device = codes.device
    batch, _, steps = codes.shape
    regional = {}
    for part, interval in (("upper", slice(0, 8)), ("lower", slice(8, 16)), ("face", slice(16, 20))):
        with codecs[part].streaming(batch):
            regional[part] = torch.cat([codecs[part].decode(codes[:, interval, t:t + 1]) for t in range(steps)], 1)
    frames = steps * config.frame_chunk_size
    full = torch.zeros(batch, frames, 55, 3, device=device)
    masks = {"upper": np.logical_or(dataset.upper_mask, dataset.hands_mask),
             "lower": dataset.lower_mask.astype(bool), "face": dataset.face_mask.astype(bool)}
    target = torch.zeros(frames, 55, 3, device=device)
    reference = sample["motion"].to(device)
    for part in ("upper", "lower", "face"):
        mask = torch.as_tensor(masks[part], device=device)
        joints = int(mask.sum())
        value = regional[part][..., :joints * 6].reshape(batch, frames, joints, 6)
        full[:, :, mask] = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(value))
        target[:, mask] = reference[:, mask]
    target_trans = sample["transl"].to(device).clone()
    target_trans[:, (0, 2)] -= target_trans[:1, (0, 2)]
    lower_joints = int(np.sum(masks["lower"]))
    velocity = regional["lower"][..., lower_joints * 6:lower_joints * 6 + 3]
    translation = codec_audit._integrate_velocity(velocity, target_trans[:1].expand(batch, -1).clone(),
                                                chunk_size=config.frame_chunk_size, fps=config.motion_fps,
                                                kinematics=kinematics)
    return {"poses": full.reshape(batch, frames, 165), "expressions": regional["face"][..., 6:],
            "translation": translation, "target_pose": target.reshape(frames, 165), "target_translation": target_trans}


@torch.no_grad()
def evaluate_recording(model, codecs, sample, info, all_infos, plan, config, cli, speaker, *, generation_seed, score_tokens):
    device = next(model.parameters()).device
    steps = info["steps"]
    audio, text = model.process_conditions(info["audio"][None].to(device), info["text"][None].to(device))
    audio, text = audio.squeeze(1), text.squeeze(1)
    generated = torch.empty(3, 20, 0, dtype=torch.long, device=device)
    rng = torch.Generator().manual_seed(cli.seed if generation_seed is None else generation_seed)
    totals = np.zeros((3, 20, 4), dtype=float)
    gold = None
    valid = torch.ones(20, steps, dtype=torch.bool, device=device)
    if score_tokens:
        prepared = codec_audit.prepare_codec_inputs(sample, config, device=device)
        gold = torch.cat([codecs[p].encode(prepared["inputs"][p]) for p in ("upper", "lower", "face")], 1)
        if gold.shape != (1, 20, steps):
            raise ValueError("Reference codec tokens do not align with cached speech")
        if int(sample["lower_valid_mask"]) == 0:
            gold[:, 8:] = model.pad_token_id
            valid[8:] = False
    donor_audio = donor_text = None
    changed_suffix = None
    coverage = {"scored_positions": 0, "scored_positions_with_changed_future": 0,
                "remaining_future_groups_sum": 0}
    plans = {p["clip_start_step"]: p for p in plan if p["recording_id"] == info["recording_id"]}
    generation = generation_seed is not None
    for step in range(steps):
        start = step // cli.clip_steps * cli.clip_steps
        end = min(steps, start + cli.clip_steps)
        if step == start:
            entry = plans[start]
            donor = all_infos[entry["donor_recording_id"]]
            lo, hi = entry["donor_start_step"], entry["donor_end_step"]
            donor_audio, donor_text = model.process_conditions(donor["audio"][None, :, lo:hi].to(device), donor["text"][None, :, lo:hi].to(device))
            donor_audio, donor_text = donor_audio.squeeze(1), donor_text.squeeze(1)
            changed = (info["audio"][:, start:end] != donor["audio"][:, lo:hi]).any(0) | (info["text"][:, start:end] != donor["text"][:, lo:hi]).any(0)
            changed_suffix = torch.cat((changed.long().flip(0).cumsum(0).flip(0), torch.zeros(1, dtype=torch.long)))
        available_future = end - step - 1
        scoring = score_tokens and step % cli.token_stride == 0 and available_future >= cli.min_future_steps
        if not generation and not scoring:
            continue
        av, tv = speech_views(audio, text, donor_audio, donor_text, step=step, clip_start=start, clip_end=end)
        current_audio = audio[:, step:step + 1].repeat(3, 1, 1)
        current_text = text[:, step:step + 1].repeat(3, 1, 1)
        if scoring:
            coverage["scored_positions"] += 1
            coverage["remaining_future_groups_sum"] += available_future
            coverage["scored_positions_with_changed_future"] += int(changed_suffix[step - start + 1] > 0)
            histories = gold[:, :, :step].expand(3, -1, -1)
            hidden, q0 = temporal_prediction(model, histories, av, tv, speaker, cli.clip_steps)
            logits = teacher_forced_logits(model, hidden, q0, gold[:, :, step:step + 1],
                                          current_audio, current_text, speaker, drop_lower=config.drop_lower_crossattn)
            totals += count_tokens(logits, gold[0, :, step], valid[:, step])
        if generation:
            histories = generated
            gav, gtv = av, tv
            gac, gtc = current_audio, current_text
            if cli.cfg_scale != 1:
                histories = histories.repeat(2, 1, 1)
                gav, gtv = torch.cat((av, torch.zeros_like(av))), torch.cat((tv, torch.zeros_like(tv)))
                gac, gtc = torch.cat((gac, torch.zeros_like(gac))), torch.cat((gtc, torch.zeros_like(gtc)))
            hidden, q0 = temporal_prediction(model, histories, gav, gtv, speaker, cli.clip_steps)
            prediction = generate_current_codes(model, hidden, q0, gac, gtc, speaker, rng=rng,
                                                sampling=cli.sampler == "production", cfg_scale=cli.cfg_scale,
                                                drop_lower=config.drop_lower_crossattn)
            generated = torch.cat((generated, prediction), -1)
        if (step + 1) % 50 == 0 or step == steps - 1:
            print(f"  {info['recording_id']}: {step + 1}/{steps} steps", flush=True)
    return totals, generated if generation else None, coverage


def write_report(output, token_summary, generated_metrics):
    combined = []
    for condition in CONDITIONS:
        row = {"condition": condition}
        for scope in ("all", "temporal_q0", "depth", "upper_hands", "lower", "face"):
            entry = next((r for r in token_summary if r["condition"] == condition and r["scope"] == scope), None)
            row[f"ce_{scope}"] = entry["ce_nats"] if entry else None
            row[f"accuracy_{scope}"] = entry["accuracy"] if entry else None
        for metric in ("fgd", "beat_alignment", "gt_beat_alignment", "facial_l2", "expression_mse", "l1_diversity", "gt_l1_diversity"):
            values = [r[metric] for r in generated_metrics if r["condition"] == condition and metric in r]
            row[metric] = float(np.mean(values)) if values else None
            row[metric + "_seed_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        combined.append(row)
    write_csv(output / "side_by_side.csv", combined)
    columns = ("condition", "ce_all", "accuracy_all", "ce_temporal_q0", "accuracy_temporal_q0", "fgd", "beat_alignment", "facial_l2", "expression_mse")
    lines = ["# Fixed-checkpoint future-speech comparison", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in combined:
        lines.append("| " + " | ".join(str(row[c]) if c == "condition" else f"{row[c]:.6f}" if row[c] is not None else "" for c in columns) + " |")
    lines += ["", "Accuracy is a fraction; CE is unweighted hard token CE in nats over the original heads including padding as an output class. Padding targets are excluded.",
              "Token metrics are teacher-forced and use identical gold gesture prefixes. Generation is autoregressive; histories diverge as an outcome of the interventions.",
              "All views share weights and speaker identity. Mismatched future audio/text are jointly taken from a different recording by the same speaker, with equal duration.",
              "Future scope is each nonoverlapping training-length speech clip; motion histories do not reset at those clip boundaries.",
              "Generated metrics use the existing evaluator. facial_l2 retains its existing vertex-space facial metric definition; it is not expression-coefficient MSE.",
              "expression_mse separately measures generated versus reference expression coefficients at 25 fps, before the existing evaluator's resampling and jaw replacement.",
              "Correct-past and donor-future memories are rebuilt at every prediction, so correct-future hidden states are never cached into the mismatched view."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cli):
    if cli.token_stride < 1 or cli.min_future_steps < 1 or cli.bootstrap < 0:
        raise ValueError("Invalid token stride, future minimum, or bootstrap count")
    if cli.mode != "tokens" and not cli.device.startswith("cuda"):
        raise ValueError("The existing generated-motion evaluator requires CUDA")
    if not math.isfinite(cli.cfg_scale) or cli.cfg_scale < 0:
        raise ValueError("CFG scale must be finite and nonnegative")
    if len(cli.generation_seeds) != len(set(cli.generation_seeds)) or any(s < 0 for s in cli.generation_seeds):
        raise ValueError("Generation seeds must be distinct and nonnegative")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    overrides = {k: str(v) for k, v in {"beatx_cache_path": cli.eval_hdf5, "beatx_data_path": cli.data_dir,
                 "upperbodycodec_ckpt": cli.upper_codec, "lowerbodycodec_ckpt": cli.lower_codec,
                 "facecodec_ckpt": cli.face_codec}.items() if v is not None}
    settings = codec_audit.AuditSettings(lm_config=str(cli.config), path_overrides=overrides)
    config = codec_audit.parse_lm_config(settings)
    if "ConditionCRegretRVQ" not in config.trainer:
        raise ValueError("Supply the ForeMotion shared-weight Condition C training config")
    config.dataset_ratio = cli.dataset_ratio or ("scott_beatx_fulllength" if "scott" in config.dataset_ratio else "full_beatx_fulllength")
    config.embody3d_cache_path = None
    if config.motion_fps != 25 or config.frame_chunk_size != 2:
        raise ValueError("This experiment uses the current 25-fps / 12.5-Hz architecture")
    cli.clip_steps = cli.clip_steps or config.pose_length // config.frame_chunk_size
    if cli.clip_steps < 2:
        raise ValueError("Future speech clips must contain at least two steps")
    if cli.mode != "generation" and cli.min_future_steps >= cli.clip_steps:
        raise ValueError("min-future-steps must be smaller than clip-steps")
    speaker_map_path = cli.speaker_map or cli.checkpoint.parent / "speaker_id_to_index.json"
    speaker_map = json.loads(speaker_map_path.read_text(encoding="utf-8"))
    dataset, infos = load_data(cli, config)
    targets = infos[:cli.max_recordings] if cli.max_recordings else infos
    if cli.max_recordings is not None and cli.max_recordings < 1:
        raise ValueError("max-recordings must be positive")
    if any(i["speaker"] not in speaker_map for i in targets):
        raise ValueError("Speaker not found in the checkpoint's saved training speaker map")
    plan = make_donor_plan(infos, targets, cli.clip_steps, cli.seed)
    checkpoints = {"generator": cli.checkpoint, **{p: Path(getattr(config, field)) for p, field in codec_audit.CHECKPOINT_FIELDS.items()}}
    checkpoint_hashes = {k: codec_audit._sha256(p) for k, p in checkpoints.items()}
    output = cli.output_dir or ROOT / "reports" / f"future_speech_comparison_{datetime.now():%Y%m%d_%H%M%S}"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new/empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "donor_manifest.csv", plan)
    write_csv(output / "recordings.csv", [{k: v for k, v in i.items() if k not in {"audio", "text"}} for i in targets])
    metadata = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
                "checkpoint_sha256": checkpoint_hashes,
                "config_sha256": codec_audit._sha256(cli.config), "speaker_map": speaker_map,
                "recordings": len(targets), "donor_pool_recordings": len(infos), "split": cli.split,
                "torch_version": torch.__version__, "future_scope": "nonoverlapping speech clips; same lower past boundary",
                "token_targets": "deterministic canonical codec codes; identical teacher-forced history in all views",
                "token_weighting": "valid-token micro average; no face multiplier, KL, RVQ soft targets or auxiliary losses",
                "token_positions": "positions matching token-stride and min-future-steps; identical positions/targets in all views",
                "facial_metric_protocol": "existing facial_l2 uses resampling to 30 fps and ground-truth jaw substitution; expression_mse uses untouched generated coefficients at 25 fps",
                "generation": "common uniform sampling variates across conditions; autoregressive histories; no temporal KV reuse",
                "codec_target_mode": "batched encode, matching deterministic production validation",
                "pose_valid_policy": "same supplied finite motion as production evaluation; pose_valid flags are not used to remove frames",
                "partial_run": cli.max_recordings is not None}
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    totals_rows, generated_rows = [], []
    coverage_rows = []
    generation_record_rows = []
    lookup = {i["recording_id"]: i for i in infos}
    seeds = cli.generation_seeds if cli.mode != "tokens" else [None]
    started = time.monotonic()
    try:
        codecs = codec_audit.load_codecs(config, device=device)
        model = load_generator(config, codecs, cli.checkpoint, device)
        device = next(model.parameters()).device
        metadata["execution_device"] = str(device)
        metadata["gesture_context_steps"] = model.context
        metadata["speech_context_groups"] = model.temporal_transformer.layers[0].cross_attns[0].context
        metrics = metric_instances(config) if cli.mode != "tokens" else None
        for seed_index, seed in enumerate(seeds):
            expression_sums = {name: [0.0, 0] for name in CONDITIONS}
            if metrics:
                for instance in metrics.values():
                    instance.reset()
            for number, info in enumerate(targets, start=1):
                print(f"Recording {number}/{len(targets)}, seed={seed}: {info['recording_id']}", flush=True)
                sample = get_sample(dataset, info)
                scoring = cli.mode != "generation" and seed_index == 0
                totals, generated, coverage = evaluate_recording(model, codecs, sample, info, lookup, plan, config, cli,
                                                       speaker_map[info["speaker"]], generation_seed=seed,
                                                       score_tokens=scoring)
                if scoring:
                    totals_rows.extend(token_rows(totals, info))
                    coverage_rows.append({"recording_id": info["recording_id"], "speaker": info["speaker"], **coverage})
                    write_csv(output / "token_by_recording_codebook.csv", totals_rows)
                    write_csv(output / "token_intervention_coverage.csv", coverage_rows)
                if generated is not None:
                    folder = output / "generated_tokens" / str(seed)
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / f"{info['recording_id']}.npz", codes=generated.cpu().numpy(), conditions=np.array(CONDITIONS))
                    motion = decode_motion(codecs, generated, sample, dataset, config)
                    for index, condition in enumerate(CONDITIONS):
                        instance = metrics[condition]
                        before = instance.dataset_len
                        expression_error = (motion["expressions"][index].float() - sample["expressions"].to(device)).square()
                        expression_sums[condition][0] += float(expression_error.sum())
                        expression_sums[condition][1] += expression_error.numel()
                        generation_record_rows.append({"recording_id": info["recording_id"], "speaker": info["speaker"],
                                                       "condition": condition, "seed": seed,
                                                       "expression_mse": float(expression_error.mean()), "frames": info["frames"]})
                        instance.update({"rec_pose": motion["poses"][index], "rec_exps": motion["expressions"][index],
                                         "rec_trans": motion["translation"][index], "tar_pose": motion["target_pose"],
                                         "tar_exps": sample["expressions"].to(device), "tar_beta": sample["beta"][0].to(device),
                                         "tar_trans": motion["target_translation"], "file_id": info["recording_id"]})
                        if instance.dataset_len != before + 1:
                            raise ValueError(f"Existing evaluator did not process {info['recording_id']} for {condition}")
                    write_csv(output / "generation_by_recording.csv", generation_record_rows)
                    if cli.save_motion:
                        folder = output / "generated_motion" / str(seed)
                        folder.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(folder / f"{info['recording_id']}.npz",
                                            **{k: v.cpu().numpy() for k, v in motion.items()}, conditions=np.array(CONDITIONS))
            if metrics:
                for condition, instance in metrics.items():
                    if instance.dataset_len != len(targets):
                        raise ValueError("A condition evaluated a different recording set")
                    values = instance.compute_metrics()
                    if not values or not all(math.isfinite(v) for v in values.values()):
                        raise ValueError(f"Missing/nonfinite generated metrics for {condition}")
                    numerator, denominator = expression_sums[condition]
                    values["expression_mse"] = numerator / denominator
                    generated_rows.append({"condition": condition, "seed": seed, "recordings": instance.dataset_len, **values})
                write_csv(output / "generation_metrics.csv", generated_rows)
        summary = aggregate_tokens(totals_rows) if totals_rows else []
        write_csv(output / "token_summary.csv", summary)
        write_csv(output / "token_by_speaker.csv", aggregate_tokens(totals_rows, by_speaker=True) if totals_rows else [])
        write_csv(output / "paired_token_differences.csv", paired_differences(totals_rows, cli.bootstrap, cli.seed) if totals_rows else [])
        write_report(output, summary, generated_rows)
        metadata.update(status="complete", elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        metadata.update(status="failed", error=str(exc), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        dataset.close()
    print(f"Comparison complete: {output.resolve()}")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training config belonging to this ForeMotion checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-hdf5", type=Path, required=True, help="Full-utterance evaluation cache")
    parser.add_argument("--speaker-map", type=Path, help="Defaults to speaker_id_to_index.json beside checkpoint")
    parser.add_argument("--dataset-ratio", choices=("scott_beatx_fulllength", "full_beatx_fulllength", "goodspk_beatx_fulllength"))
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--speakers", nargs="+")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--upper-codec", type=Path)
    parser.add_argument("--lower-codec", type=Path)
    parser.add_argument("--face-codec", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("both", "tokens", "generation"), default="both")
    parser.add_argument("--clip-steps", type=int, help="Defaults to training pose_length / frame_chunk_size (125)")
    parser.add_argument("--token-stride", type=int, default=1)
    parser.add_argument("--min-future-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2342, help="Fixed donor plan and token bootstrap seed")
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=[2342])
    parser.add_argument("--sampler", choices=("production", "greedy"), default="production")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--save-motion", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Error: {exc}")
