"""Opt-in deterministic reconstruction validation for the existing codecs.

The training objective is unchanged. Validation streams each complete cached
clip once, with all codebooks and without dropout, crop, padding or a training
warmup mask. Finite pose_valid=False frames remain included. Feature families
have separate units; no mixed-unit average is used to select a checkpoint.
Optional FGD follows the existing BEATX evaluator's 25->30 fps pose resampling
and per-clip multiple-of-32 truncation. It sees rotations only, with the other
body parts copied from GT; it never replaces a reconstructed jaw with GT.
"""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger

from scripts.codec_reconstruction_metrics import merge_metric_sums, finalize_metrics


FEATURE_GROUPS = {
    "upper": {"rotation6d": (0, 258)},
    "lower": {"rotation6d": (0, 54), "translation_velocity": (54, 57), "contact": (57, 61)},
    "face": {"rotation6d": (0, 6), "expression": (6, 106)},
}
BEST_RECONSTRUCTION_METRIC = {
    "upper": "rotation6d_mse", "lower": "rotation6d_mse", "face": "expression_mse",
}
PSNR_RANGES = {"rotation6d": 2.0, "contact": 1.0}


def standard_feature_statistics(part, target, decoded, valid_mask=None):
    """Return pooled sufficient statistics for each native feature family.

    Inputs are [B,T,D] (or [T,D]). MSE/MAE count scalar coordinates; RMSE is
    rooted only after sums/counts are combined. Decoded values are not clipped.
    """
    if part not in FEATURE_GROUPS:
        raise ValueError(f"Unsupported codec part: {part}")
    target = torch.as_tensor(target).detach()
    decoded = torch.as_tensor(decoded).detach().to(target.device)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if decoded.ndim == 2:
        decoded = decoded.unsqueeze(0)
    expected = max(end for _, end in FEATURE_GROUPS[part].values())
    if target.shape != decoded.shape or target.ndim != 3 or target.shape[-1] != expected:
        raise ValueError(f"{part} reconstruction requires matching [B,T,{expected}] tensors")
    if valid_mask is None:
        valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
    else:
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=target.device)
        if valid.ndim == 1 and target.shape[0] == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != target.shape[:2]:
            raise ValueError("Feature validity mask must match [B,T]")
    result = {}
    for group, (start, end) in FEATURE_GROUPS[part].items():
        error = (decoded[..., start:end].double() - target[..., start:end].double())[valid]
        if not bool(torch.isfinite(error).all()):
            raise ValueError(f"Non-finite {part}/{group} reconstruction")
        count = error.numel()
        squared_sum = float(error.square().sum().item())
        result[f"{group}_mse"] = dict(sum=squared_sum, count=count, reduction="mean")
        result[f"{group}_rmse"] = dict(sum=squared_sum, count=count, reduction="rmse")
        result[f"{group}_mae"] = dict(sum=float(error.abs().sum().item()), count=count, reduction="mean")
    return result


def finalize_standard_metrics(statistics, part):
    """Pool feature errors and derive PSNR from fixed, meaningful ranges.

    A perfect reconstruction has mathematically infinite PSNR. No PSNR is
    invented for unbounded expression coefficients or translation velocity.
    """
    values = finalize_metrics(statistics)
    for group in FEATURE_GROUPS[part]:
        if group in PSNR_RANGES:
            mse = values.get(f"{group}_mse")
            values[f"{group}_psnr_db"] = (None if mse is None else
                math.inf if mse == 0 else 10 * math.log10(PSNR_RANGES[group] ** 2 / mse))
    return values


def _model(trainer):
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save_best_reconstruction(trainer, part, epoch, metrics):
    """Save bare current codec weights and explicit validation-selection metadata.

    Lower codec selection is rotation6d MSE, intentionally excluding velocity
    and contacts; their separately reported errors must also be reviewed.
    Returns True only for a strict improvement of the selected metric.
    """
    from safetensors.torch import save_file

    if getattr(trainer, "global_rank", 0) != 0:
        return False
    metric = BEST_RECONSTRUCTION_METRIC[part]
    value = metrics.get(metric)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Cannot select a reconstruction checkpoint without finite {metric}")
    directory = Path(trainer.checkpoint_path)
    metadata_path = directory / "best_reconstruction.json"
    previous = getattr(trainer, "_codec_best_reconstruction", None)
    if previous is None and getattr(trainer.args, "is_continue", False) and metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["metric"] != metric or previous["part"] != part:
            raise ValueError("Existing best reconstruction uses a different selection metric or part")
        trainer._codec_best_reconstruction = previous
    if previous is not None and float(value) >= previous["value"]:
        return False
    model = _model(trainer)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "best_reconstruction.safetensors"
    state = {name: tensor.detach().cpu().contiguous().clone()
             for name, tensor in model.state_dict().items()}
    temporary = directory / "best_reconstruction.pending.safetensors"
    save_file(state, str(temporary), metadata={"format": "pt", "selection_metric": metric,
                                               "epoch": str(int(epoch))})
    os.replace(temporary, checkpoint_path)
    metadata = dict(epoch=int(epoch), metric=metric, value=float(value), part=part,
                    split="val", scope="all_frames_of_complete_cached_clips",
                    mode="streaming_all_codebooks", fps=float(trainer.args.motion_fps),
                    checkpoint=str(checkpoint_path.resolve()), metrics=_json_finite(dict(metrics)),
                    fgd_status=getattr(trainer, "_codec_fgd_status", "disabled"))
    temporary_json = directory / "best_reconstruction.pending.json"
    temporary_json.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary_json, metadata_path)
    trainer._codec_best_reconstruction = metadata
    logger.info(f"Best {part} reconstruction: {metric}={value:.8g} at epoch {epoch}")
    return True


@contextmanager
def frozen_validation_codec(model):
    """Temporarily freeze evaluation without altering training parameter flags."""
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    quantizer = getattr(model, "quantizer", None)
    previous_codebooks = getattr(quantizer, "num_codebooks", None)
    try:
        model.eval()
        for parameter, _ in gradients:
            parameter.requires_grad_(False)
        if quantizer is not None and hasattr(quantizer, "set_num_codebooks"):
            quantizer.set_num_codebooks(quantizer.total_codebooks)
        yield model
    finally:
        if previous_codebooks is not None and hasattr(quantizer, "set_num_codebooks"):
            quantizer.set_num_codebooks(previous_codebooks)
        for parameter, requires_grad in gradients:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training


def prepare_validation_inputs(sample, part, args, device):
    """Use existing codec preprocessing on an uncollated complete clip."""
    from . import rotation_conversions as rc
    from .tools import estimate_linear_velocity

    def rotation(name):
        value = sample[f"motion_{name}"].to(device=device, dtype=torch.float32)
        value = value.reshape(value.shape[0], -1, 3)
        return rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(value)).reshape(1, value.shape[0], -1)
    if part == "upper":
        result = torch.cat((rotation("upper"), rotation("hands")), dim=-1)
    elif part == "face":
        expression = sample["expressions"].to(device=device, dtype=torch.float32).unsqueeze(0)
        result = torch.cat((rotation("face"), expression), dim=-1)
    elif part == "lower":
        translation = sample["transl"].to(device=device, dtype=torch.float32).unsqueeze(0).clone()
        if translation.shape[1] < 2:
            raise ValueError("Lower codec requires at least two frames for velocity preprocessing")
        translation[..., (0, 2)] -= translation[:, :1, (0, 2)].clone()
        velocity = estimate_linear_velocity(translation, dt=1 / args.motion_fps)
        contact = sample["contact"].to(device=device, dtype=torch.float32).unsqueeze(0)
        result = torch.cat((rotation("lower"), velocity, contact), dim=-1)
    else:
        raise ValueError(f"Unsupported codec part: {part}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"Non-finite validation input in {sample.get('filechunk_id', '<unknown>')}")
    return result


class _BEATXReconstructionFGD:
    """Existing rotation-only evaluator, isolated reconstructed part + GT rest."""
    def __init__(self, args, device):
        from .metric_utils.motion_representation import VAESKConv
        from .metrics import FIDCalculator, match_motion_fps

        # The convenience wrapper turns ValueError into 1e10. Call the
        # underlying moment calculation so failures remain unavailable scores.
        self._distance = FIDCalculator.calculate_frechet_distance
        self._resample = match_motion_fps
        self.device = device
        self.source_fps = args.motion_fps
        checkpoint = Path(args.beatx_data_path) / "weights" / "AESKConv_240_100.bin"
        model_path = Path(args.deps_path) / "smplx_2020"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        settings = SimpleNamespace(vae_length=240, vae_test_dim=330, variational=False,
                                   data_path_1=str(model_path) + os.sep, vae_layer=4, vae_grow=[1, 1, 2, 1])
        self.encoder = VAESKConv(settings).to(device).eval().requires_grad_(False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model_state"]
        self.encoder.load_state_dict({name.removeprefix("module."): value for name, value in state.items()}, strict=True)
        self.predicted = []
        self.target = []
        self.trimmed_frames = 0

    @torch.no_grad()
    def add(self, sample, part, decoded, dataset):
        from . import rotation_conversions as rc

        if sample.get("dataset_name") != "BEATX":
            raise ValueError("The optional reconstruction FGD requires BEATX validation clips")
        target = sample["motion"].reshape(-1, 55, 3).detach().cpu()
        predicted = target.clone()
        names = ("upper", "hands") if part == "upper" else (part,)
        ids = torch.cat([torch.as_tensor(getattr(dataset, f"{name}_mask")).bool().nonzero().flatten()
                         for name in names])
        rotation_features = FEATURE_GROUPS[part]["rotation6d"][1]
        rotations = rc.rotation_6d_to_matrix(decoded[0, :, :rotation_features].reshape(-1, len(ids), 6))
        predicted[:, ids] = rc.matrix_to_axis_angle(rotations).cpu()
        for pose, destination in ((target, self.target), (predicted, self.predicted)):
            resampled = self._resample(pose.numpy(), source_fps=self.source_fps, target_fps=30)
            frames = len(resampled)
            usable = frames - frames % 32
            if destination is self.target:
                self.trimmed_frames += frames - usable
            if not usable:
                continue
            pose_tensor = torch.as_tensor(resampled[:usable], device=self.device, dtype=torch.float32)
            features = rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(pose_tensor)).reshape(1, usable, 330)
            latent = self.encoder.map2latent(features).reshape(-1, 240)
            if not bool(torch.isfinite(latent).all()):
                raise ValueError("Non-finite FGD latent")
            destination.append(latent.detach().cpu().numpy())

    def compute(self):
        import numpy as np

        if not self.target or sum(len(value) for value in self.target) < 2:
            raise ValueError("Insufficient validation latents for FGD")
        predicted = np.concatenate(self.predicted)
        target = np.concatenate(self.target)
        value = self._distance(predicted.mean(0), np.cov(predicted, rowvar=False),
                               target.mean(0), np.cov(target, rowvar=False))
        if not math.isfinite(float(value)):
            raise ValueError("Non-finite reconstruction FGD")
        return float(value)


@torch.no_grad()
def run_standard_codec_validation(trainer, part, epoch):
    """Validate rank-zero whole clips once; all DDP ranks enter this function.

    Direct dataset access avoids random collate cropping and distributed-sampler
    duplicate clips. Other ranks wait; codec forward uses the unwrapped model,
    so no DDP-forward collective is entered only by rank zero.
    """
    from scripts.codec_reconstruction_audit import reconstruct_codec
    from .distributed import sync_tracker_meters

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    model = _model(trainer)
    device = next(model.parameters()).device
    sync_tracker_meters(trainer.tracker, states=("train",), device=device)
    if distributed:
        torch.distributed.barrier()
    if getattr(trainer, "global_rank", 0) != 0:
        if distributed:
            status = [None]
            torch.distributed.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise RuntimeError(f"Rank-zero standard codec validation failed: {status[0]}")
        return None
    validation_error = None
    try:
        statistics = {}
        dataset = trainer.val_data
        indices = list(range(len(dataset)))
        if hasattr(dataset, "_chunk_refs"):
            indices.sort(key=lambda i: dataset._chunk_refs[i].chunk_id)
        fgd = None
        trainer._codec_fgd_status = "disabled"
        if getattr(trainer.args, "codec_eval_fgd", False):
            try:
                fgd = _BEATXReconstructionFGD(trainer.args, device)
                trainer._codec_fgd_status = "available"
            except Exception as error:
                trainer._codec_fgd_status = f"unavailable: {type(error).__name__}: {error}"
                logger.warning(f"Optional codec FGD {trainer._codec_fgd_status}; standard validation continues")
        clip_count, frame_count = 0, 0
        with frozen_validation_codec(model):
            for index in indices:
                sample = dataset[index]
                if hasattr(dataset, "_chunk_refs") and sample["filechunk_id"] != dataset._chunk_refs[index].chunk_id:
                    raise ValueError("Validation dataset silently substituted a requested clip")
                inputs = prepare_validation_inputs(sample, part, trainer.args, device)
                decoded, _ = reconstruct_codec(model, inputs, mode="streaming")
                merge_metric_sums(statistics, standard_feature_statistics(part, inputs, decoded))
                clip_count += 1
                frame_count += inputs.shape[1]
                if fgd is not None:
                    try:
                        fgd.add(sample, part, decoded, dataset)
                    except Exception as error:
                        trainer._codec_fgd_status = f"unavailable: {type(error).__name__}: {error}"
                        logger.warning(f"Optional codec FGD {trainer._codec_fgd_status}; standard validation continues")
                        fgd = None
        metrics = finalize_standard_metrics(statistics, part)
        metrics.update(clip_count=clip_count, frame_count=frame_count)
        if getattr(trainer.args, "codec_eval_fgd", False):
            if fgd is not None:
                try:
                    metrics["rotation_only_part_isolated_fgd_30fps"] = fgd.compute()
                    metrics["fgd_trimmed_resampled_frames"] = fgd.trimmed_frames
                except Exception as error:
                    trainer._codec_fgd_status = f"unavailable: {type(error).__name__}: {error}"
                    logger.warning(f"Optional codec FGD {trainer._codec_fgd_status}; standard validation continues")
            metrics["fgd_available"] = int(trainer._codec_fgd_status == "available")
        save_best_reconstruction(trainer, part, epoch, metrics)
        best = trainer._codec_best_reconstruction
        logged = {f"codec_standard/{name}": value for name, value in metrics.items()
                  if value is not None and math.isfinite(float(value))}
        logged.update({"codec_standard/best_reconstruction_value": best["value"],
                       "codec_standard/best_reconstruction_epoch": best["epoch"]})
        wandb_logger = getattr(trainer, "wandb_logger", None)
        if wandb_logger is not None:
            wandb_logger.log_validation(tracker=trainer.tracker, epoch=epoch, extra_values=logged)
        writer = getattr(trainer, "writer", None)
        if writer is not None:
            for name, value in logged.items():
                writer.add_scalar(f"val/{name}", value, epoch)
        trainer._last_codec_standard_metrics = metrics
        logger.info(f"Standard {part} validation epoch {epoch}: {metrics}")
        return metrics
    except Exception as error:
        validation_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        if distributed:
            torch.distributed.broadcast_object_list([validation_error], src=0)
