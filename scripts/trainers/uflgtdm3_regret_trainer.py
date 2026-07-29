"""Causal MIBURI training with a frozen future-aware q0 teacher.

The student is the released causal ``GTemporalDepthModel3``.  A separately
loaded full-gesture/full-audio-text checkpoint supplies a detached posterior
distribution for one masked target per sample.  Only the teacher's temporal
q0 path is retained on GPU; its complete kinematic transformer is discarded.
"""

from __future__ import annotations

import copy
import math
import os

import torch
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import load_file

from miburi.models import (
    GTemporalDepthModel3FutureGestureFullCondition,
    forward_q0_regret_kl,
    is_temporal_q0_state_key,
    retain_temporal_q0_only,
)
from miburi.models import loaders
from miburi.models.gesture_lm_future_gesture import (
    build_masked_future_gesture_inputs,
)

from .uflgtdm3_trainer import UpperFaceLowerGTDM3Trainer
from .utils import tools as other_tools


def _canonical_checkpoint_key(name: str) -> str:
    """Remove wrappers used by DDP and older model containers."""

    if name.startswith("module."):
        name = name[len("module."):]
    if name.startswith("m."):
        name = name[len("m."):]
    return name


class UpperFaceLowerGTDM3RegretTrainer(UpperFaceLowerGTDM3Trainer):
    """Original causal student plus frozen future-aware q0 forward KL."""

    _REGRET_METRICS = (
        ("regret_loss", False),
        ("regret_weighted", False),
        ("regret_alpha", False),
        ("regret_teacher_q0_ce", False),
        ("regret_student_q0_ce", False),
        ("regret_teacher_q0_acc", True),
        ("regret_student_q0_acc", True),
        ("regret_teacher_entropy", False),
        ("regret_student_entropy", False),
        ("regret_teacher_gt_prob", True),
        ("regret_student_gt_prob", True),
        ("regret_teacher_advantage", True),
    )

    def __init__(self, args):
        super().__init__(args)
        self._extend_tracker()

        self.regret_teacher = None
        self.regret_horizon_motion_frames = int(
            args.regret_future_horizon_frames
        )
        if self.regret_horizon_motion_frames <= 0:
            raise ValueError(
                "regret_future_horizon_frames must be positive."
            )
        codec_frames_per_token = int(
            self.upper_gesture_codec.frame_size
        )
        self.regret_horizon_tokens = math.ceil(
            self.regret_horizon_motion_frames / codec_frames_per_token
        )
        self.regret_past_context_tokens = int(
            self._student_model().context
        )
        if self.regret_past_context_tokens <= 0:
            raise ValueError(
                "Regret training requires a positive causal gesture context."
            )

        self._validate_regret_hyperparameters()
        self.regret_start_epoch = self._resolve_regret_start_epoch()

        if args.is_train:
            teacher_checkpoint = args.regret_teacher_ckpt
            if not teacher_checkpoint:
                raise ValueError(
                    "UpperFaceLowerGTDM3Regret requires "
                    "--regret_teacher_ckpt."
                )
            if not os.path.isfile(teacher_checkpoint):
                raise FileNotFoundError(
                    "Regret teacher checkpoint does not exist: "
                    f"{teacher_checkpoint}"
                )
            self.regret_teacher = self._build_regret_teacher(
                teacher_checkpoint
            )

        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] Regret setup: "
            f"future horizon={self.regret_horizon_motion_frames} motion "
            f"frames/{self.regret_horizon_tokens} gesture tokens; "
            f"past context={self.regret_past_context_tokens}; "
            f"start epoch={self.regret_start_epoch}; "
            f"alpha={self.args.regret_initial_weight}->"
            f"{self.args.regret_weight} over "
            f"{self.args.regret_ramp_epochs} epochs; "
            f"temperature={self.args.regret_temperature}"
        )

    def _student_model(self):
        return self.model.module if self.args.ddp else self.model

    def _extend_tracker(self):
        old_tracker = self.tracker
        names = list(old_tracker.metric_names)
        directions = [
            old_tracker.is_higher_better[name]
            for name in old_tracker.metric_names
        ]
        for name, direction in self._REGRET_METRICS:
            if name not in names:
                names.append(name)
                directions.append(direction)
        self.tracker = other_tools.EpochTracker(names, directions)

    def _validate_regret_hyperparameters(self):
        if self.args.regret_temperature <= 0:
            raise ValueError("regret_temperature must be positive.")
        if self.args.regret_weight < 0:
            raise ValueError("regret_weight cannot be negative.")
        if not (
            0
            <= self.args.regret_initial_weight
            <= self.args.regret_weight
        ):
            raise ValueError(
                "regret_initial_weight must lie between zero and "
                "regret_weight."
            )
        if self.args.regret_ramp_epochs < 0:
            raise ValueError("regret_ramp_epochs cannot be negative.")

    def _resolve_regret_start_epoch(self) -> int:
        configured = int(self.args.regret_start_epoch)
        if configured >= 0:
            return configured
        if self.args.is_continue and self.args.continue_ckpt:
            checkpoint_name = os.path.basename(self.args.continue_ckpt)
            try:
                checkpoint_epoch = int(
                    checkpoint_name.split("_")[1].replace(
                        ".safetensors",
                        "",
                    )
                )
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    "Could not infer regret start epoch from continuation "
                    f"checkpoint {checkpoint_name!r}; pass "
                    "--regret_start_epoch explicitly."
                ) from exc
            return checkpoint_epoch + 1
        return int(self.args.pretrain_warmup_epochs)

    def _build_regret_teacher(self, checkpoint_path: str):
        student = self._student_model()
        text_procemb = (
            student.text_procemb.weight.detach().cpu().clone()
        )
        audio_procemb = [
            embedding.weight.detach().cpu().clone()
            for embedding in student.audio_procemb
        ]

        upper_layers = copy.deepcopy(
            self.upper_gesture_codec.quantizer.vq.layers
        ).cpu()
        lower_layers = copy.deepcopy(
            self.lower_gesture_codec.quantizer.vq.layers
        ).cpu()
        face_layers = copy.deepcopy(
            self.face_gesture_codec.quantizer.vq.layers
        ).cpu()
        gesture_codec_layers = upper_layers + lower_layers + face_layers
        for layer in gesture_codec_layers:
            layer.requires_grad_(False)
        gesture_codec_layers.eval()

        gesture_lm_kwargs = dict(loaders.get_gesturelm_kwargs())
        teacher = GTemporalDepthModel3FutureGestureFullCondition(
            num_heads=self.args.gestureformer_heads,
            num_layers=self.args.gestureformer_layers,
            depformer_heads=self.args.gestureformer_depformer_heads,
            depformer_layers=self.args.gestureformer_depformer_layers,
            query2mem_scale=self.codec_difference,
            num_temp_classifiers=self.args.num_temp_classifiers,
            dtype=(
                torch.float32
                if self.args.param_dtype == "float32"
                else torch.bfloat16
            ),
            text_procemb=text_procemb,
            audio_procemb=audio_procemb,
            gesture_codec_layers=gesture_codec_layers,
            # The q0-only teacher never executes the auxiliary VAD path.
            vad_guidance=False,
            vad_use_face_logits=False,
            body_parts=3,
            bp_dist=None,
            textaudio_emb_freeze=self.args.textaudio_emb_freeze,
            **gesture_lm_kwargs,
        )
        self._load_temporal_teacher_checkpoint(
            teacher,
            checkpoint_path,
        )
        retain_temporal_q0_only(teacher)
        teacher.requires_grad_(False)
        teacher.eval()
        teacher.to(self.local_rank)

        student_n_q = student.n_q
        if (
            teacher.n_q != student_n_q
            or teacher.card != student.card
            or teacher.dim != student.dim
        ):
            raise ValueError(
                "Teacher/student token architecture mismatch: "
                f"teacher=(K={teacher.n_q}, card={teacher.card}, "
                f"dim={teacher.dim}), student=(K={student_n_q}, "
                f"card={student.card}, dim={student.dim})."
            )

        parameter_count = sum(
            parameter.numel()
            for parameter in teacher.parameters()
        )
        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] Loaded frozen "
            f"temporal-only regret teacher from {checkpoint_path}; "
            f"parameters retained={parameter_count:,}"
        )
        return teacher

    @staticmethod
    def _load_temporal_teacher_checkpoint(teacher, checkpoint_path):
        checkpoint_state = {
            _canonical_checkpoint_key(name): tensor
            for name, tensor in load_file(
                checkpoint_path,
                device="cpu",
            ).items()
        }
        teacher_state = teacher.state_dict()
        required_names = {
            name
            for name in teacher_state
            if is_temporal_q0_state_key(name)
        }
        checkpoint_temporal_names = {
            name
            for name in checkpoint_state
            if is_temporal_q0_state_key(name)
        }
        missing = sorted(required_names - checkpoint_state.keys())
        if missing:
            raise RuntimeError(
                "Teacher checkpoint is missing temporal q0 tensors: "
                + ", ".join(missing[:10])
            )
        unexpected = sorted(
            checkpoint_temporal_names - required_names
        )
        if unexpected:
            raise RuntimeError(
                "Teacher checkpoint has temporal q0 tensors not present in "
                "the configured teacher architecture: "
                + ", ".join(unexpected[:10])
            )

        shape_mismatches = []
        selected_state = {}
        for name in required_names:
            checkpoint_tensor = checkpoint_state[name]
            expected_tensor = teacher_state[name]
            if checkpoint_tensor.shape != expected_tensor.shape:
                shape_mismatches.append(
                    f"{name}: checkpoint={tuple(checkpoint_tensor.shape)}, "
                    f"expected={tuple(expected_tensor.shape)}"
                )
            else:
                selected_state[name] = checkpoint_tensor
        if shape_mismatches:
            raise RuntimeError(
                "Teacher temporal q0 tensor shape mismatch: "
                + "; ".join(shape_mismatches[:10])
            )

        incompatible = teacher.load_state_dict(
            selected_state,
            strict=False,
        )
        missing_temporal = [
            name
            for name in incompatible.missing_keys
            if is_temporal_q0_state_key(name)
        ]
        if missing_temporal:
            raise RuntimeError(
                "Failed to install teacher temporal q0 tensors: "
                + ", ".join(missing_temporal[:10])
            )

    def _regret_alpha(self, epoch: int) -> float:
        if epoch < self.regret_start_epoch:
            return 0.0
        maximum = float(self.args.regret_weight)
        initial = float(self.args.regret_initial_weight)
        ramp_epochs = int(self.args.regret_ramp_epochs)
        if ramp_epochs == 0:
            return maximum
        progress = min(
            1.0,
            max(
                0.0,
                (epoch - self.regret_start_epoch) / ramp_epochs,
            ),
        )
        return initial + (maximum - initial) * progress

    def _select_regret_target_times(
        self,
        token_loss_mask,
        *,
        split: str,
        epoch: int,
        iteration: int,
    ):
        batch = token_loss_mask.shape[0]
        valid_lengths = token_loss_mask[:, 0].sum(dim=-1).long()
        maximum_targets = (
            valid_lengths - self.regret_horizon_tokens - 1
        )
        if (maximum_targets < 0).any():
            raise ValueError(
                "Sequence is too short for regret horizon: valid lengths="
                f"{valid_lengths.tolist()}, horizon="
                f"{self.regret_horizon_tokens} gesture tokens."
            )

        offsets = (
            torch.arange(
                batch,
                device=token_loss_mask.device,
            )
            + iteration * batch
            + int(self.args.random_seed)
        )
        if split == "train":
            offsets = (
                offsets
                + epoch * 1_000_003
                + self.global_rank * 97_409
            )
        hashed = (offsets * 48_271 + 1) % 2_147_483_647
        return hashed % (maximum_targets + 1)

    def _update_regret_metrics(
        self,
        *,
        split,
        raw_kl,
        weighted_kl,
        alpha,
        student_logits,
        teacher_logits,
        targets,
    ):
        temperature = float(self.args.regret_temperature)
        student_float = student_logits.float()
        teacher_float = teacher_logits.float()
        student_probs = F.softmax(
            student_float / temperature,
            dim=-1,
        )
        teacher_probs = F.softmax(
            teacher_float / temperature,
            dim=-1,
        )
        student_ce = F.cross_entropy(
            student_float,
            targets,
        )
        teacher_ce = F.cross_entropy(
            teacher_float,
            targets,
        )
        student_acc = (
            student_float.argmax(dim=-1) == targets
        ).float().mean()
        teacher_acc = (
            teacher_float.argmax(dim=-1) == targets
        ).float().mean()
        student_entropy = -(
            student_probs
            * student_probs.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        teacher_entropy = -(
            teacher_probs
            * teacher_probs.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        target_index = targets[:, None]
        student_gt_prob = student_probs.gather(
            1,
            target_index,
        ).mean()
        teacher_gt_prob = teacher_probs.gather(
            1,
            target_index,
        ).mean()
        teacher_advantage = (
            teacher_probs.gather(1, target_index)
            > student_probs.gather(1, target_index)
        ).float().mean()

        values = {
            "regret_loss": raw_kl,
            "regret_weighted": weighted_kl,
            "regret_alpha": alpha,
            "regret_teacher_q0_ce": teacher_ce,
            "regret_student_q0_ce": student_ce,
            "regret_teacher_q0_acc": teacher_acc,
            "regret_student_q0_acc": student_acc,
            "regret_teacher_entropy": teacher_entropy,
            "regret_student_entropy": student_entropy,
            "regret_teacher_gt_prob": teacher_gt_prob,
            "regret_student_gt_prob": student_gt_prob,
            "regret_teacher_advantage": teacher_advantage,
        }
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            self.tracker.update_meter(name, split, float(value))

    def _compute_regret(
        self,
        *,
        split,
        logits,
        gesture_tokens,
        token_loss_mask,
        input_codes,
        audio_codes,
        text_codes,
        sum_condition,
        epoch,
        iteration,
    ):
        if self.regret_teacher is None:
            raise RuntimeError(
                "Regret loss requested without a loaded teacher."
            )
        alpha = self._regret_alpha(epoch)
        target_times = self._select_regret_target_times(
            token_loss_mask,
            split=split,
            epoch=epoch,
            iteration=iteration,
        )
        teacher_temporal_codes = build_masked_future_gesture_inputs(
            input_codes,
            gesture_tokens,
            target_times,
            horizon_tokens=self.regret_horizon_tokens,
            past_context_tokens=self.regret_past_context_tokens,
            mask_token_id=self.modelout_ignore_index,
        )

        self.regret_teacher.eval()
        with torch.no_grad():
            _, teacher_selected_logits = (
                self.regret_teacher.forward_oracle_temporal_targets(
                    teacher_temporal_codes,
                    audio_codes,
                    text_codes,
                    sum_condition,
                    target_times,
                )
            )
        batch_indices = torch.arange(
            logits.shape[0],
            device=logits.device,
        )
        real_vocabulary = int(self.modelout_ignore_index)
        student_selected_logits = logits[
            batch_indices,
            0,
            target_times,
            :real_vocabulary,
        ]
        teacher_selected_logits = teacher_selected_logits[
            :,
            0,
            0,
            :real_vocabulary,
        ]
        targets = gesture_tokens[
            batch_indices,
            0,
            target_times,
        ]
        if (targets >= real_vocabulary).any():
            raise RuntimeError(
                "Regret selected a PAD q0 target, indicating an invalid "
                "target-time mask."
            )

        raw_kl = forward_q0_regret_kl(
            student_selected_logits,
            teacher_selected_logits,
            temperature=float(self.args.regret_temperature),
        )
        weighted_kl = (
            raw_kl
            * alpha
            / self._student_model().n_q
        )
        self._update_regret_metrics(
            split=split,
            raw_kl=raw_kl,
            weighted_kl=weighted_kl,
            alpha=alpha,
            student_logits=student_selected_logits,
            teacher_logits=teacher_selected_logits,
            targets=targets,
        )
        return weighted_kl

    def get_additional_training_loss(
        self,
        logits,
        gesture_tokens,
        pad_loss_mask,
        epoch,
        iteration,
        **batch_context,
    ):
        super().get_additional_training_loss(
            logits,
            gesture_tokens,
            pad_loss_mask,
            epoch,
            iteration,
        )
        alpha = self._regret_alpha(epoch)
        if alpha <= 0:
            self.tracker.update_meter(
                "regret_alpha",
                "train",
                0.0,
            )
            return None
        weighted = self._compute_regret(
            split="train",
            logits=logits,
            gesture_tokens=gesture_tokens,
            token_loss_mask=pad_loss_mask,
            epoch=epoch,
            iteration=iteration,
            **batch_context,
        )
        if (
            iteration % self.args.log_period == 0
            and self.global_rank == 0
        ):
            logger.info(
                f"Regret q0 KL="
                f"{self.tracker.loss_meters['regret_loss']['train'].val:.4f}, "
                f"alpha={alpha:.4f}, "
                f"weighted="
                f"{self.tracker.loss_meters['regret_weighted']['train'].val:.4f}"
            )
        return weighted

    def record_validation_diagnostics(
        self,
        logits,
        gesture_tokens,
        pad_loss_mask,
        **batch_context,
    ):
        super().record_validation_diagnostics(
            logits,
            gesture_tokens,
            pad_loss_mask,
        )
        epoch = int(batch_context.pop("epoch"))
        iteration = int(batch_context.pop("iteration"))
        alpha = self._regret_alpha(epoch)
        if alpha <= 0 or self.regret_teacher is None:
            self.tracker.update_meter(
                "regret_alpha",
                "val",
                max(0.0, alpha),
            )
            return
        self._compute_regret(
            split="val",
            logits=logits,
            gesture_tokens=gesture_tokens,
            token_loss_mask=pad_loss_mask,
            epoch=epoch,
            iteration=iteration,
            **batch_context,
        )
