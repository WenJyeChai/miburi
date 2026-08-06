"""Phase-2 MIBURI training with a frozen temporal path and fresh depth path.

Only temporal-q0 tensors are loaded from the source checkpoint.  The
kinematic/depth modules retain their ordinary random/codec initialization and
are optimized with coherent stochastic RVQ prefixes and distance-aware soft
targets.  Validation remains the released deterministic hard-CE evaluation.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import load_file

from miburi.models import encode_stochastic_rvq, is_temporal_q0_state_key

from .uflgtdm3_trainer import UpperFaceLowerGTDM3Trainer
from .utils import tools as other_tools


def _canonical_checkpoint_key(name: str) -> str:
    if name.startswith("module."):
        name = name[len("module."):]
    if name.startswith("m."):
        name = name[len("m."):]
    return name


class UpperFaceLowerGTDM3FrozenTemporalRVQTrainer(
    UpperFaceLowerGTDM3Trainer
):
    """Train only a new kinematic transformer under a frozen temporal q0."""

    _METRICS = (
        ("kinematic_hard_ce", False),
        ("kinematic_soft_ce", False),
        ("kinematic_upper_hard_ce", False),
        ("kinematic_lower_hard_ce", False),
        ("kinematic_face_hard_ce", False),
        ("kinematic_soft_weight", False),
        ("rvq_sample_probability", False),
        ("rvq_sampled_nonnearest", False),
        ("rvq_changed_from_deterministic", False),
        ("rvq_target_entropy", False),
    )

    _TEMPORAL_MODULE_NAMES = (
        "audio_procemb",
        "text_procemb",
        "temporal_gemb",
        "temporal_gproj",
        "spk_tempemb",
        "temporal_transformer",
        "out_norm",
        "temporal_classifier",
        "temp_condproj",
    )

    def __init__(self, args):
        super().__init__(args)
        self._extend_tracker()
        self._validate_hyperparameters()

        if args.is_train:
            checkpoint_path = args.frozen_temporal_ckpt
            if not checkpoint_path:
                raise ValueError(
                    "UpperFaceLowerGTDM3FrozenTemporalRVQ requires "
                    "--frozen_temporal_ckpt."
                )
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    "Frozen temporal checkpoint does not exist: "
                    f"{checkpoint_path}"
                )
            self._load_temporal_checkpoint(checkpoint_path)
            self._freeze_temporal_path()
            self.configure_train_mode()
            self._log_parameter_split(checkpoint_path)

    def _student_model(self):
        return self.model.module if self.args.ddp else self.model

    def _extend_tracker(self):
        old_tracker = self.tracker
        names = list(old_tracker.metric_names)
        directions = [
            old_tracker.is_higher_better[name]
            for name in old_tracker.metric_names
        ]
        for name, direction in self._METRICS:
            if name not in names:
                names.append(name)
                directions.append(direction)
        self.tracker = other_tools.EpochTracker(names, directions)

    def _validate_hyperparameters(self):
        if int(self.args.kinematic_rvq_topk) < 2:
            raise ValueError("kinematic_rvq_topk must be at least two.")
        if self.args.kinematic_rvq_temperature <= 0:
            raise ValueError(
                "kinematic_rvq_temperature must be positive."
            )
        if not 0 <= self.args.kinematic_rvq_sample_probability <= 1:
            raise ValueError(
                "kinematic_rvq_sample_probability must lie in [0, 1]."
            )
        if not 0 <= self.args.kinematic_rvq_soft_target_weight <= 1:
            raise ValueError(
                "kinematic_rvq_soft_target_weight must lie in [0, 1]."
            )
        if self.args.kinematic_rvq_start_epoch < 0:
            raise ValueError(
                "kinematic_rvq_start_epoch cannot be negative."
            )
        if self.args.kinematic_rvq_ramp_epochs < 0:
            raise ValueError(
                "kinematic_rvq_ramp_epochs cannot be negative."
            )
        if self.args.kinematic_rvq_distance_chunk_size <= 0:
            raise ValueError(
                "kinematic_rvq_distance_chunk_size must be positive."
            )

    def _load_temporal_checkpoint(self, checkpoint_path):
        checkpoint_state = {
            _canonical_checkpoint_key(name): tensor
            for name, tensor in load_file(
                checkpoint_path,
                device="cpu",
            ).items()
        }
        model = self._student_model()
        model_state = model.state_dict()
        required_names = {
            name for name in model_state if is_temporal_q0_state_key(name)
        }
        missing = sorted(required_names - checkpoint_state.keys())
        if missing:
            raise RuntimeError(
                "Source checkpoint is missing temporal tensors: "
                + ", ".join(missing[:10])
            )

        selected_state = {}
        shape_mismatches = []
        for name in required_names:
            source = checkpoint_state[name]
            expected = model_state[name]
            if source.shape != expected.shape:
                shape_mismatches.append(
                    f"{name}: checkpoint={tuple(source.shape)}, "
                    f"expected={tuple(expected.shape)}"
                )
            else:
                selected_state[name] = source
        if shape_mismatches:
            raise RuntimeError(
                "Frozen temporal tensor shape mismatch: "
                + "; ".join(shape_mismatches[:10])
            )

        incompatible = model.load_state_dict(selected_state, strict=False)
        missing_after_load = [
            name
            for name in incompatible.missing_keys
            if is_temporal_q0_state_key(name)
        ]
        if missing_after_load:
            raise RuntimeError(
                "Failed to install frozen temporal tensors: "
                + ", ".join(missing_after_load[:10])
            )

    def _freeze_temporal_path(self):
        for name, parameter in self._student_model().named_parameters():
            if is_temporal_q0_state_key(name):
                parameter.requires_grad_(False)
                # UpperFaceLowerGTDM3Trainer prepares fp32 optimizer masters
                # before this phase-specific freeze is applied.  They are
                # never used by frozen parameters and would otherwise waste
                # substantial GPU memory.
                if hasattr(parameter, "_mp_param"):
                    del parameter._mp_param

        accidentally_trainable = [
            name
            for name, parameter in self._student_model().named_parameters()
            if is_temporal_q0_state_key(name) and parameter.requires_grad
        ]
        if accidentally_trainable:
            raise RuntimeError(
                "Temporal parameters remain trainable: "
                + ", ".join(accidentally_trainable[:10])
            )

    def configure_train_mode(self):
        """Keep frozen temporal dropout disabled after ``model.train()``."""

        model = self._student_model()
        for module_name in self._TEMPORAL_MODULE_NAMES:
            module = getattr(model, module_name, None)
            if module is not None:
                module.eval()

    def _log_parameter_split(self, checkpoint_path):
        temporal = 0
        trainable = 0
        frozen_other = 0
        for name, parameter in self._student_model().named_parameters():
            count = parameter.numel()
            if is_temporal_q0_state_key(name):
                temporal += count
            elif parameter.requires_grad:
                trainable += count
            else:
                frozen_other += count
        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] Loaded/froze "
            f"temporal path from {checkpoint_path}; "
            f"temporal={temporal:,}, trainable kinematic={trainable:,}, "
            f"other frozen={frozen_other:,} parameters."
        )

    def _regularization_scale(self, epoch, iteration):
        start = float(self.args.kinematic_rvq_start_epoch)
        fractional_epoch = epoch + iteration / max(1, self.train_length)
        if fractional_epoch < start:
            return 0.0
        ramp = float(self.args.kinematic_rvq_ramp_epochs)
        if ramp == 0:
            return 1.0
        return min(1.0, max(0.0, (fractional_epoch - start) / ramp))

    def encode_training_gesture_codes(
        self,
        codec_motion_inputs,
        *,
        epoch,
        iteration,
    ):
        scale = self._regularization_scale(epoch, iteration)
        sample_probability = (
            float(self.args.kinematic_rvq_sample_probability) * scale
        )
        soft_weight = (
            float(self.args.kinematic_rvq_soft_target_weight) * scale
        )
        self.tracker.update_meter(
            "rvq_sample_probability",
            "train",
            sample_probability,
        )
        self.tracker.update_meter(
            "kinematic_soft_weight",
            "train",
            soft_weight,
        )

        if sample_probability == 0 and soft_weight == 0:
            codes, context = super().encode_training_gesture_codes(
                codec_motion_inputs,
                epoch=epoch,
                iteration=iteration,
            )
            context.update(
                {
                    "soft_weight": 0.0,
                    "rvq_results": None,
                }
            )
            return codes, context

        results = []
        deterministic_codes = []
        codecs = (
            self.upper_gesture_codec,
            self.lower_gesture_codec,
            self.face_gesture_codec,
        )
        for codec, motion in zip(codecs, codec_motion_inputs):
            deterministic, stochastic = encode_stochastic_rvq(
                codec,
                motion,
                topk=int(self.args.kinematic_rvq_topk),
                temperature=float(
                    self.args.kinematic_rvq_temperature
                ),
                sample_probability=sample_probability,
                distance_chunk_size=int(
                    self.args.kinematic_rvq_distance_chunk_size
                ),
            )
            deterministic_codes.append(deterministic)
            results.append(stochastic)

        self.tracker.update_meter(
            "rvq_sampled_nonnearest",
            "train",
            torch.stack(
                [result.sampled_nonnearest_fraction for result in results]
            ).mean().item(),
        )
        self.tracker.update_meter(
            "rvq_changed_from_deterministic",
            "train",
            torch.stack(
                [
                    result.changed_from_deterministic_fraction
                    for result in results
                ]
            ).mean().item(),
        )
        self.tracker.update_meter(
            "rvq_target_entropy",
            "train",
            torch.stack(
                [result.target_entropy for result in results]
            ).mean().item(),
        )
        return tuple(deterministic_codes), {
            "rvq_results": results,
            "soft_weight": soft_weight,
        }

    def prepare_kinematic_training_inputs(
        self,
        input_codes,
        target_codes,
        token_loss_mask,
        *,
        epoch,
        iteration,
        encoding_context,
        **batch_context,
    ):
        del epoch, iteration, batch_context
        results = encoding_context.get("rvq_results")
        if results is None:
            return None, target_codes, token_loss_mask, encoding_context

        stochastic_codes = torch.cat(
            [result.codes for result in results],
            dim=1,
        )
        soft_indices = torch.cat(
            [result.topk_indices for result in results],
            dim=1,
        )
        soft_probabilities = torch.cat(
            [result.topk_probabilities for result in results],
            dim=1,
        )
        if stochastic_codes.shape != target_codes.shape:
            raise RuntimeError(
                "Stochastic/deterministic code shapes differ: "
                f"{tuple(stochastic_codes.shape)} and "
                f"{tuple(target_codes.shape)}."
            )

        # Preserve body-part dropout/PAD positions in the prefixes.  Targets
        # only inherit true dataset padding, not input-only body dropout.
        depth_input_codes = torch.where(
            input_codes == self.modelout_ignore_index,
            torch.full_like(stochastic_codes, self.modelout_ignore_index),
            stochastic_codes,
        )
        stochastic_targets = torch.where(
            token_loss_mask.bool(),
            stochastic_codes,
            torch.full_like(stochastic_codes, self.modelout_ignore_index),
        )
        # q0 belongs to the frozen temporal head.  Keep its diagnostic target
        # canonical and train only depth indices 1..K-1 below.
        stochastic_targets[:, 0] = target_codes[:, 0]

        loss_context = dict(encoding_context)
        loss_context.update(
            {
                "soft_indices": soft_indices,
                "soft_probabilities": soft_probabilities,
            }
        )
        return (
            depth_input_codes,
            stochastic_targets,
            token_loss_mask,
            loss_context,
        )

    def compute_training_ce_objective(
        self,
        logits,
        target_codes,
        token_loss_mask,
        *,
        loss_context,
    ):
        """Depth-only mixture of hard CE and distance-aware soft CE."""

        _, K, _, _ = logits.shape
        soft_weight = float(loss_context.get("soft_weight", 0.0))
        soft_indices = loss_context.get("soft_indices")
        soft_probabilities = loss_context.get("soft_probabilities")

        ce_loss = logits.sum() * 0.0
        hard_total = logits.sum() * 0.0
        soft_total = logits.sum() * 0.0
        upper_hard = logits.sum() * 0.0
        lower_hard = logits.sum() * 0.0
        face_hard = logits.sum() * 0.0
        upper_loss = 0.0
        lower_loss = 0.0
        face_loss = 0.0

        # k=0 is emitted by the frozen temporal classifier.  The kinematic
        # transformer predicts only k=1..K-1.
        for k in range(1, K):
            head_logits = logits[:, k]
            head_targets = target_codes[:, k]
            valid = token_loss_mask[:, k].bool()
            denominator = valid.sum().clamp_min(1)

            hard_tokens = F.cross_entropy(
                head_logits.reshape(-1, head_logits.shape[-1]),
                head_targets.reshape(-1),
                reduction="none",
            ).reshape_as(head_targets)
            hard_mean = (hard_tokens * valid).sum() / denominator

            if soft_indices is not None and soft_weight > 0:
                log_probabilities = F.log_softmax(
                    head_logits.float(),
                    dim=-1,
                )
                head_soft_probabilities = soft_probabilities[:, k]
                soft_tokens = -(
                    head_soft_probabilities
                    * log_probabilities.gather(
                        -1,
                        soft_indices[:, k],
                    )
                ).sum(dim=-1)
                soft_mean = (soft_tokens * valid).sum() / denominator
            else:
                soft_mean = hard_mean

            mixed_mean = (
                (1.0 - soft_weight) * hard_mean
                + soft_weight * soft_mean
            )
            head_scale = 1.0 / K
            part_value = (mixed_mean * head_scale).item()
            objective_scale = head_scale
            if k < 8:
                upper_loss += part_value
                upper_hard = upper_hard + hard_mean * head_scale
            elif k < 16:
                lower_loss += part_value
                lower_hard = lower_hard + hard_mean * head_scale
            else:
                face_loss += part_value
                face_hard = face_hard + hard_mean * head_scale
                objective_scale *= self.args.face_loss_weight

            ce_loss = ce_loss + mixed_mean * objective_scale
            hard_total = hard_total + hard_mean * objective_scale
            soft_total = soft_total + soft_mean * objective_scale

        self.tracker.update_meter(
            "kinematic_hard_ce",
            "train",
            hard_total.item(),
        )
        self.tracker.update_meter(
            "kinematic_soft_ce",
            "train",
            soft_total.item(),
        )
        self.tracker.update_meter(
            "kinematic_upper_hard_ce",
            "train",
            upper_hard.item(),
        )
        self.tracker.update_meter(
            "kinematic_lower_hard_ce",
            "train",
            lower_hard.item(),
        )
        self.tracker.update_meter(
            "kinematic_face_hard_ce",
            "train",
            face_hard.item(),
        )
        return ce_loss, upper_loss, lower_loss, face_loss

    def record_validation_diagnostics(
        self,
        logits,
        gesture_tokens,
        pad_loss_mask,
        **batch_context,
    ):
        """Keep normal q0 metrics and add deterministic depth-only CE."""

        super().record_validation_diagnostics(
            logits,
            gesture_tokens,
            pad_loss_mask,
            **batch_context,
        )
        _, K, _, _ = logits.shape
        totals = {
            "upper": logits.sum() * 0.0,
            "lower": logits.sum() * 0.0,
            "face": logits.sum() * 0.0,
        }
        total = logits.sum() * 0.0
        for k in range(1, K):
            valid = pad_loss_mask[:, k].bool()
            token_loss = F.cross_entropy(
                logits[:, k].reshape(-1, logits.shape[-1]),
                gesture_tokens[:, k].reshape(-1),
                reduction="none",
            ).reshape_as(gesture_tokens[:, k])
            head_mean = (token_loss * valid).sum() / valid.sum().clamp_min(1)
            head_scale = 1.0 / K
            if k < 8:
                part = "upper"
            elif k < 16:
                part = "lower"
            else:
                part = "face"
            totals[part] = totals[part] + head_mean * head_scale
            objective_scale = head_scale
            if part == "face":
                objective_scale *= self.args.face_loss_weight
            total = total + head_mean * objective_scale

        self.tracker.update_meter(
            "kinematic_hard_ce",
            "val",
            total.item(),
        )
        for part, value in totals.items():
            self.tracker.update_meter(
                f"kinematic_{part}_hard_ce",
                "val",
                value.item(),
            )
