"""Shared-weight GlobalRegret with the kinematic transformer's RVQ strategy.

Composes two independent, previously separate techniques:

* ``UpperFaceLowerGTDM3SharedRegretTrainer`` -- the causal student is
  self-distilled from its own bidirectional (GlobalRegret) teacher view,
  including the kinematic/depth levels, with both temporal and depth paths
  trainable.
* ``UpperFaceLowerGTDM3FrozenTemporalRVQTrainer`` -- the depth transformer
  is trained on coherent stochastic RVQ prefixes with a hard/soft-target CE
  mixture instead of deterministic-nearest-code hard CE alone.

Unlike ``UpperFaceLowerGTDM3FrozenTemporalRVQTrainer``, this trainer does
*not* freeze the temporal transformer -- only its kinematic-transformer
training strategy (stochastic RVQ prefixes + hard/soft CE) is reused here.
q0 stays an ordinary trainable head with plain hard CE, exactly as in the
released model and in ``UpperFaceLowerGTDM3SharedRegretTrainer``.

The one real interaction between the two techniques: the depth transformer's
input prefix is stochastic during training, so the GlobalRegret teacher
view's depth branch must be conditioned on that *same* stochastic prefix
(not the canonical codes) or the KL would confound "future context" with
"which RVQ prefix variant was used". See ``_teacher_depth_input_codes``
below and ``forward_teacher_view``'s ``depth_input_codes`` parameter in
``miburi/models/gesture_lm_shared_regret.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miburi.models import encode_stochastic_rvq

from .uflgtdm3_shared_regret_trainer import UpperFaceLowerGTDM3SharedRegretTrainer
from .utils import tools as other_tools


class UpperFaceLowerGTDM3SharedRegretRVQTrainer(
    UpperFaceLowerGTDM3SharedRegretTrainer
):
    """GlobalRegret self-distillation plus stochastic-RVQ depth training."""

    _RVQ_METRICS = (
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

    def __init__(self, args):
        super().__init__(args)
        self._extend_rvq_tracker()
        self._validate_rvq_hyperparameters()
        self._last_train_depth_input_codes: torch.Tensor | None = None

    def _extend_rvq_tracker(self):
        old_tracker = self.tracker
        names = list(old_tracker.metric_names)
        directions = [
            old_tracker.is_higher_better[name]
            for name in old_tracker.metric_names
        ]
        for name, direction in self._RVQ_METRICS:
            if name not in names:
                names.append(name)
                directions.append(direction)
        self.tracker = other_tools.EpochTracker(names, directions)

    def _validate_rvq_hyperparameters(self):
        if int(self.args.kinematic_rvq_topk) < 2:
            raise ValueError("kinematic_rvq_topk must be at least two.")
        if self.args.kinematic_rvq_temperature <= 0:
            raise ValueError("kinematic_rvq_temperature must be positive.")
        if not 0 <= self.args.kinematic_rvq_sample_probability <= 1:
            raise ValueError(
                "kinematic_rvq_sample_probability must lie in [0, 1]."
            )
        if not 0 <= self.args.kinematic_rvq_soft_target_weight <= 1:
            raise ValueError(
                "kinematic_rvq_soft_target_weight must lie in [0, 1]."
            )
        if self.args.kinematic_rvq_start_epoch < 0:
            raise ValueError("kinematic_rvq_start_epoch cannot be negative.")
        if self.args.kinematic_rvq_ramp_epochs < 0:
            raise ValueError("kinematic_rvq_ramp_epochs cannot be negative.")
        if self.args.kinematic_rvq_distance_chunk_size <= 0:
            raise ValueError(
                "kinematic_rvq_distance_chunk_size must be positive."
            )

    def _rvq_regularization_scale(self, epoch, iteration):
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
        scale = self._rvq_regularization_scale(epoch, iteration)
        sample_probability = (
            float(self.args.kinematic_rvq_sample_probability) * scale
        )
        soft_weight = (
            float(self.args.kinematic_rvq_soft_target_weight) * scale
        )
        self.tracker.update_meter(
            "rvq_sample_probability", "train", sample_probability,
        )
        self.tracker.update_meter(
            "kinematic_soft_weight", "train", soft_weight,
        )

        if sample_probability == 0 and soft_weight == 0:
            codes, context = super().encode_training_gesture_codes(
                codec_motion_inputs, epoch=epoch, iteration=iteration,
            )
            context.update({"soft_weight": 0.0, "rvq_results": None})
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
                temperature=float(self.args.kinematic_rvq_temperature),
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
            self._last_train_depth_input_codes = None
            return None, target_codes, token_loss_mask, encoding_context

        stochastic_codes = torch.cat(
            [result.codes for result in results], dim=1,
        )
        soft_indices = torch.cat(
            [result.topk_indices for result in results], dim=1,
        )
        soft_probabilities = torch.cat(
            [result.topk_probabilities for result in results], dim=1,
        )
        if stochastic_codes.shape != target_codes.shape:
            raise RuntimeError(
                "Stochastic/deterministic code shapes differ: "
                f"{tuple(stochastic_codes.shape)} and "
                f"{tuple(target_codes.shape)}."
            )

        # Preserve body-part dropout/PAD positions in the prefixes. Targets
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
        # q0 is trainable here (unlike the frozen-temporal trainer) but is
        # still the temporal head's own canonical target -- it never gets a
        # stochastic RVQ *target*, only (like every other level) a possibly
        # perturbed *input* prefix for whatever conditions on it downstream.
        stochastic_targets[:, 0] = target_codes[:, 0]

        loss_context = dict(encoding_context)
        loss_context.update(
            {
                "soft_indices": soft_indices,
                "soft_probabilities": soft_probabilities,
            }
        )
        self._last_train_depth_input_codes = depth_input_codes
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
        """q0: plain hard CE (trainable). k=1..K-1: hard/soft RVQ mixture."""

        B, K, T, card = logits.shape
        soft_weight = float(loss_context.get("soft_weight", 0.0))
        soft_indices = loss_context.get("soft_indices")
        soft_probabilities = loss_context.get("soft_probabilities")

        q0_logits = logits[:, 0]
        q0_targets = target_codes[:, 0]
        q0_valid = token_loss_mask[:, 0].reshape(B * T)
        q0_loss = F.cross_entropy(
            q0_logits.reshape(B * T, card),
            q0_targets.reshape(B * T),
            reduction="none",
        )
        q0_loss = (q0_loss * q0_valid).sum() / (q0_valid.sum() + 1e-12)
        q0_loss = q0_loss * (1.0 / K)

        ce_loss = q0_loss
        upper_loss = q0_loss.item()
        lower_loss = 0.0
        face_loss = 0.0

        hard_total = logits.sum() * 0.0
        soft_total = logits.sum() * 0.0
        upper_hard = logits.sum() * 0.0
        lower_hard = logits.sum() * 0.0
        face_hard = logits.sum() * 0.0

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
                    head_logits.float(), dim=-1,
                )
                head_soft_probabilities = soft_probabilities[:, k]
                soft_tokens = -(
                    head_soft_probabilities
                    * log_probabilities.gather(-1, soft_indices[:, k])
                ).sum(dim=-1)
                soft_mean = (soft_tokens * valid).sum() / denominator
            else:
                soft_mean = hard_mean

            mixed_mean = (
                (1.0 - soft_weight) * hard_mean + soft_weight * soft_mean
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
            "kinematic_hard_ce", "train", hard_total.item(),
        )
        self.tracker.update_meter(
            "kinematic_soft_ce", "train", soft_total.item(),
        )
        self.tracker.update_meter(
            "kinematic_upper_hard_ce", "train", upper_hard.item(),
        )
        self.tracker.update_meter(
            "kinematic_lower_hard_ce", "train", lower_hard.item(),
        )
        self.tracker.update_meter(
            "kinematic_face_hard_ce", "train", face_hard.item(),
        )
        return ce_loss, upper_loss, lower_loss, face_loss

    def _teacher_depth_input_codes(self, split, input_codes):
        """Feed the teacher's depth branch the same stochastic prefix.

        Only the training split ever uses a stochastic prefix (validation
        stays on the released deterministic path, matching
        ``UpperFaceLowerGTDM3FrozenTemporalRVQTrainer``), so this falls back
        to the canonical ``input_codes`` for ``split == "val"``.
        """

        if split == "train" and self._last_train_depth_input_codes is not None:
            return self._last_train_depth_input_codes
        return input_codes

    def record_validation_diagnostics(
        self,
        logits,
        gesture_tokens,
        pad_loss_mask,
        **batch_context,
    ):
        """Regret + released q0 diagnostics, plus deterministic depth CE."""

        super().record_validation_diagnostics(
            logits, gesture_tokens, pad_loss_mask, **batch_context,
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
            head_mean = (
                (token_loss * valid).sum() / valid.sum().clamp_min(1)
            )
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

        self.tracker.update_meter("kinematic_hard_ce", "val", total.item())
        for part, value in totals.items():
            self.tracker.update_meter(
                f"kinematic_{part}_hard_ce", "val", value.item(),
            )
