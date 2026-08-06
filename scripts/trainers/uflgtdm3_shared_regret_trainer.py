"""Shared-weight causal-vs-GlobalRegret self-distillation for MIBURI.

Distinct from ``uflgtdm3_regret_trainer.py``, which distills a separately
pretrained, frozen full-future checkpoint (classic offline KD). This trainer
instead follows "Regret Pre-training" literally: the causal student and its
future-aware teacher view are the *same* weights, differing only in the
temporal transformer's cross-attention mask for one extra, gradient-free
forward pass per step. No teacher checkpoint, no reset-future cache, no
second pretraining run -- see ``miburi/models/gesture_lm_shared_regret.py``
for the mask mechanism and the KL loss.

Only the GlobalRegret configuration (full-clip bidirectional audio/text
cross-attention) is implemented for the dense term. LocalRegret (a genuine
one-step lookahead, rather than unbounded bidirectional context) would need
a real attention-bias primitive beyond a boolean ``causal`` toggle and is
left as follow-up work rather than approximated.

A second, optional, sparse term (off by default, ``sparse_future_gesture_weight
== 0``) additionally distills from a masked-target, bidirectional-*self*-
attention teacher view: one selected timestep per sample has its input
tokens hidden and gesture self-attention relaxed instead of cross-attention,
isolating future *gesture* information from the future speech/text signal
the dense term supplies. See ``forward_masked_target_teacher_view`` in
``miburi/models/gesture_lm_shared_regret.py`` for what it does and does not
guard against (it is a cheap, leak-permissive upper-bound check, not the
leak-safe reset-encoded view the old reset-future trainers use).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from loguru import logger

from miburi.models import (
    dense_regret_kl,
    forward_masked_target_teacher_view,
    forward_teacher_view,
)

from .uflgtdm3_trainer import UpperFaceLowerGTDM3Trainer
from .utils import tools as other_tools


class UpperFaceLowerGTDM3SharedRegretTrainer(UpperFaceLowerGTDM3Trainer):
    """Causal student self-distilled from its own bidirectional teacher view.

    The teacher view relaxes the temporal transformer's audio/text
    cross-attention to see the full clip (GlobalRegret); gesture
    self-attention is left untouched in both views since training is fully
    teacher-forced and there is no same-stream "answer" to leak through it.
    The regret KL is computed densely over every valid position and, unlike
    ``UpperFaceLowerGTDM3RegretTrainer``, over every RVQ level (q0 *and* the
    kinematic/depth levels) rather than q0 alone: the depth transformer is
    shared weights too, so its teacher-view distribution comes for free by
    feeding it the teacher's temporal hidden state instead of the student's.
    """

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
        ("regret_depth_teacher_ce", False),
        ("regret_depth_student_ce", False),
        ("regret_depth_teacher_acc", True),
        ("regret_depth_student_acc", True),
        ("regret_depth_teacher_entropy", False),
        ("regret_depth_student_entropy", False),
        ("regret_depth_teacher_gt_prob", True),
        ("regret_depth_student_gt_prob", True),
        ("regret_depth_teacher_advantage", True),
    )

    _Q0_METRIC_NAMES = {
        "teacher_ce": "regret_teacher_q0_ce",
        "student_ce": "regret_student_q0_ce",
        "teacher_acc": "regret_teacher_q0_acc",
        "student_acc": "regret_student_q0_acc",
        "teacher_entropy": "regret_teacher_entropy",
        "student_entropy": "regret_student_entropy",
        "teacher_gt_prob": "regret_teacher_gt_prob",
        "student_gt_prob": "regret_student_gt_prob",
        "teacher_advantage": "regret_teacher_advantage",
    }
    _DEPTH_METRIC_NAMES = {
        "teacher_ce": "regret_depth_teacher_ce",
        "student_ce": "regret_depth_student_ce",
        "teacher_acc": "regret_depth_teacher_acc",
        "student_acc": "regret_depth_student_acc",
        "teacher_entropy": "regret_depth_teacher_entropy",
        "student_entropy": "regret_depth_student_entropy",
        "teacher_gt_prob": "regret_depth_teacher_gt_prob",
        "student_gt_prob": "regret_depth_student_gt_prob",
        "teacher_advantage": "regret_depth_teacher_advantage",
    }

    _SPARSE_METRICS = (
        ("sparse_future_gesture_loss", False),
        ("sparse_future_gesture_weighted", False),
        ("sparse_future_gesture_alpha", False),
        ("sparse_fg_teacher_q0_ce", False),
        ("sparse_fg_student_q0_ce", False),
        ("sparse_fg_teacher_q0_acc", True),
        ("sparse_fg_student_q0_acc", True),
        ("sparse_fg_teacher_entropy", False),
        ("sparse_fg_student_entropy", False),
        ("sparse_fg_teacher_gt_prob", True),
        ("sparse_fg_student_gt_prob", True),
        ("sparse_fg_teacher_advantage", True),
        ("sparse_fg_depth_teacher_ce", False),
        ("sparse_fg_depth_student_ce", False),
        ("sparse_fg_depth_teacher_acc", True),
        ("sparse_fg_depth_student_acc", True),
        ("sparse_fg_depth_teacher_entropy", False),
        ("sparse_fg_depth_student_entropy", False),
        ("sparse_fg_depth_teacher_gt_prob", True),
        ("sparse_fg_depth_student_gt_prob", True),
        ("sparse_fg_depth_teacher_advantage", True),
    )

    _SPARSE_Q0_METRIC_NAMES = {
        "teacher_ce": "sparse_fg_teacher_q0_ce",
        "student_ce": "sparse_fg_student_q0_ce",
        "teacher_acc": "sparse_fg_teacher_q0_acc",
        "student_acc": "sparse_fg_student_q0_acc",
        "teacher_entropy": "sparse_fg_teacher_entropy",
        "student_entropy": "sparse_fg_student_entropy",
        "teacher_gt_prob": "sparse_fg_teacher_gt_prob",
        "student_gt_prob": "sparse_fg_student_gt_prob",
        "teacher_advantage": "sparse_fg_teacher_advantage",
    }
    _SPARSE_DEPTH_METRIC_NAMES = {
        "teacher_ce": "sparse_fg_depth_teacher_ce",
        "student_ce": "sparse_fg_depth_student_ce",
        "teacher_acc": "sparse_fg_depth_teacher_acc",
        "student_acc": "sparse_fg_depth_student_acc",
        "teacher_entropy": "sparse_fg_depth_teacher_entropy",
        "student_entropy": "sparse_fg_depth_student_entropy",
        "teacher_gt_prob": "sparse_fg_depth_teacher_gt_prob",
        "student_gt_prob": "sparse_fg_depth_student_gt_prob",
        "teacher_advantage": "sparse_fg_depth_teacher_advantage",
    }

    def __init__(self, args):
        super().__init__(args)
        self._extend_tracker()

        self.regret_include_depth_levels = bool(
            getattr(args, "regret_include_depth_levels", True)
        )
        self._validate_regret_hyperparameters()
        self.regret_start_epoch = self._resolve_regret_start_epoch()

        self.sparse_future_gesture_weight = float(
            getattr(args, "sparse_future_gesture_weight", 0.0)
        )
        self.sparse_future_gesture_horizon_tokens = int(
            getattr(args, "sparse_future_gesture_horizon_tokens", 1)
        )
        self.sparse_future_gesture_past_context_tokens = int(
            self._student_model().context
        )
        self._validate_sparse_future_gesture_hyperparameters()

        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] Shared-weight "
            f"GlobalRegret setup: depth_levels="
            f"{self.regret_include_depth_levels}; "
            f"start_epoch={self.regret_start_epoch}; "
            f"alpha={self.args.regret_initial_weight}->"
            f"{self.args.regret_weight} over "
            f"{self.args.regret_ramp_epochs} epochs; "
            f"temperature={self.args.regret_temperature}"
        )
        if self.sparse_future_gesture_weight > 0:
            logger.info(
                f"[GPU{self.global_rank}:{self.local_rank}] Sparse "
                "future-gesture regret enabled: weight="
                f"{self.sparse_future_gesture_weight} (rides the same ramp "
                "as regret_alpha, rescaled); horizon_tokens="
                f"{self.sparse_future_gesture_horizon_tokens}; "
                "cross-attention stays causal in this view."
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
        for name, direction in self._REGRET_METRICS + self._SPARSE_METRICS:
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

    def _validate_sparse_future_gesture_hyperparameters(self):
        if self.sparse_future_gesture_weight < 0:
            raise ValueError(
                "sparse_future_gesture_weight cannot be negative."
            )
        if self.sparse_future_gesture_horizon_tokens < 1:
            raise ValueError(
                "sparse_future_gesture_horizon_tokens must be at least one "
                "(one masks only the literal target token)."
            )
        if self.sparse_future_gesture_past_context_tokens <= 0:
            raise ValueError(
                "Sparse future-gesture regret requires a positive causal "
                "gesture context."
            )

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

    def _sparse_future_gesture_alpha(self, epoch: int) -> float:
        """Rides ``_regret_alpha``'s ramp shape, rescaled to its own weight.

        Reuses ``regret_start_epoch``/``regret_ramp_epochs``/
        ``regret_initial_weight`` rather than introducing a second schedule:
        this term only ever activates once (and proportionally to how far
        into) the dense regret term already has, just scaled to
        ``sparse_future_gesture_weight`` instead of ``regret_weight``.
        """

        if self.sparse_future_gesture_weight <= 0:
            return 0.0
        dense_alpha = self._regret_alpha(epoch)
        dense_maximum = float(self.args.regret_weight)
        if dense_maximum <= 0:
            return 0.0
        progress = dense_alpha / dense_maximum
        return progress * self.sparse_future_gesture_weight

    def _teacher_depth_input_codes(self, split, input_codes):
        """Extension point: which prefix feeds the teacher's depth branch.

        Defaults to the same ``input_codes`` the temporal branch uses
        (this trainer's own behavior). A subclass that also regularizes the
        depth transformer's input prefix (e.g. stochastic-RVQ sampling)
        should override this to return that same prefix here, so the
        teacher and student depth branches are conditioned on identical
        inputs and the KL only reflects future-context, not prefix choice.
        """

        del split
        return input_codes

    def _build_ca_depth_padding_mask(self, gesture_tokens):
        if not getattr(self.args, "drop_lower_crossattn", False):
            return None
        mask = gesture_tokens.new_zeros(gesture_tokens.shape, dtype=torch.bool)
        upper_k = self.upper_gesture_codec.num_codebooks
        lower_k = self.lower_gesture_codec.num_codebooks
        mask[:, upper_k:upper_k + lower_k, :] = True
        return mask

    def _select_sparse_future_gesture_target_times(
        self,
        token_loss_mask,
        *,
        split,
        epoch,
        iteration,
    ):
        """One deterministic target position per sample.

        Same hash-based selection ``UpperFaceLowerGTDM3RegretTrainer`` uses
        for its single-target-per-sample KL: not random-per-call, so the
        same (epoch, iteration, rank, batch-row) always maps to the same
        target, which keeps runs reproducible and comparable across ranks.
        """

        batch = token_loss_mask.shape[0]
        valid_lengths = token_loss_mask[:, 0].sum(dim=-1).long()
        maximum_targets = (
            valid_lengths - self.sparse_future_gesture_horizon_tokens - 1
        )
        if (maximum_targets < 0).any():
            raise ValueError(
                "Sequence is too short for the sparse future-gesture "
                f"horizon: valid lengths={valid_lengths.tolist()}, "
                f"horizon={self.sparse_future_gesture_horizon_tokens} "
                "gesture tokens."
            )
        offsets = (
            torch.arange(batch, device=token_loss_mask.device)
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

    @staticmethod
    def _sparse_position_mask(pad_loss_mask, target_times):
        """``[B,K,T]`` mask, True only at each sample's selected timestep.

        Still respects ordinary validity (PAD / dropped body parts) at that
        position -- a selected target that happens to land on a dropped
        lower/face codebook contributes nothing for that codebook, exactly
        like the dense term's masking.
        """

        B, K, T = pad_loss_mask.shape
        position = torch.zeros(
            B, T, dtype=torch.bool, device=pad_loss_mask.device,
        )
        position[torch.arange(B, device=pad_loss_mask.device), target_times] = True
        position = position.unsqueeze(1).expand(-1, K, -1)
        return position & pad_loss_mask.bool()

    @staticmethod
    def _regret_stat_dict(student_logits, teacher_logits, targets, temperature):
        student_float = student_logits.float()
        teacher_float = teacher_logits.float()
        student_probs = F.softmax(student_float / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_float / temperature, dim=-1)
        student_ce = F.cross_entropy(student_float, targets)
        teacher_ce = F.cross_entropy(teacher_float, targets)
        student_acc = (
            student_float.argmax(dim=-1) == targets
        ).float().mean()
        teacher_acc = (
            teacher_float.argmax(dim=-1) == targets
        ).float().mean()
        student_entropy = -(
            student_probs * student_probs.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        teacher_entropy = -(
            teacher_probs * teacher_probs.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        target_index = targets[:, None]
        student_gt_prob = student_probs.gather(1, target_index).mean()
        teacher_gt_prob = teacher_probs.gather(1, target_index).mean()
        teacher_advantage = (
            teacher_probs.gather(1, target_index)
            > student_probs.gather(1, target_index)
        ).float().mean()
        return {
            "teacher_ce": teacher_ce,
            "student_ce": student_ce,
            "teacher_acc": teacher_acc,
            "student_acc": student_acc,
            "teacher_entropy": teacher_entropy,
            "student_entropy": student_entropy,
            "teacher_gt_prob": teacher_gt_prob,
            "student_gt_prob": student_gt_prob,
            "teacher_advantage": teacher_advantage,
        }

    def _update_slice_metrics(
        self,
        split,
        student_logits,
        teacher_logits,
        targets,
        valid_mask,
        *,
        name_map,
    ):
        if not bool(valid_mask.any()):
            return
        flat_student = student_logits[valid_mask]
        flat_teacher = teacher_logits[valid_mask]
        flat_targets = targets[valid_mask]
        stats = self._regret_stat_dict(
            flat_student,
            flat_teacher,
            flat_targets,
            float(self.args.regret_temperature),
        )
        for key, value in stats.items():
            self.tracker.update_meter(
                name_map[key],
                split,
                float(value.detach().item()),
            )

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
        valid_mask,
    ):
        self.tracker.update_meter(
            "regret_loss", split, float(raw_kl.detach().item()),
        )
        self.tracker.update_meter(
            "regret_weighted", split, float(weighted_kl.detach().item()),
        )
        self.tracker.update_meter("regret_alpha", split, float(alpha))

        self._update_slice_metrics(
            split,
            student_logits[:, 0],
            teacher_logits[:, 0],
            targets[:, 0],
            valid_mask[:, 0],
            name_map=self._Q0_METRIC_NAMES,
        )
        if student_logits.shape[1] > 1:
            self._update_slice_metrics(
                split,
                student_logits[:, 1:],
                teacher_logits[:, 1:],
                targets[:, 1:],
                valid_mask[:, 1:],
                name_map=self._DEPTH_METRIC_NAMES,
            )

    def _update_sparse_future_gesture_metrics(
        self,
        *,
        split,
        raw_kl,
        weighted_kl,
        alpha,
        student_logits,
        teacher_logits,
        targets,
        valid_mask,
    ):
        self.tracker.update_meter(
            "sparse_future_gesture_loss", split, float(raw_kl.detach().item()),
        )
        self.tracker.update_meter(
            "sparse_future_gesture_weighted",
            split,
            float(weighted_kl.detach().item()),
        )
        self.tracker.update_meter(
            "sparse_future_gesture_alpha", split, float(alpha),
        )

        self._update_slice_metrics(
            split,
            student_logits[:, 0],
            teacher_logits[:, 0],
            targets[:, 0],
            valid_mask[:, 0],
            name_map=self._SPARSE_Q0_METRIC_NAMES,
        )
        if student_logits.shape[1] > 1:
            self._update_slice_metrics(
                split,
                student_logits[:, 1:],
                teacher_logits[:, 1:],
                targets[:, 1:],
                valid_mask[:, 1:],
                name_map=self._SPARSE_DEPTH_METRIC_NAMES,
            )

    def _compute_regret(
        self,
        *,
        split,
        logits,
        gesture_tokens,
        pad_loss_mask,
        alpha,
        input_codes,
        audio_codes,
        text_codes,
        sum_condition,
        sample_ids=None,
    ):
        del sample_ids
        student = self._student_model()
        real_vocab = int(self.modelout_ignore_index)

        ca_depth_padding_mask = self._build_ca_depth_padding_mask(
            gesture_tokens,
        )
        teacher_temp_logits, teacher_depth_logits = forward_teacher_view(
            student,
            input_codes=input_codes,
            audio_codes=audio_codes,
            text_codes=text_codes,
            sum_condition=sum_condition,
            ca_depth_padding_mask=ca_depth_padding_mask,
            include_depth_levels=self.regret_include_depth_levels,
            depth_input_codes=self._teacher_depth_input_codes(
                split, input_codes,
            ),
        )
        if self.regret_include_depth_levels:
            teacher_logits = torch.cat(
                [teacher_temp_logits, teacher_depth_logits], dim=1,
            )
        else:
            teacher_logits = teacher_temp_logits

        K = teacher_logits.shape[1]
        student_logits = logits[:, :K, :, :real_vocab]
        teacher_logits = teacher_logits[:, :, :, :real_vocab]
        valid_mask = pad_loss_mask[:, :K].bool()
        targets = gesture_tokens[:, :K]

        raw_kl = dense_regret_kl(
            student_logits,
            teacher_logits,
            valid_mask,
            temperature=float(self.args.regret_temperature),
        )
        weighted_kl = raw_kl * alpha

        self._update_regret_metrics(
            split=split,
            raw_kl=raw_kl,
            weighted_kl=weighted_kl,
            alpha=alpha,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            valid_mask=valid_mask,
        )
        return weighted_kl

    def _compute_sparse_future_gesture_regret(
        self,
        *,
        split,
        logits,
        gesture_tokens,
        pad_loss_mask,
        epoch,
        iteration,
        input_codes,
        audio_codes,
        text_codes,
        sum_condition,
        sample_ids=None,
    ):
        del sample_ids
        alpha = self._sparse_future_gesture_alpha(epoch)
        if alpha <= 0:
            self.tracker.update_meter(
                "sparse_future_gesture_alpha", split, 0.0,
            )
            return None

        student = self._student_model()
        real_vocab = int(self.modelout_ignore_index)
        target_times = self._select_sparse_future_gesture_target_times(
            pad_loss_mask, split=split, epoch=epoch, iteration=iteration,
        )
        ca_depth_padding_mask = self._build_ca_depth_padding_mask(
            gesture_tokens,
        )
        teacher_temp_logits, teacher_depth_logits = (
            forward_masked_target_teacher_view(
                student,
                input_codes=input_codes,
                target_codes=gesture_tokens,
                target_times=target_times,
                audio_codes=audio_codes,
                text_codes=text_codes,
                sum_condition=sum_condition,
                horizon_tokens=self.sparse_future_gesture_horizon_tokens,
                past_context_tokens=(
                    self.sparse_future_gesture_past_context_tokens
                ),
                mask_token_id=self.modelout_ignore_index,
                ca_depth_padding_mask=ca_depth_padding_mask,
                include_depth_levels=self.regret_include_depth_levels,
                depth_input_codes=self._teacher_depth_input_codes(
                    split, input_codes,
                ),
            )
        )
        if self.regret_include_depth_levels:
            teacher_logits = torch.cat(
                [teacher_temp_logits, teacher_depth_logits], dim=1,
            )
        else:
            teacher_logits = teacher_temp_logits

        K = teacher_logits.shape[1]
        student_logits = logits[:, :K, :, :real_vocab]
        teacher_logits = teacher_logits[:, :, :, :real_vocab]
        position_mask = self._sparse_position_mask(
            pad_loss_mask[:, :K], target_times,
        )
        targets = gesture_tokens[:, :K]

        raw_kl = dense_regret_kl(
            student_logits,
            teacher_logits,
            position_mask,
            temperature=float(self.args.regret_temperature),
        )
        weighted_kl = raw_kl * alpha

        self._update_sparse_future_gesture_metrics(
            split=split,
            raw_kl=raw_kl,
            weighted_kl=weighted_kl,
            alpha=alpha,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            valid_mask=position_mask,
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
        batch_context.pop("sample_ids", None)
        alpha = self._regret_alpha(epoch)
        total = None
        if alpha > 0:
            total = self._compute_regret(
                split="train",
                logits=logits,
                gesture_tokens=gesture_tokens,
                pad_loss_mask=pad_loss_mask,
                alpha=alpha,
                **batch_context,
            )
            if (
                iteration % self.args.log_period == 0
                and self.global_rank == 0
            ):
                logger.info(
                    f"SharedRegret KL="
                    f"{self.tracker.loss_meters['regret_loss']['train'].val:.4f}, "
                    f"alpha={alpha:.4f}, "
                    f"weighted="
                    f"{self.tracker.loss_meters['regret_weighted']['train'].val:.4f}"
                )
        else:
            self.tracker.update_meter("regret_alpha", "train", 0.0)

        if self.sparse_future_gesture_weight > 0:
            sparse_weighted = self._compute_sparse_future_gesture_regret(
                split="train",
                logits=logits,
                gesture_tokens=gesture_tokens,
                pad_loss_mask=pad_loss_mask,
                epoch=epoch,
                iteration=iteration,
                **batch_context,
            )
            if sparse_weighted is not None:
                total = (
                    sparse_weighted if total is None
                    else total + sparse_weighted
                )
        return total

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
        iteration = int(batch_context.pop("iteration", 0))
        batch_context.pop("sample_ids", None)
        alpha = self._regret_alpha(epoch)
        if alpha > 0:
            self._compute_regret(
                split="val",
                logits=logits,
                gesture_tokens=gesture_tokens,
                pad_loss_mask=pad_loss_mask,
                alpha=alpha,
                **batch_context,
            )
        else:
            self.tracker.update_meter(
                "regret_alpha", "val", max(0.0, alpha),
            )

        if self.sparse_future_gesture_weight > 0:
            self._compute_sparse_future_gesture_regret(
                split="val",
                logits=logits,
                gesture_tokens=gesture_tokens,
                pad_loss_mask=pad_loss_mask,
                epoch=epoch,
                iteration=iteration,
                **batch_context,
            )
