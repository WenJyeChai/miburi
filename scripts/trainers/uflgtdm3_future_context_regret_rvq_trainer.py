"""Matched versus past-as-future training with identical sampled KL targets."""
from __future__ import annotations

import hashlib

import torch
from loguru import logger

from miburi.models.gesture_lm_past_future import (
    eligible_context_positions, forward_future_context_teacher,
    past_as_future, sample_context_positions,
)
from miburi.models.gesture_lm_shared_regret import dense_regret_kl
from .uflgtdm3_shared_regret_rvq_trainer import UpperFaceLowerGTDM3SharedRegretRVQTrainer


class UpperFaceLowerGTDM3FutureContextRegretRVQTrainer(UpperFaceLowerGTDM3SharedRegretRVQTrainer):
    """Keep full student CE/RVQ; replace only the sampled teacher KL view.

    History is taken from the current training clip, so no data-cache change
    is needed. Only later positions have enough history for this intervention.
    This is a separate trainer: all existing dense ForeMotion runs are intact.
    """

    def __init__(self, args):
        if args.regret_future_context not in ("matched", "past_as_future"):
            raise ValueError("Unknown regret_future_context")
        if args.regret_context_targets < 1:
            raise ValueError("regret_context_targets must be positive")
        if float(args.dense_future_gesture_weight) != 0:
            raise ValueError("Future-content control requires dense_future_gesture_weight=0")
        if args.varying_frame_length:
            raise ValueError("Use fixed-length aligned clips for the future-content control")
        if args.num_temp_classifiers != 1:
            raise ValueError("Future-content control requires one temporal q0 head")
        super().__init__(args)
        self._future_context_step = (0, 0)
        old = self.tracker
        extra = ["context_eligible_fraction", "context_selected_fraction", "context_changed_future_fraction"]
        names = list(old.metric_names) + extra
        from .utils import tools
        self.tracker = tools.EpochTracker(names, [old.is_higher_better.get(n, False) for n in names])
        logger.info(
            f"Future-content control: {args.regret_future_context}, up to "
            f"{args.regret_context_targets} KL targets/clip; full student CE; "
            "GlobalRegret past access; historical donor ends at current group."
        )

    def get_additional_training_loss(self, *args, **kwargs):
        # The base calls this hook with positional epoch/iteration.
        self._future_context_step = (
            kwargs.get("epoch", args[3] if len(args) > 3 else 0),
            kwargs.get("iteration", args[4] if len(args) > 4 else 0),
        )
        return super().get_additional_training_loss(*args, **kwargs)

    def record_validation_diagnostics(self, *args, **kwargs):
        # Stable validation targets across checkpoints; independent of train RNG.
        self._future_context_step = (0, 0)
        return super().record_validation_diagnostics(*args, **kwargs)

    def _compute_regret(
        self, *, split, logits, gesture_tokens, pad_loss_mask, alpha,
        input_codes, audio_codes, text_codes, sum_condition, sample_ids=None,
    ):
        model = self._student_model()
        batch, _, steps, _ = logits.shape
        valid = pad_loss_mask[:, 0].bool()
        lengths = valid.sum(-1)
        expected = torch.arange(steps, device=valid.device)[None] < lengths[:, None]
        if not torch.equal(valid, expected):
            raise ValueError("Future-content control requires contiguous valid q0 prefixes")
        scale = model.query2mem_scale
        if audio_codes.shape[-1] != steps * scale or text_codes.shape[-1] != steps * scale:
            raise ValueError("Speech and gesture clips are not aligned")
        depth = self._teacher_depth_input_codes(split, input_codes)
        cross_mask = self._build_ca_depth_padding_mask(gesture_tokens)
        students, teachers, targets, masks = [], [], [], []
        eligible_count = selected_count = changed_count = future_count = 0
        for length in sorted(set(lengths.tolist())):
            eligible = eligible_context_positions(length)
            if not eligible:
                continue
            indices = (lengths == length).nonzero(as_tuple=True)[0]
            ids = indices.tolist()
            identity = [str(sample_ids[i]) if sample_ids is not None else str(i) for i in ids]
            seed_text = repr((self.args.random_seed, split, self._future_context_step, identity, length))
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
            positions = sample_context_positions(length, self.args.regret_context_targets, seed)
            eligible_count += len(ids) * len(eligible)
            selected_count += len(ids) * len(positions)
            a, x = audio_codes[indices, :, :length * scale], text_codes[indices, :, :length * scale]
            teacher = forward_future_context_teacher(
                model, input_codes=input_codes[indices, :, :length], audio_codes=a,
                text_codes=x, sum_condition=sum_condition[indices], positions=positions,
                context=self.args.regret_future_context,
                include_depth_levels=self.regret_include_depth_levels,
                depth_input_codes=depth[indices, :, :length],
                ca_depth_padding_mask=(cross_mask[indices, :, :length] if cross_mask is not None else None),
            )
            heads = teacher.shape[1]
            def flatten(tensor):
                # [B,K,N,...] -> [B*N,K,1,...], preserving per-head KL means.
                dims = [0, 2, 1] + list(range(3, tensor.ndim))
                return tensor.permute(*dims).reshape(-1, heads, 1, *tensor.shape[3:])
            students.append(flatten(logits[indices, :heads, :length, :self.modelout_ignore_index][:, :, positions]))
            teachers.append(flatten(teacher[..., :self.modelout_ignore_index]))
            targets.append(flatten(gesture_tokens[indices, :heads, :length][:, :, positions]))
            masks.append(flatten(pad_loss_mask[indices, :heads, :length][:, :, positions].bool()))
            for t in positions:
                cut = (t + 1) * scale
                # Compare complete joint audio/text groups, not floating embeddings.
                joint = torch.cat((a, x), dim=1).transpose(1, 2)
                replacement = past_as_future(joint, t, scale)
                changed = (joint[:, cut:] != replacement[:, cut:]).any(-1)
                changed_count += changed.sum().item()
                future_count += changed.numel()
        denominator = max(1, int(valid.sum().item()))
        self.tracker.update_meter("context_eligible_fraction", split, eligible_count / denominator)
        self.tracker.update_meter("context_selected_fraction", split, selected_count / denominator)
        # Measures proposed history substitution in BOTH controls, even matched.
        self.tracker.update_meter("context_changed_future_fraction", split, changed_count / max(1, future_count))
        if not students:
            for name, value in (("regret_loss", 0), ("regret_weighted", 0), ("regret_alpha", alpha)):
                self.tracker.update_meter(name, split, value)
            return logits.sum() * 0.0
        student, teacher, target, mask = map(lambda values: torch.cat(values, dim=0), (students, teachers, targets, masks))
        raw = dense_regret_kl(student, teacher, mask, temperature=float(self.args.regret_temperature))
        weighted = raw * alpha
        self._update_regret_metrics(
            split=split, raw_kl=raw, weighted_kl=weighted, alpha=alpha,
            student_logits=student, teacher_logits=teacher, targets=target, valid_mask=mask,
        )
        return weighted
