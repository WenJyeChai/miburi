"""Word- and sentence-bounded SharedRegret-RVQ trainers for BEATX.

These trainers preserve the causal student, stochastic RVQ depth objective,
and shared-weight stop-gradient distillation used by
``UpperFaceLowerGTDM3SharedRegretRVQTrainer``.  They replace only the
teacher's unrestricted full-clip audio/text view with a per-query linguistic
cutoff recovered from BEATX's Whisper JSON timestamps.
"""

from __future__ import annotations

import os

from loguru import logger

from miburi.models import forward_linguistic_teacher_view
from miburi.models import loaders

from .uflgtdm3_shared_regret_rvq_trainer import (
    UpperFaceLowerGTDM3SharedRegretRVQTrainer,
)
from .utils import tools as other_tools
from .utils.linguistic_boundaries import WhisperLinguisticBoundaryIndex


class _UpperFaceLowerGTDM3LinguisticRegretRVQTrainer(
    UpperFaceLowerGTDM3SharedRegretRVQTrainer
):
    """Common implementation; subclasses choose ``word`` or ``sentence``."""

    linguistic_boundary_mode: str
    _LINGUISTIC_METRICS = (
        ("linguistic_lookahead_tokens", False),
        ("linguistic_lookahead_seconds", False),
        ("linguistic_active_fraction", True),
        ("linguistic_max_lookahead_tokens", False),
    )

    def __init__(self, args):
        super().__init__(args)
        self._extend_linguistic_tracker()
        self.linguistic_boundaries = WhisperLinguisticBoundaryIndex(
            args.beatx_data_path,
            mode=self.linguistic_boundary_mode,
            motion_fps=float(args.motion_fps),
            pose_length=int(args.pose_length),
            condition_frame_rate=float(loaders.FRAME_RATE),
            gesture_frame_rate=float(self.upper_gesture_codec.frame_rate),
            max_lookahead_seconds=float(
                args.linguistic_regret_max_lookahead_seconds
            ),
            transcript_subdir=args.linguistic_regret_transcript_subdir,
        )
        if args.is_train and not os.path.isdir(
            self.linguistic_boundaries.transcript_dir
        ):
            raise FileNotFoundError(
                "Linguistic regret training requires the BEATX Whisper "
                "transcript directory, but it was not found at "
                f"{self.linguistic_boundaries.transcript_dir!r}."
            )
        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] Linguistic regret "
            f"teacher: mode={self.linguistic_boundary_mode}; source="
            f"{self.linguistic_boundaries.transcript_dir}; max_lookahead="
            f"{args.linguistic_regret_max_lookahead_seconds}s (0=uncapped); "
            "outside-boundary policy=causal"
        )

    def _extend_linguistic_tracker(self):
        old_tracker = self.tracker
        names = list(old_tracker.metric_names)
        directions = [
            old_tracker.is_higher_better[name]
            for name in old_tracker.metric_names
        ]
        for name, direction in self._LINGUISTIC_METRICS:
            if name not in names:
                names.append(name)
                directions.append(direction)
        self.tracker = other_tools.EpochTracker(names, directions)

    def _forward_regret_teacher_view(
        self,
        *,
        split,
        input_codes,
        audio_codes,
        text_codes,
        sum_condition,
        ca_depth_padding_mask,
        depth_input_codes,
        sample_ids=None,
    ):
        if sample_ids is None:
            raise RuntimeError(
                "Linguistic regret requires filechunk_id metadata for every "
                "sample, but the trainer received sample_ids=None."
            )
        if len(sample_ids) != input_codes.shape[0]:
            raise ValueError(
                "Expected one filechunk_id per sequence, got "
                f"{len(sample_ids)} ids for batch {input_codes.shape[0]}."
            )
        if audio_codes.shape[-1] != text_codes.shape[-1]:
            raise ValueError(
                "Audio/text token lengths must match for a shared linguistic "
                f"boundary mask, got {audio_codes.shape[-1]} and "
                f"{text_codes.shape[-1]}."
            )

        boundary_batch = self.linguistic_boundaries.build_cross_attention_bias(
            sample_ids,
            query_length=input_codes.shape[-1],
            condition_length=audio_codes.shape[-1],
            device=audio_codes.device,
        )
        lookahead = boundary_batch.lookahead_tokens.float()
        active = boundary_batch.active_positions.float()
        condition_rate = self.linguistic_boundaries.condition_frame_rate
        self.tracker.update_meter(
            "linguistic_lookahead_tokens",
            split,
            lookahead.mean().item(),
        )
        self.tracker.update_meter(
            "linguistic_lookahead_seconds",
            split,
            (lookahead.mean() / condition_rate).item(),
        )
        self.tracker.update_meter(
            "linguistic_active_fraction",
            split,
            active.mean().item(),
        )
        self.tracker.update_meter(
            "linguistic_max_lookahead_tokens",
            split,
            lookahead.max().item(),
        )

        return forward_linguistic_teacher_view(
            self._student_model(),
            input_codes=input_codes,
            audio_codes=audio_codes,
            text_codes=text_codes,
            sum_condition=sum_condition,
            cross_attn_bias=boundary_batch.attention_bias,
            ca_depth_padding_mask=ca_depth_padding_mask,
            include_depth_levels=self.regret_include_depth_levels,
            depth_input_codes=depth_input_codes,
        )


class UpperFaceLowerGTDM3WordRegretRVQTrainer(
    _UpperFaceLowerGTDM3LinguisticRegretRVQTrainer
):
    """Teacher sees audio/text through the end of the enclosing word."""

    linguistic_boundary_mode = "word"


class UpperFaceLowerGTDM3SentenceRegretRVQTrainer(
    _UpperFaceLowerGTDM3LinguisticRegretRVQTrainer
):
    """Teacher sees audio/text through the end of the Whisper segment."""

    linguistic_boundary_mode = "sentence"
