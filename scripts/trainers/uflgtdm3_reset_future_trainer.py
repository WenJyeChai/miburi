"""Full-condition future teacher with independently reset-encoded motion.

T0 and T1 already exist as the released causal trainer and the masked
full-condition future trainer.  This T2 trainer keeps T1's architecture,
initialization policy, and selected-frame objective, but replaces every intact
future upper/lower/face token with a fresh encoding of a globally preprocessed
raw-motion window. The fixed-cache pilot uses a reproducible target manifest.
"""

from __future__ import annotations

from loguru import logger

from miburi.models import (
    GTemporalDepthModel3FutureGestureFullCondition,
    build_reset_future_teacher_inputs,
    build_reset_future_teacher_inputs_from_codes,
)
from .dataloaders.utils.reset_future_cache import (
    ResetFutureManifestCache,
)

from .uflgtdm3_future_gesture_trainer import (
    _UpperFaceLowerGTDM3FutureGestureTrainer,
)


class UpperFaceLowerGTDM3ResetFutureTrainer(
    _UpperFaceLowerGTDM3FutureGestureTrainer
):
    """T2 reset-suffix teacher with complete paired audio/text context."""

    model_class = GTemporalDepthModel3FutureGestureFullCondition
    expose_future_audio_text = True

    def __init__(self, args):
        super().__init__(args)

        if args.future_motion_mode != "reset":
            raise ValueError(
                "UpperFaceLowerGTDM3ResetFuture requires "
                "future_motion_mode=reset."
            )
        self.reset_prefix_drop_tokens = int(
            args.reset_prefix_drop_tokens
        )
        if self.reset_prefix_drop_tokens < 0:
            raise ValueError(
                "reset_prefix_drop_tokens cannot be negative."
            )
        if args.future_segment_embedding:
            raise NotImplementedError(
                "The first T2 comparison intentionally reuses T1's exact "
                "parameter-free architecture. Set future_segment_embedding="
                "False. A learned reset-segment embedding belongs in a later "
                "architecture ablation."
            )

        frame_size = int(self.upper_gesture_codec.frame_size)
        if self.future_horizon_motion_frames % frame_size:
            raise ValueError(
                "T2's raw suffix must begin on an exact gesture-token "
                "boundary: future_gesture_horizon_frames="
                f"{self.future_horizon_motion_frames}, frame_size="
                f"{frame_size}."
            )
        self.maximum_reset_suffix_frames = None
        self.reset_future_window_tokens = None
        future_window_ms = float(args.future_window_ms)
        if future_window_ms < 0:
            raise ValueError("future_window_ms cannot be negative.")
        if future_window_ms > 0:
            requested_frames = (
                future_window_ms
                * float(args.motion_fps)
                / 1000.0
            )
            rounded_frames = round(requested_frames)
            if abs(requested_frames - rounded_frames) > 1e-6:
                raise ValueError(
                    f"future_window_ms={future_window_ms} does not map to "
                    f"an integer frame count at {args.motion_fps} FPS."
                )
            if rounded_frames % frame_size:
                raise ValueError(
                    f"future_window_ms={future_window_ms} maps to "
                    f"{rounded_frames} frames, which is not aligned to the "
                    f"{frame_size}-frame gesture-token boundary."
                )
            self.maximum_reset_suffix_frames = int(rounded_frames)
            self.reset_future_window_tokens = (
                self.maximum_reset_suffix_frames // frame_size
            )
            visible_window_tokens = (
                self.reset_future_window_tokens
                - self.reset_prefix_drop_tokens
            )
            if visible_window_tokens <= 0:
                raise ValueError(
                    "The configured reset future window contains no visible "
                    "token after reset_prefix_drop_tokens."
                )

        # lower = 9 joints * rot6d + xyz velocity + four contacts. Derive the
        # velocity offset from the configured feature width rather than
        # duplicating the joint count.
        velocity_start = int(args.lowertrans_nfeats) - 7
        if velocity_start < 0:
            raise ValueError(
                "lowertrans_nfeats must include xyz velocity and four "
                "contact features."
            )
        self.lower_velocity_feature_slice = (
            velocity_start,
            velocity_start + 3,
        )
        self._logged_reset_boundaries: set[str] = set()
        self._logged_cache_fallback = False
        self._last_reset_future_valid_mask = None
        self._last_reset_future_boundaries = None
        self.reset_future_cache_mode = str(
            args.reset_future_cache_mode
        )
        self.reset_future_cache = None
        if self.reset_future_cache_mode != "off":
            if self.reset_future_window_tokens is None:
                raise ValueError(
                    "Cached T2 training requires a fixed positive "
                    "future_window_ms."
                )
            if not args.reset_future_cache_dir:
                if self.reset_future_cache_mode == "required":
                    raise ValueError(
                        "reset_future_cache_mode=required needs "
                        "reset_future_cache_dir."
                    )
                logger.warning(
                    "Reset cache mode=prefer but no cache directory was "
                    "configured; falling back to online encoding."
                )
            else:
                cardinalities = {
                    int(codec.cardinality)
                    for codec in (
                        self.upper_gesture_codec,
                        self.lower_gesture_codec,
                        self.face_gesture_codec,
                    )
                }
                if len(cardinalities) != 1:
                    raise RuntimeError(
                        "Gesture codec cardinalities do not match."
                    )
                try:
                    self.reset_future_cache = ResetFutureManifestCache(
                        args.reset_future_cache_dir,
                        args=args,
                        expected_codebooks=sum(
                            int(codec.num_codebooks)
                            for codec in (
                                self.upper_gesture_codec,
                                self.lower_gesture_codec,
                                self.face_gesture_codec,
                            )
                        ),
                        expected_cardinality=cardinalities.pop(),
                        require_complete=True,
                    )
                except Exception:
                    if self.reset_future_cache_mode == "required":
                        raise
                    logger.exception(
                        "Reset cache could not be opened; mode=prefer will "
                        "use online fixed-window encoding."
                    )

        suffix_description = (
            "remaining aligned clip"
            if self.maximum_reset_suffix_frames is None
            else f"{self.maximum_reset_suffix_frames} motion frames"
        )
        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] T2 reset-future "
            f"teacher: suffix={suffix_description}; "
            f"prefix_drop={self.reset_prefix_drop_tokens} tokens; "
            "segment_embedding=False; lower boundary velocity is zeroed."
        )
        if self.reset_future_cache is not None:
            logger.info(
                f"[GPU{self.global_rank}:{self.local_rank}] T2 fixed "
                f"manifest cache loaded from "
                f"{args.reset_future_cache_dir}; targets/clip="
                f"{args.reset_future_targets_per_clip}; window="
                f"{self.reset_future_window_tokens} tokens."
            )

    def minimum_future_anchor_tokens(self):
        if self.reset_future_window_tokens is not None:
            return (
                self.future_horizon_tokens
                + self.reset_future_window_tokens
                - 1
            )
        return (
            self.future_horizon_tokens
            + self.reset_prefix_drop_tokens
        )

    def visible_future_offset_seconds(self):
        return (
            (
                self.future_horizon_tokens
                + self.reset_prefix_drop_tokens
            )
            * self.args.frame_chunk_size
            / self.args.motion_fps
        )

    def _build_reset_temporal_codes(
        self,
        input_codes,
        target_codes,
        target_times,
        codec_motion_inputs,
        *,
        reset_code_windows=None,
    ):
        if reset_code_windows is None:
            temporal_codes, valid_mask, boundaries = (
                build_reset_future_teacher_inputs(
                    input_codes,
                    target_codes,
                    target_times,
                    codec_motion_inputs=codec_motion_inputs,
                    gesture_codecs=(
                        self.upper_gesture_codec,
                        self.lower_gesture_codec,
                        self.face_gesture_codec,
                    ),
                    horizon_tokens=self.future_horizon_tokens,
                    past_context_tokens=self.past_context_tokens,
                    mask_token_id=self.modelout_ignore_index,
                    motion_fps=int(self.args.motion_fps),
                    reset_prefix_drop_tokens=(
                        self.reset_prefix_drop_tokens
                    ),
                    maximum_suffix_frames=(
                        self.maximum_reset_suffix_frames
                    ),
                    lower_velocity_feature_slice=(
                        self.lower_velocity_feature_slice
                    ),
                )
            )
        else:
            temporal_codes, valid_mask, boundaries = (
                build_reset_future_teacher_inputs_from_codes(
                    input_codes,
                    target_codes,
                    target_times,
                    reset_code_windows=reset_code_windows,
                    horizon_tokens=self.future_horizon_tokens,
                    past_context_tokens=self.past_context_tokens,
                    mask_token_id=self.modelout_ignore_index,
                    motion_fps=int(self.args.motion_fps),
                    frame_size=int(
                        self.upper_gesture_codec.frame_size
                    ),
                    reset_prefix_drop_tokens=(
                        self.reset_prefix_drop_tokens
                    ),
                )
            )
        self._last_reset_future_valid_mask = valid_mask.detach()
        self._last_reset_future_boundaries = boundaries
        return temporal_codes

    def _log_reset_boundaries(self, split, boundaries):
        if split in self._logged_reset_boundaries or not boundaries:
            return
        first = boundaries[0]
        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] T2 {split} boundary "
            f"check: target token/frame={first.target_token}/"
            f"{first.target_frame}; suffix token/frame="
            f"{first.suffix_start_token}/{first.suffix_start_frame}; "
            f"offset={first.offset_ms:.1f}ms; encoded="
            f"{first.encoded_tokens} tokens; visible="
            f"{first.visible_tokens}; prefix_drop="
            f"{self.reset_prefix_drop_tokens}."
        )
        self._logged_reset_boundaries.add(split)

    def prepare_temporal_teacher_inputs(
        self,
        input_codes,
        target_codes,
        audio_codes,
        text_codes,
        token_loss_mask,
        split,
        epoch,
        iteration,
        **batch_context,
    ):
        codec_motion_inputs = batch_context.pop(
            "codec_motion_inputs",
            None,
        )
        sample_ids = batch_context.pop("sample_ids", None)
        if batch_context:
            raise ValueError(
                "Unexpected reset-future batch context: "
                + ", ".join(sorted(batch_context))
            )
        if (
            codec_motion_inputs is None
            and self.reset_future_cache is None
        ):
            raise RuntimeError(
                "T2 reset-future training requires globally preprocessed "
                "upper/lower/face codec motion inputs."
            )
        reset_code_windows = None
        if self.reset_future_cache is not None:
            if sample_ids is None:
                raise RuntimeError(
                    "Cached T2 training requires filechunk_id sample keys."
                )
            try:
                target_times, cached_windows = (
                    self.reset_future_cache.load_batch(
                        sample_ids,
                        split=split,
                        epoch=epoch,
                    )
                )
                target_times = target_times.to(
                    token_loss_mask.device
                )
                reset_code_windows = [
                    window.codes[
                        :,
                        :window.valid_future_tokens,
                    ]
                    for window in cached_windows
                ]
            except Exception:
                if self.reset_future_cache_mode == "required":
                    raise
                if not self._logged_cache_fallback:
                    logger.exception(
                        "Reset cache lookup failed; mode=prefer is falling "
                        "back to online fixed-window encoding."
                    )
                    self._logged_cache_fallback = True
                target_times = self._select_target_times(
                    token_loss_mask,
                    split,
                    epoch,
                    iteration,
                )
        else:
            target_times = self._select_target_times(
                token_loss_mask,
                split,
                epoch,
                iteration,
            )
        valid_lengths = token_loss_mask[:, 0].sum(dim=-1).long()
        required_ends = (
            target_times
            + self.future_horizon_tokens
            + (
                self.reset_future_window_tokens
                if self.reset_future_window_tokens is not None
                else 1
            )
        )
        if (required_ends > valid_lengths).any():
            raise RuntimeError(
                "Fixed reset target/window exceeds a valid gesture "
                "sequence."
            )
        temporal_codes = self._build_reset_temporal_codes(
            input_codes,
            target_codes,
            target_times,
            codec_motion_inputs,
            reset_code_windows=reset_code_windows,
        )
        selected_loss_mask = self._target_only_loss_mask(
            token_loss_mask,
            target_times,
        )
        self._log_reset_boundaries(
            split,
            self._last_reset_future_boundaries,
        )
        # T2 is the full-condition teacher, matching the existing T1 run.
        return (
            temporal_codes,
            audio_codes,
            text_codes,
            selected_loss_mask,
        )

    def build_oracle_temporal_codes(
        self,
        expanded_codes,
        target_times,
        *,
        codec_motion_inputs=None,
    ):
        if codec_motion_inputs is None:
            raise RuntimeError(
                "T2 oracle evaluation requires aligned raw codec inputs."
            )
        target_batch = target_times.numel()
        expanded_motion = tuple(
            motion.expand(target_batch, -1, -1)
            for motion in codec_motion_inputs
        )
        return self._build_reset_temporal_codes(
            expanded_codes,
            expanded_codes,
            target_times,
            expanded_motion,
        )
