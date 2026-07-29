"""Trainers for parameter-free masked-frame future-gesture teachers."""

import copy
import math

import torch
from loguru import logger

from miburi.models import (
    GTemporalDepthModel3FutureGesture,
    GTemporalDepthModel3FutureGestureFullCondition,
    loaders,
)
from miburi.models.gesture_lm_future_gesture import (
    build_masked_future_gesture_inputs,
    truncate_condition_codes_after_targets,
)

from .uflgtdm3_trainer import UpperFaceLowerGTDM3Trainer


class _UpperFaceLowerGTDM3FutureGestureTrainer(
    UpperFaceLowerGTDM3Trainer
):
    """One masked target per sample with a fixed physical future horizon."""

    model_class = GTemporalDepthModel3FutureGesture
    expose_future_audio_text = False

    def __init__(self, args):
        unsupported_auxiliary = {
            "contrastive_loss_weight": args.contrastive_loss_weight,
            "genrecon_loss_weight": args.genrecon_loss_weight,
            "gan_loss_weight": args.gan_loss_weight,
        }
        enabled_auxiliary = {
            name: value
            for name, value in unsupported_auxiliary.items()
            if value > 0
        }
        if enabled_auxiliary:
            raise ValueError(
                "Masked-frame future teachers require sequence-wide "
                "contrastive/reconstruction/GAN losses to be disabled; "
                "those losses would train on answer-visible positions. "
                f"Enabled={enabled_auxiliary}."
            )
        if args.vad_guidance and args.vad_use_face_logits:
            raise ValueError(
                "Masked-frame future teachers require "
                "vad_use_face_logits=False because teacher-forced current-"
                "frame depth logits would leak gesture information into the "
                "VAD auxiliary path."
            )

        super().__init__(args)

        self.future_horizon_motion_frames = int(
            args.future_gesture_horizon_frames
        )
        if self.future_horizon_motion_frames <= 0:
            raise ValueError(
                "future_gesture_horizon_frames must be positive."
            )
        codec_frames_per_token = int(
            self.upper_gesture_codec.frame_size
        )
        self.future_horizon_tokens = math.ceil(
            self.future_horizon_motion_frames / codec_frames_per_token
        )
        self.future_horizon_seconds = (
            self.future_horizon_motion_frames / args.motion_fps
        )
        self.past_context_tokens = int(self.model.context)
        if self.past_context_tokens <= 0:
            raise ValueError(
                "Masked-frame future teachers require a positive original "
                "temporal context."
            )
        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] Masked-frame future "
            f"teacher horizon: {self.future_horizon_motion_frames} motion "
            f"frames = {self.future_horizon_seconds:.3f}s = "
            f"{self.future_horizon_tokens} gesture tokens; retained past="
            f"{self.past_context_tokens} gesture tokens; "
            f"audio/text mode={self.model.temporal_condition_mode}"
        )

    def get_model(self, args):
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            loaders.DEFAULT_REPO
        )
        lm = checkpoint_info.get_moshi()
        text_procemb = copy.deepcopy(lm.text_emb.weight.data)
        audio_procemb = [
            copy.deepcopy(audio_embedding.weight.data)
            for audio_embedding in lm.emb[:8]
        ]
        del lm
        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] "
            "text/audio embedding processors loaded for masked-frame "
            f"teacher ({self.model_class.__name__})"
        )

        gesture_lm_kwargs = loaders.get_gesturelm_kwargs()
        gesture_codec_layers = (
            copy.deepcopy(self.upper_gesture_codec.quantizer.vq.layers)
            + copy.deepcopy(self.lower_gesture_codec.quantizer.vq.layers)
            + copy.deepcopy(self.face_gesture_codec.quantizer.vq.layers)
        )
        if len(gesture_codec_layers) != gesture_lm_kwargs["n_q"]:
            raise ValueError(
                f"Expected {gesture_lm_kwargs['n_q']} gesture codec layers, "
                f"got {len(gesture_codec_layers)}."
            )
        for codec_layer in gesture_codec_layers:
            codec_layer.requires_grad_(False)
        gesture_codec_layers.eval()

        return self.model_class(
            num_heads=args.gestureformer_heads,
            num_layers=args.gestureformer_layers,
            depformer_heads=args.gestureformer_depformer_heads,
            depformer_layers=args.gestureformer_depformer_layers,
            query2mem_scale=self.codec_difference,
            num_temp_classifiers=args.num_temp_classifiers,
            dtype=(
                torch.float32
                if args.param_dtype == "float32"
                else torch.bfloat16
            ),
            text_procemb=text_procemb,
            audio_procemb=audio_procemb,
            gesture_codec_layers=gesture_codec_layers,
            vad_guidance=args.vad_guidance,
            vad_use_face_logits=args.vad_use_face_logits,
            body_parts=3,
            bp_dist=None,
            textaudio_emb_freeze=args.textaudio_emb_freeze,
            **gesture_lm_kwargs,
        )

    def _select_target_times(
        self,
        token_loss_mask,
        split,
        epoch,
        iteration,
    ):
        batch = token_loss_mask.shape[0]
        valid_lengths = token_loss_mask[:, 0].sum(dim=-1).long()
        max_target_times = (
            valid_lengths - self.future_horizon_tokens - 1
        )
        if (max_target_times < 0).any():
            raise ValueError(
                "Sequence is too short for the configured future horizon: "
                f"valid lengths={valid_lengths.tolist()}, "
                f"horizon={self.future_horizon_tokens} gesture tokens."
            )

        if split == "train":
            # A private integer schedule avoids dependence on parameter
            # initialization RNG consumption, so both teacher variants see
            # exactly the same targets when run with the same seed/data.
            offsets = (
                torch.arange(batch, device=token_loss_mask.device)
                + iteration * batch
                + epoch * 1_000_003
                + self.global_rank * 97_409
                + int(self.args.random_seed)
            )
            hashed = (offsets * 48_271 + 1) % 2_147_483_647
            target_times = hashed % (max_target_times + 1)
        else:
            # Deterministic coverage makes validation directly comparable
            # across the causal-condition and full-condition teachers. Hash
            # sample order so a short validation set spans the full timeline
            # instead of covering only its earliest target positions.
            offsets = (
                torch.arange(batch, device=token_loss_mask.device)
                + iteration * batch
                + int(self.args.random_seed)
            )
            hashed = (offsets * 48_271 + 1) % 2_147_483_647
            target_times = hashed % (max_target_times + 1)
        return target_times

    @staticmethod
    def _target_only_loss_mask(token_loss_mask, target_times):
        selected = torch.zeros_like(token_loss_mask)
        batch_indices = torch.arange(
            token_loss_mask.shape[0],
            device=token_loss_mask.device,
        )
        selected[
            batch_indices,
            :,
            target_times,
        ] = token_loss_mask[
            batch_indices,
            :,
            target_times,
        ]
        return selected

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
    ):
        target_times = self._select_target_times(
            token_loss_mask,
            split,
            epoch,
            iteration,
        )
        temporal_input_codes = build_masked_future_gesture_inputs(
            input_codes,
            target_codes,
            target_times,
            horizon_tokens=self.future_horizon_tokens,
            past_context_tokens=self.past_context_tokens,
            mask_token_id=self.modelout_ignore_index,
        )
        selected_loss_mask = self._target_only_loss_mask(
            token_loss_mask,
            target_times,
        )

        if not self.expose_future_audio_text:
            audio_codes = truncate_condition_codes_after_targets(
                audio_codes,
                target_times,
                condition_steps_per_gesture=self.codec_difference,
                null_token_id=self.audio_codec_nulltoken,
            )
            text_codes = truncate_condition_codes_after_targets(
                text_codes,
                target_times,
                condition_steps_per_gesture=self.codec_difference,
                null_token_id=self.text_codec_nulltoken,
            )

        return (
            temporal_input_codes,
            audio_codes,
            text_codes,
            selected_loss_mask,
        )

    def get_temporal_auxiliary_loss_mask(self, token_loss_mask):
        # q0 is valid for every fully supervised target. This prevents VAD
        # gradients from answer-visible, non-target positions.
        return token_loss_mask[:, 0].bool()

    def test(self, *args, **kwargs):
        raise RuntimeError(
            "Masked-frame future teachers support teacher-forced "
            "train/validation comparison only. Standalone generation has no "
            "ground-truth future gestures and is intentionally disabled."
        )


class UpperFaceLowerGTDM3FutureGestureTrainer(
    _UpperFaceLowerGTDM3FutureGestureTrainer
):
    """Future gesture from 400 ms onward; audio/text stop at the target."""

    model_class = GTemporalDepthModel3FutureGesture
    expose_future_audio_text = False


class UpperFaceLowerGTDM3FutureGestureFullConditionTrainer(
    _UpperFaceLowerGTDM3FutureGestureTrainer
):
    """Future gesture from 400 ms onward plus complete audio/text."""

    model_class = GTemporalDepthModel3FutureGestureFullCondition
    expose_future_audio_text = True
