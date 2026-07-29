"""Trainers for leak-free privileged future-gesture MIBURI teachers."""

import copy

import torch
from loguru import logger

from miburi.models import (
    GTemporalDepthModel3FutureGesture,
    GTemporalDepthModel3FutureGestureFullCondition,
    loaders,
)

from .uflgtdm3_offline_trainer import UpperFaceLowerGTDM3OfflineTrainer


class _UpperFaceLowerGTDM3FutureGestureTrainer(
    UpperFaceLowerGTDM3OfflineTrainer
):
    model_class = GTemporalDepthModel3FutureGesture

    def get_model_forward_kwargs(
        self,
        epoch,
        iteration,
        target_codes=None,
    ):
        # Training may mask lower/face inputs for robustness. The privileged
        # teacher suffix must still encode the true, unmasked future target.
        return {"temporal_target_codes": target_codes}

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
            "text/audio embedding processors loaded for future-gesture "
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
            future_gesture_layers=args.future_gesture_layers,
            future_gesture_heads=args.future_gesture_heads,
            future_gesture_context=args.future_gesture_context,
            future_gesture_gate_init=args.future_gesture_gate_init,
            **gesture_lm_kwargs,
        )

    def test(self, *args, **kwargs):
        raise RuntimeError(
            "Future-gesture teachers support teacher-forced train/validation "
            "comparison only. Standalone sampling would require unavailable "
            "ground-truth future gestures and is intentionally disabled."
        )


class UpperFaceLowerGTDM3FutureGestureTrainer(
    _UpperFaceLowerGTDM3FutureGestureTrainer
):
    """Future gesture with causal paired audio/text."""

    model_class = GTemporalDepthModel3FutureGesture


class UpperFaceLowerGTDM3FutureGestureFullConditionTrainer(
    _UpperFaceLowerGTDM3FutureGestureTrainer
):
    """Future gesture with complete paired audio/text."""

    model_class = GTemporalDepthModel3FutureGestureFullCondition
