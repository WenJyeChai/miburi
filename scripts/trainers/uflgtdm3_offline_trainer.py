"""Trainer for original MIBURI with offline temporal audio/text context.

This intentionally inherits the released objective and training loop. It
does not enable C2F, self-forcing, soft recovery, or any additional loss.
"""

import copy

import torch
import torch.nn.functional as F
from loguru import logger

from miburi.models import (
    GTemporalDepthModel3Offline,
    GestureLMOfflineGen,
    loaders,
)

from .uflgtdm3_trainer import UpperFaceLowerGTDM3Trainer
from .utils import tools as other_tools


class UpperFaceLowerGTDM3OfflineTrainer(UpperFaceLowerGTDM3Trainer):
    """Original one-q0/19-depth architecture with full temporal memory."""

    def __init__(self, args):
        super().__init__(args)
        metric_names = list(self.tracker.metric_names) + [
            "temporal_q0_ce",
            "temporal_q0_acc",
        ]
        metric_directions = [
            self.tracker.is_higher_better[name]
            for name in self.tracker.metric_names
        ] + [False, True]
        self.tracker = other_tools.EpochTracker(
            metric_names,
            metric_directions,
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
            "text/audio embedding processors loaded for offline MIBURI"
        )

        gesture_lm_kwargs = loaders.get_gesturelm_kwargs()
        upper_layers = copy.deepcopy(
            self.upper_gesture_codec.quantizer.vq.layers
        )
        lower_layers = copy.deepcopy(
            self.lower_gesture_codec.quantizer.vq.layers
        )
        face_layers = copy.deepcopy(
            self.face_gesture_codec.quantizer.vq.layers
        )
        gesture_codec_layers = upper_layers + lower_layers + face_layers
        if len(gesture_codec_layers) != gesture_lm_kwargs["n_q"]:
            raise ValueError(
                f"Expected {gesture_lm_kwargs['n_q']} gesture codec layers, "
                f"got {len(gesture_codec_layers)}."
            )
        for codec_layer in gesture_codec_layers:
            codec_layer.requires_grad_(False)
        gesture_codec_layers.eval()

        return GTemporalDepthModel3Offline(
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

    def get_generation_class(self):
        return GestureLMOfflineGen

    def prepare_generation_model(self, generator, full_condition):
        if not isinstance(generator, GestureLMOfflineGen):
            raise TypeError(
                f"Expected GestureLMOfflineGen, got {type(generator)!r}."
            )
        generator.set_full_condition(full_condition)
        logger.info(
            "Installed full-sequence audio/text memory for offline temporal "
            f"generation ({full_condition.shape[-1]} condition tokens)."
        )

    def _record_temporal_q0(
        self,
        split,
        logits,
        gesture_tokens,
        pad_loss_mask,
    ):
        q0_logits = logits[:, 0]
        q0_targets = gesture_tokens[:, 0]
        q0_valid = pad_loss_mask[:, 0].bool()
        if not q0_valid.any():
            return

        token_losses = F.cross_entropy(
            q0_logits.reshape(-1, q0_logits.shape[-1]),
            q0_targets.reshape(-1),
            reduction="none",
        ).reshape_as(q0_targets)
        q0_ce = token_losses[q0_valid].mean()
        # PAD is a training-only class and must not receive credit as a
        # generated temporal token.
        q0_prediction = q0_logits[..., : self.modelout_ignore_index].argmax(
            dim=-1
        )
        q0_accuracy = (
            q0_prediction[q0_valid] == q0_targets[q0_valid]
        ).float().mean()
        self.tracker.update_meter(
            "temporal_q0_ce",
            split,
            q0_ce.item(),
        )
        self.tracker.update_meter(
            "temporal_q0_acc",
            split,
            q0_accuracy.item(),
        )

    def get_additional_training_loss(
        self,
        logits,
        gesture_tokens,
        pad_loss_mask,
        epoch,
        iteration,
    ):
        self._record_temporal_q0(
            "train",
            logits,
            gesture_tokens,
            pad_loss_mask,
        )
        # Diagnostics only: preserve the original MIBURI objective exactly.
        return None

    def record_validation_diagnostics(
        self,
        logits,
        gesture_tokens,
        pad_loss_mask,
    ):
        self._record_temporal_q0(
            "val",
            logits,
            gesture_tokens,
            pad_loss_mask,
        )
