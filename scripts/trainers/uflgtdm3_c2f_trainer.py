"""Opt-in trainer for the three-q0 global-C2F gesture LM.

The released ``UpperFaceLowerGTDM3Trainer`` remains the default. This subclass
only replaces model construction, scheduled temporal/kinematic self-forcing,
the soft-recovery auxiliary loss, and the matching streaming generator.
"""

import copy

import torch
from loguru import logger

from miburi.models import GTemporalDepthModel3C2F, GestureLMC2FGen, loaders

from .uflgtdm3_trainer import UpperFaceLowerGTDM3Trainer
from .utils import tools as other_tools


class UpperFaceLowerGTDM3C2FTrainer(UpperFaceLowerGTDM3Trainer):
    def __init__(self, args):
        super().__init__(args)
        metric_names = list(self.tracker.metric_names) + [
            "c2f_self_forcing_prob",
            "c2f_self_forcing_active",
            "c2f_temporal_q0_acc",
            "c2f_temporal_upper_q0_acc",
            "c2f_temporal_lower_q0_acc",
            "c2f_temporal_face_q0_acc",
            "soft_recovery_loss",
            "soft_recovery_weighted",
            "gumbel_tau",
            "contrastive_temperature",
            "contrastive_weight",
        ]
        metric_directions = [
            self.tracker.is_higher_better[name]
            for name in self.tracker.metric_names
        ] + [
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
        # The inherited tracker has not seen any batches yet, so replacing it
        # here preserves every legacy metric while exposing C2F diagnostics to
        # TensorBoard and the rank-zero W&B logger.
        self.tracker = other_tools.EpochTracker(
            metric_names,
            metric_directions,
        )

    def get_model(self, args):
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(loaders.DEFAULT_REPO)
        lm = checkpoint_info.get_moshi()
        text_procemb = copy.deepcopy(lm.text_emb.weight.data)
        audio_procemb = [
            copy.deepcopy(audio_embedding.weight.data)
            for audio_embedding in lm.emb[:8]
        ]
        del lm
        logger.info(
            f"[GPU{self.global_rank}:{self.local_rank}] "
            "text/audio embedding processors loaded for C2F model"
        )

        gesture_lm_kwargs = loaders.get_gesturelm_kwargs()
        upper_layers = copy.deepcopy(self.upper_gesture_codec.quantizer.vq.layers)
        lower_layers = copy.deepcopy(self.lower_gesture_codec.quantizer.vq.layers)
        face_layers = copy.deepcopy(self.face_gesture_codec.quantizer.vq.layers)
        gesture_codec_layers = upper_layers + lower_layers + face_layers
        if len(gesture_codec_layers) != gesture_lm_kwargs["n_q"]:
            raise ValueError(
                f"Expected {gesture_lm_kwargs['n_q']} gesture codec layers, "
                f"got {len(gesture_codec_layers)}."
            )
        for codec_layer in gesture_codec_layers:
            codec_layer.requires_grad_(False)
        gesture_codec_layers.eval()

        return GTemporalDepthModel3C2F(
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

    def _self_forcing_probability(self, epoch, iteration):
        if not self.args.c2f_self_forcing:
            return 0.0
        fractional_epoch = epoch + iteration / max(1, self.train_length)
        start_epoch = getattr(
            self.args, "c2f_self_forcing_start_epoch", -1.0
        )
        if start_epoch >= 0:
            ramp_epochs = getattr(
                self.args, "c2f_self_forcing_ramp_epochs", -1.0
            )
            maximum = self.args.c2f_self_forcing_max_prob
            if fractional_epoch < start_epoch:
                return 0.0
            if ramp_epochs <= 0:
                return maximum
            ramp_progress = (
                fractional_epoch - start_epoch
            ) / ramp_epochs
            return maximum * min(1.0, max(0.0, ramp_progress))

        progress = fractional_epoch / max(1, self.args.epochs)
        warmup = self.args.c2f_self_forcing_warmup_ratio
        ramp = self.args.c2f_self_forcing_ramp_ratio
        maximum = self.args.c2f_self_forcing_max_prob
        if progress < warmup:
            return 0.0
        if ramp <= 0 or progress >= warmup + ramp:
            return maximum
        return maximum * (progress - warmup) / ramp

    def get_model_forward_kwargs(self, epoch, iteration, target_codes=None):
        probability = self._self_forcing_probability(epoch, iteration)
        seed = int(self.args.random_seed)
        # A deterministic per-step draw keeps all DDP ranks on the same branch.
        draw = (
            (seed * 1_000_003 + epoch * 97_409 + iteration * 65_537)
            % 1_000_000
        ) / 1_000_000.0
        use_self_forcing = draw < probability
        self._c2f_self_forcing_probability = probability
        self._c2f_used_self_forcing = use_self_forcing
        self.tracker.update_meter(
            "c2f_self_forcing_prob", "train", probability
        )
        self.tracker.update_meter(
            "c2f_self_forcing_active", "train", float(use_self_forcing)
        )
        if (
            self.args.contrastive_loss_weight > 0
            and epoch > self.args.pretrain_warmup_epochs
        ):
            tau_start = getattr(self.args, "gumbel_tau", 1.0)
            tau_minimum = getattr(self.args, "gumbel_tau_min", 0.4)
            anneal_epochs = getattr(
                self.args, "gumbel_tau_anneal_epochs", 5
            )
            anneal_progress = min(
                1.0,
                max(
                    0.0,
                    (epoch - self.args.pretrain_warmup_epochs)
                    / max(1, anneal_epochs),
                ),
            )
            gumbel_tau = tau_start - (
                tau_start - tau_minimum
            ) * anneal_progress
            contrastive_weight = (
                (iteration / max(1, self.train_length))
                * self.args.contrastive_loss_weight
                if epoch - self.args.pretrain_warmup_epochs == 1
                else self.args.contrastive_loss_weight
            )
            self.tracker.update_meter(
                "gumbel_tau", "train", gumbel_tau
            )
            self.tracker.update_meter(
                "contrastive_temperature",
                "train",
                float(self.contrastive_loss_func.temperature),
            )
            self.tracker.update_meter(
                "contrastive_weight", "train", contrastive_weight
            )
        if iteration % self.args.log_period == 0 and self.global_rank == 0:
            logger.info(
                f"C2F self-forcing p={probability:.3f}, "
                f"active={use_self_forcing}"
            )
        return {
            "self_force_kinematic": use_self_forcing,
            "kinematic_target_codes": target_codes,
        }

    def get_additional_training_loss(
        self,
        logits,
        gesture_tokens,
        pad_loss_mask,
        epoch,
        iteration,
    ):
        del pad_loss_mask, epoch
        model = self.model.module if self.args.ddp else self.model
        if getattr(self, "_c2f_used_self_forcing", False):
            rollout = model.last_temporal_rollout_codes
            if rollout is not None:
                q0_target = gesture_tokens[:, model.coarse_slots]
                rollout_q0 = rollout[:, model.coarse_slots]
                valid = q0_target != model.pad_token_id
                correct = rollout_q0 == q0_target
                if valid.any():
                    self.tracker.update_meter(
                        "c2f_temporal_q0_acc",
                        "train",
                        correct[valid].float().mean().item(),
                    )
                part_names = ("upper", "lower", "face")
                for part_index, part_name in enumerate(part_names):
                    part_valid = valid[:, part_index]
                    if part_valid.any():
                        part_accuracy = correct[
                            :, part_index
                        ][part_valid].float().mean()
                        self.tracker.update_meter(
                            f"c2f_temporal_{part_name}_q0_acc",
                            "train",
                            part_accuracy.item(),
                        )

        weight = self.args.soft_recovery_weight
        if weight <= 0 or not getattr(self, "_c2f_used_self_forcing", False):
            return None

        prefix_codes = model.last_kinematic_input_codes
        if prefix_codes is None:
            return None
        recovery = model.soft_recovery_loss(
            logits,
            gesture_tokens,
            prefix_codes,
            topk=self.args.soft_recovery_topk,
            sigma_scale=self.args.soft_recovery_sigma_scale,
            only_wrong_prefix=self.args.soft_recovery_only_wrong_prefix,
        )
        weighted = recovery * weight
        self.tracker.update_meter(
            "soft_recovery_loss", "train", recovery.item()
        )
        self.tracker.update_meter(
            "soft_recovery_weighted", "train", weighted.item()
        )
        if iteration % self.args.log_period == 0 and self.global_rank == 0:
            logger.info(
                f"soft recovery={recovery.item():.4f}, "
                f"weighted={weighted.item():.4f}"
            )
        return weighted

    def get_generation_class(self):
        return GestureLMC2FGen
