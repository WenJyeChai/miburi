"""Trainers for parameter-free masked-frame future-gesture teachers."""

import copy
import gc
import math
import os
import shutil
import tempfile
import time

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from miburi.models import (
    GTemporalDepthModel3FutureGesture,
    GTemporalDepthModel3FutureGestureFullCondition,
    GestureLMGen,
    loaders,
)
from miburi.models.gesture_lm_future_gesture import (
    build_masked_future_gesture_inputs,
    truncate_condition_codes_after_targets,
)
from miburi.utils.sampling import sample_token

from .uflgtdm3_trainer import UpperFaceLowerGTDM3Trainer
from .dataloaders.utils.visualize import (
    mux_audio_into_video,
    render_smplx_debug_video,
    stitch_videos_hstack,
)
from .utils import rotation_conversions as rc
from .utils import tools as other_tools


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
        required_future_tokens = self.minimum_future_anchor_tokens()
        max_target_times = (
            valid_lengths - required_future_tokens - 1
        )
        if (max_target_times < 0).any():
            raise ValueError(
                "Sequence is too short for the configured future horizon: "
                f"valid lengths={valid_lengths.tolist()}, "
                f"required future={required_future_tokens} gesture tokens."
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

    def minimum_future_anchor_tokens(self):
        """Tokens required after a target for one usable future anchor."""

        return self.future_horizon_tokens

    def visible_future_offset_seconds(self):
        return (
            self.minimum_future_anchor_tokens()
            * self.args.frame_chunk_size
            / self.args.motion_fps
        )

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
        **batch_context,
    ):
        del batch_context
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

    def _oracle_generation_settings(self):
        mode = getattr(
            self.args,
            "eval_generation_mode",
            "production",
        )
        if mode == "production":
            settings = {
                "use_sampling": True,
                "temp_gtemporal": 0.9,
                "temp_gdepth": 0.9,
                "top_p_gtemporal": 0.8,
                "top_p_gdepth": 0.95,
                "check": True,
            }
            cfg_coef = self.args.cfg_scale
        elif mode == "greedy_cfg1":
            settings = {
                "use_sampling": False,
                "temp_gtemporal": 0.9,
                "temp_gdepth": 0.9,
                "top_p_gtemporal": 0.8,
                "top_p_gdepth": 0.95,
                "check": True,
            }
            cfg_coef = 1.0
        else:
            raise ValueError(
                f"Unknown eval_generation_mode={mode!r}."
            )
        return mode, settings, float(cfg_coef)

    def _oracle_predict_target_batch(
        self,
        gesture_codes,
        audio_codes,
        text_codes,
        speaker,
        target_times,
        rollout,
        depth_cross_attention_mask,
        codec_motion_inputs=None,
    ):
        """Predict complete frames for independently masked target times.

        Temporal q0 is sampled from the masked future-aware query. The
        remaining 19 tokens are then rolled out autoregressively from sampled
        prefixes, matching production kinematic inference rather than using
        the teacher-forced depth logits from ``model.forward``.
        """

        if gesture_codes.shape[0] != 1:
            raise ValueError(
                "Oracle target batching currently expects one source clip."
            )
        target_batch = target_times.numel()
        expanded_codes = gesture_codes.expand(
            target_batch,
            -1,
            -1,
        )
        temporal_codes = self.build_oracle_temporal_codes(
            expanded_codes,
            target_times,
            codec_motion_inputs=codec_motion_inputs,
        )
        temporal_audio = audio_codes.expand(
            target_batch,
            -1,
            -1,
        )
        temporal_text = text_codes.expand(
            target_batch,
            -1,
            -1,
        )
        if not self.expose_future_audio_text:
            temporal_audio = truncate_condition_codes_after_targets(
                temporal_audio,
                target_times,
                condition_steps_per_gesture=self.codec_difference,
                null_token_id=self.audio_codec_nulltoken,
            )
            temporal_text = truncate_condition_codes_after_targets(
                temporal_text,
                target_times,
                condition_steps_per_gesture=self.codec_difference,
                null_token_id=self.text_codec_nulltoken,
            )

        speaker_batch = speaker.reshape(1).expand(target_batch)
        cfg_enabled = rollout.cfg_coef != 1.0
        if cfg_enabled:
            model_temporal_codes = torch.cat(
                [temporal_codes, temporal_codes],
                dim=0,
            )
            model_audio = torch.cat(
                [
                    temporal_audio,
                    torch.full_like(
                        temporal_audio,
                        self.audio_codec_nulltoken,
                    ),
                ],
                dim=0,
            )
            model_text = torch.cat(
                [
                    temporal_text,
                    torch.full_like(
                        temporal_text,
                        self.text_codec_nulltoken,
                    ),
                ],
                dim=0,
            )
            model_speaker = torch.cat(
                [speaker_batch, speaker_batch],
                dim=0,
            )
            model_targets = torch.cat(
                [target_times, target_times],
                dim=0,
            )
        else:
            model_temporal_codes = temporal_codes
            model_audio = temporal_audio
            model_text = temporal_text
            model_speaker = speaker_batch
            model_targets = target_times

        temporal_out, q0_logits = (
            self.model.forward_oracle_temporal_targets(
                model_temporal_codes,
                model_audio,
                model_text,
                model_speaker,
                model_targets,
            )
        )
        if cfg_enabled:
            conditioned_logits, null_logits = q0_logits.chunk(2, dim=0)
            guided_q0_logits = null_logits + rollout.cfg_coef * (
                conditioned_logits - null_logits
            )
        else:
            guided_q0_logits = q0_logits
        guided_q0_logits[..., self.modelout_ignore_index] = float("-inf")
        q0_tokens = sample_token(
            guided_q0_logits.float(),
            use_sampling=rollout.use_sampling,
            temp=rollout.temp_temporal,
            top_k=rollout.top_k_temp,
            top_p=rollout.top_p_temp,
        )[:, 0, 0]

        condition_width = self.codec_difference
        current_audio = torch.stack(
            [
                audio_codes[
                    0,
                    :,
                    int(target_time) * condition_width:
                    (int(target_time) + 1) * condition_width,
                ]
                for target_time in target_times.tolist()
            ],
            dim=0,
        )
        current_text = torch.stack(
            [
                text_codes[
                    0,
                    :,
                    int(target_time) * condition_width:
                    (int(target_time) + 1) * condition_width,
                ]
                for target_time in target_times.tolist()
            ],
            dim=0,
        )
        if cfg_enabled:
            current_audio = torch.cat(
                [
                    current_audio,
                    torch.full_like(
                        current_audio,
                        self.audio_codec_nulltoken,
                    ),
                ],
                dim=0,
            )
            current_text = torch.cat(
                [
                    current_text,
                    torch.full_like(
                        current_text,
                        self.text_codec_nulltoken,
                    ),
                ],
                dim=0,
            )
            depth_speaker = torch.cat(
                [speaker_batch, speaker_batch],
                dim=0,
            )[:, None]
        else:
            depth_speaker = speaker_batch[:, None]
        depth_audio, depth_text = rollout.process_conditions(
            current_audio,
            current_text,
        )

        depth_mask = depth_cross_attention_mask[
            :,
            1:,
            :,
        ].expand(target_batch, -1, -1)
        cfg_stop_mask = (
            depth_mask[0, :, 0].cpu().tolist()
            if cfg_enabled
            else None
        )
        depth_tokens = rollout.depformer_step(
            q0_tokens,
            temporal_out,
            depth_audio,
            depth_text,
            depth_speaker,
            depth_mask,
            cfg_stop_mask,
            rollout.bp_dist,
        )
        predicted = torch.cat(
            [q0_tokens[:, None], depth_tokens],
            dim=1,
        )
        if predicted.shape != (
            target_batch,
            self.model.n_q,
        ):
            raise RuntimeError(
                "Unexpected oracle token shape "
                f"{tuple(predicted.shape)}."
            )
        return predicted

    def build_oracle_temporal_codes(
        self,
        expanded_codes,
        target_times,
        *,
        codec_motion_inputs=None,
    ):
        """Build the teacher view used by oracle target evaluation."""

        del codec_motion_inputs
        return build_masked_future_gesture_inputs(
            expanded_codes,
            expanded_codes,
            target_times,
            horizon_tokens=self.future_horizon_tokens,
            past_context_tokens=self.past_context_tokens,
            mask_token_id=self.modelout_ignore_index,
        )

    @torch.no_grad()
    def test(
        self,
        epoch,
        visualize=False,
        max_batches=None,
        save=False,
    ):
        """Ground-truth-future oracle infilling evaluation.

        This deliberately does not call the released streaming gesture
        generator. Every target time receives its own masked full-sequence
        temporal pass, followed by a production-style autoregressive
        kinematic rollout. The final lookahead horizon is excluded from both
        predicted and target metrics because no valid future anchor exists.
        """

        if self.test_loader.batch_size != 1:
            raise ValueError(
                "Future-gesture oracle evaluation requires --batch_size 1."
            )
        target_batch_size = int(
            self.args.future_oracle_target_batch_size
        )
        if target_batch_size <= 0:
            raise ValueError(
                "future_oracle_target_batch_size must be positive."
            )

        mode, generation_settings, cfg_coef = (
            self._oracle_generation_settings()
        )
        self.model.eval()
        self.upper_gesture_codec.eval()
        self.lower_gesture_codec.eval()
        self.face_gesture_codec.eval()
        self.gesture_metrics.reset()
        gc.collect()

        logger.warning(
            "ORACLE FUTURE-GESTURE EVALUATION: predictions receive "
            "ground-truth gesture tokens beginning "
            f"{self.visible_future_offset_seconds():.3f}s after each target. "
            "Reported oracle FGD is not standalone-generation FGD."
        )
        logger.info(
            f"Oracle mode={mode}; target_batch_size={target_batch_size}; "
            f"CFG={cfg_coef}; audio/text mode="
            f"{self.model.temporal_condition_mode}; final "
            f"{self.minimum_future_anchor_tokens()} gesture tokens are "
            "excluded."
        )

        rollout = GestureLMGen(
            self.model,
            cfg_coef=cfg_coef,
            **generation_settings,
        )
        results_save_path = os.path.join(
            self.checkpoint_path,
            f"oracle_{epoch}",
        )
        if visualize or save:
            os.makedirs(results_save_path, exist_ok=True)
        start_time = time.time()
        total_motion_frames = 0
        total_q0_correct = 0
        total_all_correct = 0
        total_gesture_tokens = 0
        total_code_tokens = 0

        upper_joint_mask = (
            self.test_data.upper_mask_for_flattened
            + self.test_data.hands_mask_for_flattened
        )
        lower_joint_mask = self.test_data.lower_mask_for_flattened
        face_joint_mask = self.test_data.face_mask_for_flattened

        for batch_index, dict_data in enumerate(self.test_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            file_name = dict_data["file_id"]
            sample_name = dict_data["filechunk_id"][0]
            raw_audio_batch = dict_data.get("raw_audio")
            raw_audio = (
                raw_audio_batch[0]
                if visualize and raw_audio_batch is not None
                else None
            )
            tar_pose_upper = dict_data["motion_upper"].to(self.local_rank)
            tar_pose_hands = dict_data["motion_hands"].to(self.local_rank)
            tar_pose_face = dict_data["motion_face"].to(self.local_rank)
            tar_pose_lower = dict_data["motion_lower"].to(self.local_rank)
            tar_trans = dict_data["transl"].to(self.local_rank)
            tar_contact = dict_data["contact"].to(self.local_rank)
            tar_exps = dict_data["expressions"].to(self.local_rank)
            tar_beta = dict_data["beta"].to(self.local_rank)
            audio_codes = dict_data["audio_tokens"].to(self.local_rank)
            text_codes = dict_data["text_tokens"].to(self.local_rank)
            speaker = dict_data["speaker_id"].to(self.local_rank)

            remainder = (
                tar_pose_upper.shape[1]
                % self.args.frame_chunk_size
            )
            if remainder:
                keep_frames = tar_pose_upper.shape[1] - remainder
                tar_pose_upper = tar_pose_upper[:, :keep_frames]
                tar_pose_hands = tar_pose_hands[:, :keep_frames]
                tar_pose_face = tar_pose_face[:, :keep_frames]
                tar_pose_lower = tar_pose_lower[:, :keep_frames]
                tar_trans = tar_trans[:, :keep_frames]
                tar_contact = tar_contact[:, :keep_frames]
                tar_exps = tar_exps[:, :keep_frames]
                tar_beta = tar_beta[:, :keep_frames]

            tar_trans[:, :, 0] -= tar_trans[:, 0:1, 0]
            tar_trans[:, :, 2] -= tar_trans[:, 0:1, 2]
            tar_trans_vel = other_tools.estimate_linear_velocity(
                tar_trans,
                dt=1 / self.args.motion_fps,
            )

            batch_size, motion_frames, upper_dim = tar_pose_upper.shape
            if batch_size != 1:
                raise RuntimeError(
                    "Oracle evaluation received a non-unit batch."
                )
            upper_joints = upper_dim // 3
            hand_joints = tar_pose_hands.shape[-1] // 3
            lower_joints = tar_pose_lower.shape[-1] // 3
            face_joints = tar_pose_face.shape[-1] // 3
            upper_hand_joints = upper_joints + hand_joints

            tar_pose_upperhands_aa = torch.cat(
                [tar_pose_upper, tar_pose_hands],
                dim=-1,
            )
            tar_pose_face_aa = tar_pose_face.clone()
            tar_pose_lower_aa = tar_pose_lower.clone()

            upper_6d = rc.matrix_to_rotation_6d(
                rc.axis_angle_to_matrix(
                    tar_pose_upper.reshape(
                        batch_size,
                        motion_frames,
                        upper_joints,
                        3,
                    )
                )
            ).reshape(batch_size, motion_frames, upper_joints * 6)
            hands_6d = rc.matrix_to_rotation_6d(
                rc.axis_angle_to_matrix(
                    tar_pose_hands.reshape(
                        batch_size,
                        motion_frames,
                        hand_joints,
                        3,
                    )
                )
            ).reshape(batch_size, motion_frames, hand_joints * 6)
            face_6d = rc.matrix_to_rotation_6d(
                rc.axis_angle_to_matrix(
                    tar_pose_face.reshape(
                        batch_size,
                        motion_frames,
                        face_joints,
                        3,
                    )
                )
            ).reshape(batch_size, motion_frames, face_joints * 6)
            lower_6d = rc.matrix_to_rotation_6d(
                rc.axis_angle_to_matrix(
                    tar_pose_lower.reshape(
                        batch_size,
                        motion_frames,
                        lower_joints,
                        3,
                    )
                )
            ).reshape(batch_size, motion_frames, lower_joints * 6)

            upper_codes = self.upper_gesture_codec.encode(
                torch.cat([upper_6d, hands_6d], dim=-1)
            )
            lower_codes = self.lower_gesture_codec.encode(
                torch.cat(
                    [lower_6d, tar_trans_vel, tar_contact],
                    dim=-1,
                )
            )
            face_codes = self.face_gesture_codec.encode(
                torch.cat([face_6d, tar_exps], dim=-1)
            )
            gesture_codes = torch.cat(
                [upper_codes, lower_codes, face_codes],
                dim=1,
            )

            condition_steps = min(
                audio_codes.shape[-1],
                text_codes.shape[-1],
            )
            condition_gesture_steps = (
                condition_steps // self.codec_difference
            )
            gesture_steps = min(
                gesture_codes.shape[-1],
                condition_gesture_steps,
            )
            required_future_tokens = self.minimum_future_anchor_tokens()
            if gesture_steps <= required_future_tokens:
                raise ValueError(
                    f"Clip {file_name[0]} has only {gesture_steps} aligned "
                    "gesture tokens, too short for required future "
                    f"{required_future_tokens}."
                )
            aligned_condition_steps = (
                gesture_steps * self.codec_difference
            )
            aligned_motion_frames = (
                gesture_steps * self.args.frame_chunk_size
            )
            gesture_codes = gesture_codes[:, :, :gesture_steps]
            audio_codes = audio_codes[:, :, :aligned_condition_steps]
            text_codes = text_codes[:, :, :aligned_condition_steps]

            valid_gesture_steps = (
                gesture_steps - required_future_tokens
            )
            valid_motion_frames = (
                valid_gesture_steps * self.args.frame_chunk_size
            )
            predicted_codes = torch.empty(
                (
                    1,
                    self.model.n_q,
                    valid_gesture_steps,
                ),
                device=self.local_rank,
                dtype=torch.long,
            )

            depth_cross_attention_mask = torch.zeros(
                1,
                self.model.n_q,
                1,
                device=self.local_rank,
                dtype=torch.bool,
            )
            if self.args.drop_lower_crossattn:
                lower_start = self.upper_gesture_codec.num_codebooks
                lower_end = (
                    lower_start
                    + self.lower_gesture_codec.num_codebooks
                )
                depth_cross_attention_mask[
                    :,
                    lower_start:lower_end,
                    :,
                ] = True

            progress = tqdm(
                range(0, valid_gesture_steps, target_batch_size),
                desc=f"Oracle infill {file_name[0]}",
            )
            for target_start in progress:
                target_end = min(
                    target_start + target_batch_size,
                    valid_gesture_steps,
                )
                target_times = torch.arange(
                    target_start,
                    target_end,
                    device=self.local_rank,
                    dtype=torch.long,
                )
                try:
                    target_predictions = (
                        self._oracle_predict_target_batch(
                            gesture_codes,
                            audio_codes,
                            text_codes,
                            speaker,
                            target_times,
                            rollout,
                            depth_cross_attention_mask,
                            codec_motion_inputs=(
                                torch.cat(
                                    [upper_6d, hands_6d],
                                    dim=-1,
                                )[:, :aligned_motion_frames],
                                torch.cat(
                                    [
                                        lower_6d,
                                        tar_trans_vel,
                                        tar_contact,
                                    ],
                                    dim=-1,
                                )[:, :aligned_motion_frames],
                                torch.cat(
                                    [face_6d, tar_exps],
                                    dim=-1,
                                )[:, :aligned_motion_frames],
                            ),
                        )
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError(
                        "Oracle target batch exhausted CUDA memory. Retry "
                        "with a smaller "
                        "--future_oracle_target_batch_size."
                    ) from exc
                predicted_codes[
                    0,
                    :,
                    target_start:target_end,
                ] = target_predictions.transpose(0, 1)

            target_codes = gesture_codes[
                :,
                :,
                :valid_gesture_steps,
            ]
            total_q0_correct += int(
                (
                    predicted_codes[:, 0]
                    == target_codes[:, 0]
                ).sum().item()
            )
            total_all_correct += int(
                (predicted_codes == target_codes).sum().item()
            )
            total_gesture_tokens += valid_gesture_steps
            total_code_tokens += (
                valid_gesture_steps * self.model.n_q
            )

            upper_count = self.upper_gesture_codec.num_codebooks
            lower_count = self.lower_gesture_codec.num_codebooks
            predicted_upper = predicted_codes[:, :upper_count]
            predicted_lower = predicted_codes[
                :,
                upper_count:upper_count + lower_count,
            ]
            predicted_face = predicted_codes[
                :,
                upper_count + lower_count:,
            ]
            rec_upperhands_6d = self.upper_gesture_codec.decode(
                predicted_upper
            )
            rec_lowertrans = self.lower_gesture_codec.decode(
                predicted_lower
            )
            rec_face = self.face_gesture_codec.decode(predicted_face)
            if not (
                rec_upperhands_6d.shape[1]
                == rec_lowertrans.shape[1]
                == rec_face.shape[1]
                == valid_motion_frames
            ):
                raise RuntimeError(
                    "Oracle codec decode length mismatch: "
                    f"upper={rec_upperhands_6d.shape[1]}, "
                    f"lower={rec_lowertrans.shape[1]}, "
                    f"face={rec_face.shape[1]}, expected="
                    f"{valid_motion_frames}."
                )

            rec_pose_upperhands = rc.matrix_to_axis_angle(
                rc.rotation_6d_to_matrix(
                    rec_upperhands_6d.reshape(
                        1,
                        valid_motion_frames,
                        upper_hand_joints,
                        6,
                    )
                )
            ).reshape(
                valid_motion_frames,
                upper_hand_joints * 3,
            )
            rec_pose_lower = rc.matrix_to_axis_angle(
                rc.rotation_6d_to_matrix(
                    rec_lowertrans[
                        :,
                        :,
                        :lower_joints * 6,
                    ].reshape(
                        1,
                        valid_motion_frames,
                        lower_joints,
                        6,
                    )
                )
            ).reshape(valid_motion_frames, lower_joints * 3)
            rec_pose_face = rc.matrix_to_axis_angle(
                rc.rotation_6d_to_matrix(
                    rec_face[:, :, :face_joints * 6].reshape(
                        1,
                        valid_motion_frames,
                        face_joints,
                        6,
                    )
                )
            ).reshape(valid_motion_frames, face_joints * 3)
            rec_exps = rec_face[
                0,
                :,
                face_joints * 6:,
            ]
            rec_trans_vel = rec_lowertrans[
                :,
                :,
                lower_joints * 6:lower_joints * 6 + 3,
            ]
            rec_trans, _ = other_tools.velocity2position_mixeddiff(
                rec_trans_vel,
                1 / self.args.motion_fps,
                init_pos=tar_trans[:, 0],
            )
            rec_trans = rec_trans[0]

            tar_pose_upperhands = tar_pose_upperhands_aa[
                0,
                :valid_motion_frames,
            ]
            tar_pose_lower_valid = tar_pose_lower_aa[
                0,
                :valid_motion_frames,
            ]
            tar_pose_face_valid = tar_pose_face_aa[
                0,
                :valid_motion_frames,
            ]
            tar_trans_valid = tar_trans[
                0,
                :valid_motion_frames,
            ]
            tar_exps_valid = tar_exps[
                0,
                :valid_motion_frames,
            ]

            tar_pose_upperhands = (
                self.test_data.inverse_selection_tensor(
                    tar_pose_upperhands,
                    upper_joint_mask,
                    valid_motion_frames,
                )
            )
            rec_pose_upperhands = (
                self.test_data.inverse_selection_tensor(
                    rec_pose_upperhands,
                    upper_joint_mask,
                    valid_motion_frames,
                )
            )
            tar_pose_face_valid = (
                self.test_data.inverse_selection_tensor(
                    tar_pose_face_valid,
                    face_joint_mask,
                    valid_motion_frames,
                )
            )
            rec_pose_face = self.test_data.inverse_selection_tensor(
                rec_pose_face,
                face_joint_mask,
                valid_motion_frames,
            )
            tar_pose_lower_valid = (
                self.test_data.inverse_selection_tensor(
                    tar_pose_lower_valid,
                    lower_joint_mask,
                    valid_motion_frames,
                )
            )
            rec_pose_lower = self.test_data.inverse_selection_tensor(
                rec_pose_lower,
                lower_joint_mask,
                valid_motion_frames,
            )
            rec_pose_full = (
                rec_pose_upperhands
                + rec_pose_face
                + rec_pose_lower
            )
            tar_pose_full = (
                tar_pose_upperhands
                + tar_pose_face_valid
                + tar_pose_lower_valid
            )
            beta = tar_beta[0, 0]
            self.gesture_metrics.update(
                {
                    "rec_pose": rec_pose_full,
                    "rec_exps": rec_exps,
                    "rec_trans": rec_trans,
                    "tar_pose": tar_pose_full,
                    "tar_exps": tar_exps_valid,
                    "tar_beta": beta,
                    "tar_trans": tar_trans_valid,
                    "file_id": file_name[0],
                }
            )

            if visualize or save:
                sample_save_path = os.path.join(
                    results_save_path,
                    sample_name,
                )
                os.makedirs(sample_save_path, exist_ok=True)
                rec_pose_np = rec_pose_full.detach().cpu().numpy().reshape(
                    valid_motion_frames,
                    len(self.test_data.smplx_joint_names),
                    3,
                )
                tar_pose_np = tar_pose_full.detach().cpu().numpy().reshape(
                    valid_motion_frames,
                    len(self.test_data.smplx_joint_names),
                    3,
                )
                rec_trans_np = rec_trans.detach().cpu().numpy()
                tar_trans_np = tar_trans_valid.detach().cpu().numpy()
                rec_exps_np = rec_exps.detach().cpu().numpy()
                tar_exps_np = tar_exps_valid.detach().cpu().numpy()
                beta_np = beta.detach().cpu().numpy()

                if save:
                    np.savez(
                        os.path.join(sample_save_path, "gt.npz"),
                        betas=beta_np,
                        poses=tar_pose_np,
                        expressions=tar_exps_np,
                        trans=tar_trans_np,
                        model="smplx",
                        gender="NEUTRAL_2020",
                        mocap_frame_rate=self.args.motion_fps,
                    )
                    np.savez(
                        os.path.join(sample_save_path, "pred.npz"),
                        betas=beta_np,
                        poses=rec_pose_np,
                        expressions=rec_exps_np,
                        trans=rec_trans_np,
                        model="smplx",
                        gender="NEUTRAL_2020",
                        mocap_frame_rate=self.args.motion_fps,
                    )
                    np.savez(
                        os.path.join(
                            sample_save_path,
                            "oracle_tokens.npz",
                        ),
                        predicted=predicted_codes[0].detach().cpu().numpy(),
                        target=target_codes[0].detach().cpu().numpy(),
                        future_horizon_tokens=self.future_horizon_tokens,
                        required_future_anchor_tokens=(
                            self.minimum_future_anchor_tokens()
                        ),
                        excluded_tail_motion_frames=(
                            self.minimum_future_anchor_tokens()
                            * self.args.frame_chunk_size
                        ),
                    )

                if visualize:
                    import soundfile as sf

                    logger.info(
                        f"Rendering oracle comparison for {sample_name}"
                    )
                    final_path = os.path.join(
                        sample_save_path,
                        "oracle_gt_pred_compared_audio.mp4",
                    )
                    tar_trans_viz = (
                        tar_trans_np - tar_trans_np[0:1]
                    )
                    rec_trans_viz = (
                        rec_trans_np - rec_trans_np[0:1]
                    )
                    with tempfile.TemporaryDirectory(
                        prefix="oracle_test_sbs_"
                    ) as temp_dir:
                        gt_path = os.path.join(temp_dir, "gt.mp4")
                        pred_path = os.path.join(temp_dir, "pred.mp4")
                        stitched_path = os.path.join(
                            temp_dir,
                            "stitched.mp4",
                        )
                        audio_path = os.path.join(temp_dir, "audio.wav")
                        render_smplx_debug_video(
                            smplx_model=self.smplx_model,
                            poses=tar_pose_np.reshape(
                                valid_motion_frames,
                                -1,
                            ),
                            transl=tar_trans_viz,
                            expressions=tar_exps_np,
                            betas=beta_np,
                            output_path=gt_path,
                            fps=self.args.motion_fps,
                            mesh_color=(180, 54, 54, 255),
                        )
                        render_smplx_debug_video(
                            smplx_model=self.smplx_model,
                            poses=rec_pose_np.reshape(
                                valid_motion_frames,
                                -1,
                            ),
                            transl=rec_trans_viz,
                            expressions=rec_exps_np,
                            betas=beta_np,
                            output_path=pred_path,
                            fps=self.args.motion_fps,
                            mesh_color=(36, 73, 156, 255),
                        )
                        stitch_videos_hstack(
                            [gt_path, pred_path],
                            stitched_path,
                        )
                        if not os.path.exists(stitched_path):
                            raise RuntimeError(
                                "Oracle side-by-side rendering failed; "
                                f"no output at {stitched_path}."
                            )
                        if raw_audio is None:
                            raise RuntimeError(
                                "Oracle visualization requested but the "
                                "test loader did not return raw audio."
                            )
                        sf.write(
                            audio_path,
                            raw_audio,
                            self.args.audio_fps,
                        )
                        shutil.move(stitched_path, final_path)
                        mux_audio_into_video(final_path, audio_path)
                    logger.info(
                        f"Oracle visualization saved to {final_path}"
                    )
            total_motion_frames += valid_motion_frames
            logger.info(
                f"Oracle clip {file_name[0]}: predicted "
                f"{valid_gesture_steps} gesture frames; excluded final "
                f"{self.minimum_future_anchor_tokens()}."
            )

        elapsed = time.time() - start_time
        raw_metrics = self.gesture_metrics.compute_metrics()
        oracle_metrics = {
            f"oracle_{name}": value
            for name, value in raw_metrics.items()
        }
        if "fgd" in raw_metrics:
            logger.info(
                f"ORACLE future-gesture FGD: {raw_metrics['fgd']}"
            )
        if total_gesture_tokens:
            oracle_metrics["oracle_q0_token_accuracy"] = (
                total_q0_correct / total_gesture_tokens
            )
        if total_code_tokens:
            oracle_metrics["oracle_all_token_accuracy"] = (
                total_all_correct / total_code_tokens
            )
        motion_seconds = total_motion_frames / self.args.motion_fps
        oracle_metrics["oracle_inference_seconds"] = float(elapsed)
        oracle_metrics["oracle_motion_seconds"] = float(motion_seconds)
        oracle_metrics["oracle_future_horizon_seconds"] = float(
            self.future_horizon_seconds
        )
        oracle_metrics["oracle_visible_future_offset_seconds"] = float(
            self.visible_future_offset_seconds()
        )
        oracle_metrics["oracle_excluded_tail_frames_per_clip"] = float(
            self.minimum_future_anchor_tokens()
            * self.args.frame_chunk_size
        )
        if motion_seconds > 0:
            oracle_metrics["oracle_realtime_factor"] = float(
                elapsed / motion_seconds
            )
        logger.warning(
            "Oracle metric report complete. These values use ground-truth "
            "future gesture and must be labeled oracle_* in comparisons."
        )
        return oracle_metrics


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
