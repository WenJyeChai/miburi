"""Opt-in three-q0, globally coarse-to-fine gesture language model.

This module intentionally does not alter :mod:`miburi.models.gesture_lm`.
It keeps Miburi's temporal/depth split and streaming interface, while changing
the within-frame factorisation to:

    temporal:  upper q0, lower q0, face q0 (three parallel heads)
    kinematic: U1, L1, F1, U2, L2, F2, U3, L3, F3,
               U4, L4, U5, L5, U6, L6, U7, L7

The public logits and generated tokens are always scattered back to Miburi's
canonical codec layout ``upper[0:8], lower[8:16], face[16:20]``.
"""

from __future__ import annotations

from functools import partial
import inspect
import logging
import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from ..modules.transformer import StreamingTransformer, StreamingTransformerDecoderLayer
from ..utils.sampling import sample_token
from .gesture_lm import GTemporalDepthModel3, GestureLMGen
from .lm_utils import _init_layer, ScaledEmbeddingwithPadEmbedding


logger = logging.getLogger(__name__)


# Canonical Miburi slots: upper 0..7, lower 8..15, face 16..19.
COARSE_SLOTS: tuple[int, ...] = (0, 8, 16)
KINEMATIC_SLOTS: tuple[int, ...] = (
    1, 9, 17,
    2, 10, 18,
    3, 11, 19,
    4, 12,
    5, 13,
    6, 14,
    7, 15,
)
PART_SLOTS: tuple[tuple[int, ...], ...] = (
    tuple(range(0, 8)),
    tuple(range(8, 16)),
    tuple(range(16, 20)),
)


def _part_for_slot(slot: int) -> int:
    if slot < 8:
        return 0
    if slot < 16:
        return 1
    return 2


class GTemporalDepthModel3C2F(GTemporalDepthModel3):
    """Miburi variant with three temporal q0 heads and 17 C2F depth steps.

    The parent constructor is used for the unchanged temporal backbone,
    conditioning projections, and embedding processors. Its one-head/depth
    modules are then replaced locally; the legacy class and loader remain
    unchanged.
    """

    coarse_slots = COARSE_SLOTS
    kinematic_slots = KINEMATIC_SLOTS
    part_slots = PART_SLOTS

    def __init__(self, *args, **kwargs):
        requested_heads = kwargs.pop("num_temp_classifiers", 3)
        if requested_heads != 3:
            raise ValueError(
                "GTemporalDepthModel3C2F requires num_temp_classifiers=3, "
                f"got {requested_heads}."
            )

        base_signature = inspect.signature(GTemporalDepthModel3.__init__)
        base_names = set(base_signature.parameters) - {"self", "kwargs"}
        transformer_extras = {
            key: value for key, value in kwargs.items() if key not in base_names
        }

        depformer_dim = kwargs.get("depformer_dim", 256)
        depformer_heads = kwargs.get("depformer_heads", 8)
        depformer_layers = kwargs.get("depformer_layers", 4)
        depformer_dim_feedforward = kwargs.get("depformer_dim_feedforward")
        depformer_multi_linear = kwargs.get("depformer_multi_linear", False)
        depformer_weights_per_step = kwargs.get("depformer_weights_per_step", False)
        depformer_low_rank_embeddings = kwargs.get("depformer_low_rank_embeddings")
        depformer_pos_emb = kwargs.get("depformer_pos_emb", "sin")
        hidden_scale = kwargs.get("hidden_scale", 4)
        norm = kwargs.get("norm", "layer_norm")
        norm_emb = kwargs.get("norm_emb", False)
        bias_proj = kwargs.get("bias_proj", False)
        quantize = kwargs.get("quantize", False)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        gradient_checkpointing = kwargs.get("gradient_checkpointing", False)

        super().__init__(*args, num_temp_classifiers=1, **kwargs)

        if self.n_q != 20:
            raise ValueError(
                "The C2F slot schedule is defined for Miburi's 20 gesture "
                f"codebooks, got n_q={self.n_q}."
            )
        if quantize:
            raise NotImplementedError(
                "Quantized construction is not supported by the experimental "
                "C2F model. Train it from scratch without quantize=True."
            )

        self.num_temp_classifiers = 3
        self.num_kinematic_steps = len(self.kinematic_slots)
        self.depformer_dim = depformer_dim
        self.depformer_multi_linear = depformer_multi_linear

        # Replace the legacy single temporal classifier with independent
        # upper/lower/face q0 heads. They still share the temporal hidden state.
        self.temporal_classifier = nn.ModuleList(
            [nn.Linear(self.dim, self.card + 1, bias=bias_proj) for _ in range(3)]
        )

        pad_embedding_factory = partial(
            ScaledEmbeddingwithPadEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.initial_token_id,
            pad_idx=self.pad_token_id,
            low_rank=depformer_low_rank_embeddings,
        )

        # All three q0 embeddings are summed into a single coarse context.
        self.coarse_gemb = nn.ModuleList()
        self.coarse_gproj = nn.ModuleList()
        for slot in self.coarse_slots:
            codebook = self.gesture_codec_layers[slot]._codebook
            self.coarse_gemb.append(
                pad_embedding_factory(
                    self.card,
                    codebook.dim,
                    _weight=codebook.embedding,
                    _freeze=False,
                )
            )
            self.coarse_gproj.append(
                nn.Linear(codebook.dim, depformer_dim, bias=False)
            )

        # At step j>0, the preceding token in the global C2F order is embedded.
        # Step zero uses a learned start vector in addition to the q0 context.
        self.prefix_gemb = nn.ModuleList()
        self.prefix_gproj = nn.ModuleList()
        for slot in self.kinematic_slots[:-1]:
            codebook = self.gesture_codec_layers[slot]._codebook
            self.prefix_gemb.append(
                pad_embedding_factory(
                    self.card,
                    codebook.dim,
                    _weight=codebook.embedding,
                    _freeze=False,
                )
            )
            self.prefix_gproj.append(
                nn.Linear(codebook.dim, depformer_dim, bias=False)
            )
        self.depth_start = nn.Parameter(
            torch.empty(1, 1, depformer_dim, device=device, dtype=dtype)
        )

        if depformer_multi_linear:
            self.depformer_in = nn.ModuleList(
                [
                    nn.Linear(self.dim, depformer_dim, bias=False)
                    for _ in range(self.num_kinematic_steps)
                ]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(self.dim, depformer_dim, bias=False)]
            )

        # Keep a distinct speaker embedding for each refinement position, as in
        # the original Miburi depformer.
        embedding_factory = partial(
            type(self.spk_tempemb),
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.initial_token_id,
            low_rank=depformer_low_rank_embeddings,
        )
        self.spk_depemb = nn.ModuleList(
            [
                embedding_factory(self.num_spks, depformer_dim)
                for _ in range(self.num_kinematic_steps)
            ]
        )

        depth_kwargs = dict(transformer_extras)
        depth_kwargs.update(
            {
                key.removeprefix("depformer_"): value
                for key, value in transformer_extras.items()
                if key.startswith("depformer_")
            }
        )
        depth_kwargs = {
            key: value
            for key, value in depth_kwargs.items()
            if not key.startswith("depformer_")
        }
        depth_kwargs["positional_embedding"] = depformer_pos_emb
        if depformer_weights_per_step:
            depth_kwargs["weights_per_step"] = self.num_kinematic_steps
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)

        self.depth_transformer = StreamingTransformer(
            d_model=depformer_dim,
            num_heads=depformer_heads,
            num_layers=depformer_layers,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            device=device,
            dtype=dtype,
            context=self.num_kinematic_steps,
            memory_context=self.query2mem_scale,
            upsample_factor=1,
            quantize=False,
            causal=self.causal,
            crossattn_causal=False,
            checkpointing=gradient_checkpointing,
            layer_class=StreamingTransformerDecoderLayer,
            num_memories=2,
            **depth_kwargs,
        )
        if depformer_weights_per_step:
            # Miburi's StreamingMultiheadCrossAttention advances its K/V
            # weight offset by memory length on every streaming depth step
            # (the implementation marks this path TODO). Sharing K/V is also
            # the appropriate factorisation here: audio/text memories are
            # common to all depth steps, while Q/output remain step-specific.
            for layer in self.depth_transformer.layers:
                for cross_attention in layer.cross_attns:
                    cross_attention.key_projs = nn.ModuleList(
                        [cross_attention.key_projs[0]]
                    )
                    cross_attention.val_projs = nn.ModuleList(
                        [cross_attention.val_projs[0]]
                    )
        self.depth_transformer.set_streaming_detached(True)
        self.depformer_classifier = nn.ModuleList(
            [
                nn.Linear(depformer_dim, self.card + 1, bias=bias_proj)
                for _ in range(self.num_kinematic_steps)
            ]
        )

        # These legacy depth embeddings are no longer part of this model.
        self.depformer_gemb = nn.ModuleList()
        self.depformer_gproj = nn.ModuleList()

        self.last_kinematic_input_codes: torch.Tensor | None = None
        self.last_used_self_forcing = False
        self._init_c2f_weights()
        self.to(device=device, dtype=dtype)

    def _init_c2f_weights(self) -> None:
        for head in self.temporal_classifier:
            _init_layer(head)
        for projection in self.coarse_gproj:
            _init_layer(projection)
        for projection in self.prefix_gproj:
            _init_layer(projection)
        for embedding in self.coarse_gemb:
            _init_layer(embedding.pad_embedding)
        for embedding in self.prefix_gemb:
            _init_layer(embedding.pad_embedding)
        for projection in self.depformer_in:
            _init_layer(projection)
        for classifier in self.depformer_classifier:
            _init_layer(classifier)
        for layer in self.depth_transformer.layers:
            layer.apply(_init_layer)
        nn.init.trunc_normal_(
            self.depth_start,
            mean=0.0,
            std=self.depformer_dim ** -0.5,
            a=-3 * self.depformer_dim ** -0.5,
            b=3 * self.depformer_dim ** -0.5,
        )

    def forward_temporal(
        self,
        sequence: torch.Tensor,
        audio_condition: torch.Tensor,
        text_condition: torch.Tensor,
        sum_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the shared temporal backbone and emit three parallel q0 logits."""
        _, K, _ = sequence.shape
        if K != self.n_q:
            raise ValueError(f"Expected {self.n_q} codebooks, got {K}.")

        input_: torch.Tensor | None = None
        for slot in range(self.n_q):
            slot_emb = self.temporal_gemb[slot](sequence[:, slot])
            slot_emb = self.temporal_gproj[slot](slot_emb)
            input_ = slot_emb if input_ is None else input_ + slot_emb
        assert input_ is not None

        if sum_condition is not None:
            input_ = input_ + self.spk_tempemb(sum_condition).to(input_)

        audio_emb = self.temp_condproj[0](audio_condition.to(input_))
        text_emb = self.temp_condproj[1](text_condition.to(input_))
        transformer_out = self.temporal_transformer(
            input_,
            memories=[audio_emb, text_emb],
        )
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)

        logits = torch.stack(
            [head(transformer_out) for head in self.temporal_classifier],
            dim=1,
        )
        return transformer_out, logits

    def _coarse_context(self, coarse_codes: torch.Tensor) -> torch.Tensor:
        """Sum upper/lower/face q0 embeddings into one depth-conditioning vector."""
        if coarse_codes.shape[1] != len(self.coarse_slots):
            raise ValueError(
                f"Expected three q0 streams, got shape {tuple(coarse_codes.shape)}."
            )
        context: torch.Tensor | None = None
        for index in range(len(self.coarse_slots)):
            embedded = self.coarse_gemb[index](coarse_codes[:, index])
            embedded = self.coarse_gproj[index](embedded)
            context = embedded if context is None else context + embedded
        assert context is not None
        return context

    def _depth_memories(
        self,
        audio_condition: torch.Tensor,
        text_condition: torch.Tensor,
        B: int,
        T: int,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        expected = T * self.query2mem_scale
        if audio_condition.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} audio frames for {T} gesture frames, "
                f"got {audio_condition.shape[1]}."
            )
        audio_emb = audio_condition.reshape(
            B, T, self.query2mem_scale, self.cond_dim
        ).reshape(B * T, self.query2mem_scale, self.cond_dim)
        text_emb = text_condition.reshape(
            B, T, self.query2mem_scale, self.cond_dim
        ).reshape(B * T, self.query2mem_scale, self.cond_dim)
        return [
            self.dep_condproj[0](audio_emb.to(dtype)),
            self.dep_condproj[1](text_emb.to(dtype)),
        ]

    def forward_depth_training(
        self,
        codes: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_condition: torch.Tensor,
        text_condition: torch.Tensor,
        sum_condition: torch.Tensor | None = None,
        ca_query_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict all 17 residual slots using a shifted global-C2F prefix."""
        B, K, T = codes.shape
        if K != self.n_q:
            raise ValueError(f"Expected {self.n_q} input codebooks, got {K}.")

        coarse_codes = codes[:, self.coarse_slots]
        coarse_context = self._coarse_context(coarse_codes)
        depth_inputs = []
        input_padding = []

        for step, _target_slot in enumerate(self.kinematic_slots):
            temporal = self.depformer_in[
                step if self.depformer_multi_linear else 0
            ](transformer_out)
            if step == 0:
                prefix = self.depth_start.expand(B, T, -1)
                prefix_padding = torch.zeros(
                    B, T, dtype=torch.bool, device=codes.device
                )
            else:
                previous_slot = self.kinematic_slots[step - 1]
                prefix = self.prefix_gemb[step - 1](codes[:, previous_slot])
                prefix = self.prefix_gproj[step - 1](prefix)
                prefix_padding = codes[:, previous_slot] == self.pad_token_id
            depth_inputs.append(temporal + coarse_context + prefix)
            input_padding.append(prefix_padding)

        depformer_inputs = torch.stack(depth_inputs, dim=2).reshape(
            B * T, self.num_kinematic_steps, self.depformer_dim
        )
        key_padding_mask = torch.stack(input_padding, dim=2).reshape(
            B * T, self.num_kinematic_steps
        )

        if sum_condition is not None:
            speaker_inputs = torch.stack(
                [embedding(sum_condition) for embedding in self.spk_depemb],
                dim=2,
            ).reshape(B * T, self.num_kinematic_steps, self.depformer_dim)
            depformer_inputs = depformer_inputs + speaker_inputs.to(depformer_inputs)

        if self.body_part_emb is not None and self.bp_dist is not None:
            part_indices = self.bp_dist[
                torch.tensor(self.kinematic_slots, dtype=torch.long)
            ].to(depformer_inputs.device)
            depformer_inputs = depformer_inputs + self.body_part_emb(
                part_indices
            ).unsqueeze(0).to(depformer_inputs)

        query_mask = None
        if ca_query_padding_mask is not None:
            expected_shape = (B, self.num_kinematic_steps, T)
            if ca_query_padding_mask.shape != expected_shape:
                raise ValueError(
                    f"Expected depth query mask {expected_shape}, got "
                    f"{tuple(ca_query_padding_mask.shape)}."
                )
            query_mask = ca_query_padding_mask.permute(0, 2, 1).reshape(
                B * T, self.num_kinematic_steps
            )

        depformer_out = self.depth_transformer(
            depformer_inputs,
            memories=self._depth_memories(
                audio_condition,
                text_condition,
                B,
                T,
                depformer_inputs.dtype,
            ),
            key_padding_mask=key_padding_mask,
            ca_query_padding_mask=query_mask,
        )
        return torch.stack(
            [
                classifier(depformer_out[:, step]).reshape(B, T, -1)
                for step, classifier in enumerate(self.depformer_classifier)
            ],
            dim=1,
        )

    def forward_depth(
        self,
        step: int,
        previous_code: torch.Tensor,
        coarse_codes: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_condition: torch.Tensor,
        text_condition: torch.Tensor,
        sum_condition: torch.Tensor | None = None,
        ca_query_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One streaming kinematic step for training rollout or online inference."""
        B, K, T = previous_code.shape
        if K != 1 or T != 1:
            raise ValueError("previous_code must have shape [B, 1, 1].")
        if coarse_codes.shape != (B, 3, 1):
            raise ValueError(
                f"coarse_codes must have shape {(B, 3, 1)}, got "
                f"{tuple(coarse_codes.shape)}."
            )

        temporal = self.depformer_in[
            step if self.depformer_multi_linear else 0
        ](transformer_out)
        coarse = self._coarse_context(coarse_codes)
        if step == 0:
            prefix = self.depth_start.expand(B, 1, -1)
        else:
            prefix = self.prefix_gemb[step - 1](previous_code[:, 0])
            prefix = self.prefix_gproj[step - 1](prefix)
        depformer_input = temporal + coarse + prefix

        if sum_condition is not None:
            depformer_input = depformer_input + self.spk_depemb[step](
                sum_condition
            ).to(depformer_input)

        if self.body_part_emb is not None and self.bp_dist is not None:
            slot = self.kinematic_slots[step]
            part_index = self.bp_dist[slot].to(depformer_input.device)
            depformer_input = depformer_input + self.body_part_emb(
                part_index
            ).view(1, 1, -1).to(depformer_input)

        query_mask = None
        if ca_query_padding_mask is not None:
            if ca_query_padding_mask.shape != (B, 1, 1):
                raise ValueError(
                    "A streaming depth query mask must have shape [B, 1, 1]."
                )
            query_mask = ca_query_padding_mask[:, :, 0]

        memories = [
            self.dep_condproj[0](audio_condition.to(depformer_input)),
            self.dep_condproj[1](text_condition.to(depformer_input)),
        ]
        depformer_out = self.depth_transformer(
            depformer_input,
            memories=memories,
            ca_query_padding_mask=query_mask,
        )
        return self.depformer_classifier[step](depformer_out).unsqueeze(1)

    @torch.no_grad()
    def _greedy_self_forced_codes(
        self,
        target_codes: torch.Tensor,
        transformer_out: torch.Tensor,
        temporal_logits: torch.Tensor,
        audio_condition: torch.Tensor,
        text_condition: torch.Tensor,
        sum_condition: torch.Tensor | None,
        ca_query_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Roll out detached model prefixes for scheduled self-forcing."""
        B, _, T = target_codes.shape
        generated = torch.full_like(target_codes, self.pad_token_id)
        coarse = temporal_logits.argmax(dim=-1)
        for coarse_index, slot in enumerate(self.coarse_slots):
            valid = target_codes[:, slot] != self.pad_token_id
            generated[:, slot] = torch.where(
                valid,
                coarse[:, coarse_index],
                torch.full_like(coarse[:, coarse_index], self.pad_token_id),
            )

        flat_transformer = transformer_out.reshape(B * T, 1, self.dim)
        flat_audio = audio_condition.reshape(
            B, T, self.query2mem_scale, self.cond_dim
        ).reshape(B * T, self.query2mem_scale, self.cond_dim)
        flat_text = text_condition.reshape(
            B, T, self.query2mem_scale, self.cond_dim
        ).reshape(B * T, self.query2mem_scale, self.cond_dim)
        flat_speaker = (
            sum_condition.reshape(B * T, 1)
            if sum_condition is not None
            else None
        )
        flat_coarse = generated[:, self.coarse_slots].permute(0, 2, 1).reshape(
            B * T, 3, 1
        )
        flat_query_mask = None
        if ca_query_padding_mask is not None:
            flat_query_mask = ca_query_padding_mask.permute(0, 2, 1).reshape(
                B * T, self.num_kinematic_steps, 1
            )

        if self.depth_transformer.is_streaming:
            raise RuntimeError("Depth transformer is already in streaming mode.")
        previous = torch.full(
            (B * T, 1, 1),
            self.initial_token_id,
            dtype=torch.long,
            device=target_codes.device,
        )
        with self.depth_transformer.streaming(B * T):
            for step, slot in enumerate(self.kinematic_slots):
                step_mask = (
                    flat_query_mask[:, step : step + 1]
                    if flat_query_mask is not None
                    else None
                )
                logits = self.forward_depth(
                    step,
                    previous,
                    flat_coarse,
                    flat_transformer,
                    flat_audio,
                    flat_text,
                    sum_condition=flat_speaker,
                    ca_query_padding_mask=step_mask,
                )
                logits[..., self.pad_token_id] = float("-inf")
                next_token = logits.argmax(dim=-1).reshape(B, T)
                valid = target_codes[:, slot] != self.pad_token_id
                next_token = torch.where(
                    valid,
                    next_token,
                    torch.full_like(next_token, self.pad_token_id),
                )
                generated[:, slot] = next_token
                previous = next_token.reshape(B * T, 1, 1)
        return generated

    def forward(
        self,
        codes: torch.Tensor,
        audio_codes: torch.Tensor,
        text_codes: torch.Tensor,
        sum_condition: torch.Tensor | None = None,
        ca_depth_padding_mask: torch.Tensor | None = None,
        self_force_kinematic: bool = False,
        kinematic_target_codes: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Training forward with optional detached, self-forced depth prefixes."""
        B, K, T = codes.shape
        if K != self.n_q:
            raise ValueError(f"Expected {self.n_q} codebooks, got {K}.")
        if ca_depth_padding_mask is not None and ca_depth_padding_mask.shape != codes.shape:
            raise ValueError(
                f"Expected ca_depth_padding_mask {tuple(codes.shape)}, got "
                f"{tuple(ca_depth_padding_mask.shape)}."
            )

        initial = self._get_initial_token().expand(B, K, -1)
        temporal_input = torch.cat([initial, codes], dim=2)[:, :, :-1]
        temporal_speaker = (
            sum_condition.unsqueeze(1).expand(-1, T)
            if sum_condition is not None
            else None
        )
        audio_condition, text_condition = self.process_conditions(
            audio_codes, text_codes
        )
        audio_condition = audio_condition.squeeze(1)
        text_condition = text_condition.squeeze(1)

        transformer_out, temporal_logits = self.forward_temporal(
            temporal_input,
            audio_condition,
            text_condition,
            temporal_speaker,
        )
        depth_query_mask = (
            ca_depth_padding_mask[:, self.kinematic_slots]
            if ca_depth_padding_mask is not None
            else None
        )

        self.last_used_self_forcing = bool(self_force_kinematic)
        if self_force_kinematic:
            rollout_targets = (
                kinematic_target_codes
                if kinematic_target_codes is not None
                else codes
            )
            if rollout_targets.shape != codes.shape:
                raise ValueError(
                    "kinematic_target_codes must match the model input shape."
                )
            depth_input_codes = self._greedy_self_forced_codes(
                rollout_targets,
                transformer_out,
                temporal_logits,
                audio_condition,
                text_condition,
                temporal_speaker,
                depth_query_mask,
            )
            self.last_kinematic_input_codes = depth_input_codes.detach()
        else:
            depth_input_codes = codes
            self.last_kinematic_input_codes = None

        depth_logits = self.forward_depth_training(
            depth_input_codes,
            transformer_out,
            audio_condition,
            text_condition,
            sum_condition=temporal_speaker,
            ca_query_padding_mask=depth_query_mask,
        )

        combined_logits = temporal_logits.new_empty(
            B, self.n_q, T, self.card + 1
        )
        combined_logits[:, self.coarse_slots] = temporal_logits
        combined_logits[:, self.kinematic_slots] = depth_logits

        if self.training and self.vad_guidance:
            if self.vad_use_face_logits:
                face_logits = combined_logits[:, 16:20]
                face_flat = face_logits.permute(0, 2, 1, 3).reshape(B, T, -1)
                vad_input = torch.cat(
                    [transformer_out, self.vad_face_proj(face_flat)], dim=-1
                )
            else:
                vad_input = transformer_out
            vad_logits = self.vad_predictor(vad_input).squeeze(-1)
            return combined_logits, vad_logits
        return combined_logits

    def soft_recovery_loss(
        self,
        logits: torch.Tensor,
        target_codes: torch.Tensor,
        prefix_codes: torch.Tensor,
        *,
        topk: int = 8,
        sigma_scale: float = 1.0,
        only_wrong_prefix: bool = True,
    ) -> torch.Tensor:
        """Soft residual target loss for self-forced kinematic prefixes.

        For each residual codebook, this reconstructs the target part's final
        RVQ latent, subtracts the generated same-part prefix, and distributes
        target probability over the nearest entries of the current codebook.
        The canonical hard-token CE remains the primary trainer loss.
        """
        if logits.shape[:3] != target_codes.shape:
            raise ValueError("Logit/token shapes do not agree.")
        if prefix_codes.shape != target_codes.shape:
            raise ValueError("prefix_codes must match target_codes.")
        if topk < 1:
            raise ValueError("topk must be positive.")

        total = logits.sum() * 0.0
        count = 0
        earlier_slots: list[int] = list(self.coarse_slots)

        for slot in self.kinematic_slots:
            part = _part_for_slot(slot)
            part_slots = self.part_slots[part]
            same_part_prefix = [
                prefix_slot
                for prefix_slot in earlier_slots
                if _part_for_slot(prefix_slot) == part
            ]
            valid = (target_codes[:, part_slots] != self.pad_token_id).all(dim=1)
            if only_wrong_prefix:
                wrong = torch.zeros_like(valid)
                for prefix_slot in same_part_prefix:
                    wrong |= (
                        prefix_codes[:, prefix_slot]
                        != target_codes[:, prefix_slot]
                    )
                valid &= wrong

            if valid.any():
                with torch.no_grad():
                    target_latent: torch.Tensor | None = None
                    for part_slot in part_slots:
                        weight = (
                            self.gesture_codec_layers[part_slot]
                            ._codebook.embedding.detach()
                        )
                        token = target_codes[:, part_slot].clamp(
                            min=0, max=self.card - 1
                        )
                        embedded = F.embedding(token, weight)
                        target_latent = (
                            embedded
                            if target_latent is None
                            else target_latent + embedded
                        )
                    assert target_latent is not None

                    prefix_latent = torch.zeros_like(target_latent)
                    for prefix_slot in same_part_prefix:
                        weight = (
                            self.gesture_codec_layers[prefix_slot]
                            ._codebook.embedding.detach()
                        )
                        token = prefix_codes[:, prefix_slot].clamp(
                            min=0, max=self.card - 1
                        )
                        prefix_latent += F.embedding(token, weight)
                    residual = (target_latent - prefix_latent)[valid].float()

                    current_weight = (
                        self.gesture_codec_layers[slot]
                        ._codebook.embedding.detach()
                        .float()
                    )
                    distances = (
                        residual.square().sum(dim=-1, keepdim=True)
                        + current_weight.square().sum(dim=-1).unsqueeze(0)
                        - 2.0 * residual @ current_weight.t()
                    )
                    k = min(topk, self.card)
                    nearest_distances, nearest_indices = torch.topk(
                        distances, k=k, dim=-1, largest=False
                    )
                    shifted = nearest_distances - nearest_distances[:, :1]
                    if k > 1:
                        variance = shifted[:, 1:].median(dim=-1).values
                    else:
                        variance = torch.ones_like(shifted[:, 0])
                    variance = (
                        variance * float(sigma_scale) ** 2
                    ).clamp_min(1e-6)
                    soft_targets = torch.softmax(
                        -shifted / (2.0 * variance.unsqueeze(-1)), dim=-1
                    )

                slot_logits = logits[:, slot, :, : self.card][valid].float()
                selected_log_probs = torch.gather(
                    F.log_softmax(slot_logits, dim=-1),
                    dim=-1,
                    index=nearest_indices,
                )
                per_item = -(soft_targets * selected_log_probs).sum(dim=-1)
                total = total + per_item.sum()
                count += per_item.numel()
            earlier_slots.append(slot)

        return total / max(count, 1)


class GestureLMC2FGen(GestureLMGen):
    """Streaming generator matching :class:`GTemporalDepthModel3C2F`."""

    glm_model: GTemporalDepthModel3C2F

    @torch.no_grad()
    def step(
        self,
        condition: torch.Tensor | tp.List[torch.Tensor],
        ca_query_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor] | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError("Streaming state is not initialized.")
        glm_model = self.glm_model

        audio_tokens = condition[:, 1:]
        text_tokens = condition[:, :1]
        if self.cfg_coef != 1.0:
            audio_tokens = torch.cat(
                [audio_tokens, torch.full_like(audio_tokens, self.audio_codec_nulltoken)],
                dim=0,
            )
            text_tokens = torch.cat(
                [text_tokens, torch.full_like(text_tokens, self.text_codec_nulltoken)],
                dim=0,
            )
        audio_emb, text_emb = self.process_conditions(audio_tokens, text_tokens)
        B_cfg, condT, _ = audio_emb.shape
        B = state.batch_size
        expected_batch = B if self.cfg_coef == 1.0 else 2 * B
        if B_cfg != expected_batch:
            raise ValueError(f"Expected condition batch {expected_batch}, got {B_cfg}.")
        if condT != glm_model.query2mem_scale:
            raise ValueError(
                f"Expected {glm_model.query2mem_scale} condition frames, got {condT}."
            )

        context_length = state.cache.shape[2]
        position = state.offset % context_length
        if state.offset == 0:
            state.cache[:, :, position] = glm_model.initial_token_id
        temporal_input = state.cache[:, :, position : position + 1]
        if self.check and (temporal_input == glm_model.ungenerated_token_id).any():
            raise RuntimeError("Temporal cache contains an ungenerated token.")

        sum_condition = state.condition_sum.expand(B_cfg, 1)
        if self.cfg_coef != 1.0:
            temporal_input = temporal_input.repeat(2, 1, 1)
        transformer_out, temporal_logits = state.graphed_temp(
            temporal_input,
            audio_emb,
            text_emb,
            sum_condition,
        )
        if self.cfg_coef != 1.0:
            temporal_logits, null_logits = temporal_logits.chunk(2, dim=0)
            temporal_logits = null_logits + self.cfg_coef * (
                temporal_logits - null_logits
            )
        temporal_logits[..., glm_model.pad_token_id] = float("-inf")
        coarse_tokens = sample_token(
            temporal_logits.float(),
            use_sampling=self.use_sampling,
            temp=self.temp_temporal,
            top_k=self.top_k_temp,
            top_p=self.top_p_temp,
        )
        if coarse_tokens.shape != (B, 3, 1):
            raise RuntimeError(
                f"Expected sampled q0 shape {(B, 3, 1)}, got "
                f"{tuple(coarse_tokens.shape)}."
            )
        coarse_tokens = coarse_tokens[:, :, 0]

        depth_mask = (
            ca_query_padding_mask[:, glm_model.kinematic_slots]
            if ca_query_padding_mask is not None
            else None
        )
        cfg_stop_mask = (
            depth_mask[0, :, 0].cpu().tolist()
            if self.cfg_coef != 1.0 and depth_mask is not None
            else None
        )
        depth_tokens = state.graphed_depth(
            coarse_tokens,
            transformer_out,
            audio_emb,
            text_emb,
            sum_condition,
            depth_mask,
            cfg_stop_mask,
            None,
        )

        full_tokens = torch.empty(
            B, glm_model.n_q, dtype=torch.long, device=self.device
        )
        full_tokens[:, glm_model.coarse_slots] = coarse_tokens
        full_tokens[:, glm_model.kinematic_slots] = depth_tokens

        state.offset += 1
        position = state.offset % context_length
        state.cache[:, :, position] = full_tokens
        return state.cache[:, :, position : position + 1]

    def depformer_step(
        self,
        coarse_tokens: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_condition: torch.Tensor,
        text_condition: torch.Tensor,
        sum_condition: torch.Tensor | None = None,
        ca_query_padding_mask: torch.Tensor | None = None,
        cfg_stop_mask: list[bool] | None = None,
        bp_dist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del bp_dist
        B, num_coarse = coarse_tokens.shape
        if num_coarse != 3:
            raise ValueError("Expected three coarse tokens.")
        B_cfg = B if self.cfg_coef == 1.0 else 2 * B
        glm_model = self.glm_model
        previous = torch.full(
            (B, 1, 1),
            glm_model.initial_token_id,
            dtype=torch.long,
            device=coarse_tokens.device,
        )
        generated: list[torch.Tensor] = []

        if glm_model.depth_transformer.is_streaming:
            raise RuntimeError("Depth transformer is already streaming.")
        with glm_model.depth_transformer.streaming(B_cfg):
            for step in range(glm_model.num_kinematic_steps):
                step_mask = (
                    ca_query_padding_mask[:, step : step + 1]
                    if ca_query_padding_mask is not None
                    else None
                )
                cfg_stop = (
                    cfg_stop_mask[step] if cfg_stop_mask is not None else False
                )
                input_previous = previous
                input_coarse = coarse_tokens[:, :, None]
                if self.cfg_coef != 1.0:
                    input_previous = input_previous.repeat(2, 1, 1)
                    input_coarse = input_coarse.repeat(2, 1, 1)
                    if step_mask is not None:
                        step_mask = step_mask.repeat(2, 1, 1)

                logits = glm_model.forward_depth(
                    step,
                    input_previous,
                    input_coarse,
                    transformer_out,
                    audio_condition,
                    text_condition,
                    sum_condition=sum_condition,
                    ca_query_padding_mask=step_mask,
                )
                if self.cfg_coef != 1.0:
                    logits, null_logits = logits.chunk(2, dim=0)
                    if not cfg_stop:
                        logits = null_logits + self.cfg_coef * (
                            logits - null_logits
                        )
                logits[..., glm_model.pad_token_id] = float("-inf")
                next_token = sample_token(
                    logits.float(),
                    use_sampling=self.use_sampling,
                    temp=self.temp_depth,
                    top_k=self.top_k_depth,
                    top_p=self.top_p_depth,
                )
                if next_token.shape != (B, 1, 1):
                    raise RuntimeError(
                        f"Unexpected depth token shape {tuple(next_token.shape)}."
                    )
                previous = next_token
                generated.append(next_token[:, 0, 0])

        return torch.stack(generated, dim=1)
