"""Fresh-state raw-suffix encoding for future-aware gesture teachers.

The existing masked-frame teacher uses gesture tokens produced by encoding the
complete motion clip.  Because the gesture codecs are causal, a token after the
protected target interval can still carry codec state originating inside that
interval.  This module builds an alternative view in which every future body
part is encoded from the globally preprocessed raw-motion suffix alone.

The helpers deliberately require non-streaming codecs.  A non-streaming
``GestureMimiCodec.encode`` call has no convolution buffers, Transformer KV
caches, or position offsets left over from another sequence.  Rejecting an
already-streaming codec is safer than resetting and then accidentally reusing
its state for the next sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .gesture_codec import GestureMimiCodec
from .gesture_lm_future_gesture import build_masked_future_gesture_inputs


@dataclass(frozen=True)
class ResetSuffixEncoding:
    """Discrete and continuous views of one independently encoded suffix."""

    codes: torch.Tensor
    prequantized_latent: torch.Tensor
    suffix_start_frame: int
    suffix_start_token: int
    valid_token_mask: torch.Tensor


@dataclass(frozen=True)
class ResetFutureBoundary:
    """Auditable frame/token boundary information for one teacher sample."""

    target_token: int
    target_frame: int
    suffix_start_token: int
    suffix_start_frame: int
    encoded_tokens: int
    visible_tokens: int
    offset_ms: float


def _require_fresh_nonstreaming_codec(codec: GestureMimiCodec) -> None:
    """Fail rather than silently reuse any live streaming cache."""

    active_states = [
        name or "<root>"
        for name, state in codec.get_streaming_state().items()
        if state is not None
    ]
    if active_states:
        raise RuntimeError(
            "Reset-suffix encoding requires a completely fresh non-streaming "
            "codec, but active streaming state was found in: "
            + ", ".join(active_states[:10])
        )


def _encode_fresh_segment(
    codec: GestureMimiCodec,
    motion_segment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a same-length ``[B,F,D]`` batch without persistent state."""

    if motion_segment.dim() != 3 or motion_segment.shape[0] <= 0:
        raise ValueError(
            "Fresh segment encoding expects a non-empty [B,F,D] batch, got "
            f"{tuple(motion_segment.shape)}."
        )
    if motion_segment.shape[1] <= 0:
        raise ValueError("Cannot reset-encode an empty motion suffix.")
    _require_fresh_nonstreaming_codec(codec)
    codec.eval()
    with torch.no_grad():
        prequantized = codec.encode_to_latent(
            motion_segment,
            quantize=False,
        )
        quantizer_input = codec._cast_for_module(
            prequantized,
            codec.quantizer,
        )
        codes = codec.quantizer.encode(quantizer_input)
    # Encoding must not leave any cache behind.
    _require_fresh_nonstreaming_codec(codec)
    return codes, prequantized


def encode_intact_motion(
    codec: GestureMimiCodec,
    preprocessed_motion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a complete globally preprocessed clip with no streaming state."""

    return _encode_fresh_segment(codec, preprocessed_motion)


def encode_reset_suffix(
    codec: GestureMimiCodec,
    preprocessed_motion: torch.Tensor,
    *,
    suffix_start_frame: int,
    reset_prefix_drop_tokens: int = 0,
    zero_first_frame_feature_slice: tuple[int, int] | None = None,
    maximum_suffix_frames: int | None = None,
) -> ResetSuffixEncoding:
    """Encode a raw-motion suffix as a deterministic function of that suffix.

    ``preprocessed_motion`` must already be expressed in the full clip's
    coordinate system.  This function only slices; it never realigns,
    recenters, or recomputes speaker/pose normalization.

    ``zero_first_frame_feature_slice`` is used for lower-body translation
    velocity.  Its first suffix-frame derivative otherwise crosses the
    protected interval, so the affected features are explicitly zeroed before
    the independent encoding.
    """

    if preprocessed_motion.dim() != 3:
        raise ValueError(
            "Expected globally preprocessed motion [B,F,D], got "
            f"{tuple(preprocessed_motion.shape)}."
        )
    if preprocessed_motion.shape[0] <= 0:
        raise ValueError("Reset suffix encoding received an empty batch.")
    frame_size = int(codec.frame_size)
    if frame_size <= 0:
        raise ValueError(f"Invalid codec frame size {frame_size}.")
    if suffix_start_frame < 0 or suffix_start_frame >= preprocessed_motion.shape[1]:
        raise ValueError(
            f"suffix_start_frame={suffix_start_frame} is outside a "
            f"{preprocessed_motion.shape[1]}-frame clip."
        )
    if suffix_start_frame % frame_size:
        raise ValueError(
            f"Suffix frame {suffix_start_frame} is not aligned to the "
            f"{frame_size}-frame codec token boundary."
        )
    if reset_prefix_drop_tokens < 0:
        raise ValueError("reset_prefix_drop_tokens cannot be negative.")

    suffix_end_frame = preprocessed_motion.shape[1]
    if maximum_suffix_frames is not None:
        if maximum_suffix_frames <= 0:
            raise ValueError("maximum_suffix_frames must be positive.")
        suffix_end_frame = min(
            suffix_end_frame,
            suffix_start_frame + maximum_suffix_frames,
        )
    suffix_frame_count = suffix_end_frame - suffix_start_frame
    if suffix_frame_count % frame_size:
        # Never allow the encoder to create a partially aligned token.
        suffix_frame_count -= suffix_frame_count % frame_size
        suffix_end_frame = suffix_start_frame + suffix_frame_count
    if suffix_frame_count <= 0:
        raise ValueError("No complete codec token remains in the suffix.")

    suffix = preprocessed_motion[
        :,
        suffix_start_frame:suffix_end_frame,
    ].clone()
    if zero_first_frame_feature_slice is not None:
        feature_start, feature_end = zero_first_frame_feature_slice
        if not 0 <= feature_start < feature_end <= suffix.shape[-1]:
            raise ValueError(
                "Invalid boundary feature slice "
                f"{zero_first_frame_feature_slice} for D={suffix.shape[-1]}."
            )
        suffix[:, 0, feature_start:feature_end] = 0

    codes, prequantized = _encode_fresh_segment(codec, suffix)
    expected_tokens = suffix_frame_count // frame_size
    if codes.shape[-1] != expected_tokens:
        raise RuntimeError(
            "Reset codec token/frame misalignment: "
            f"{suffix_frame_count} frames at frame_size={frame_size} should "
            f"produce {expected_tokens} tokens, got {codes.shape[-1]}."
        )
    valid = torch.ones(
        expected_tokens,
        device=codes.device,
        dtype=torch.bool,
    )
    valid[: min(reset_prefix_drop_tokens, expected_tokens)] = False
    return ResetSuffixEncoding(
        codes=codes,
        prequantized_latent=prequantized,
        suffix_start_frame=suffix_start_frame,
        suffix_start_token=suffix_start_frame // frame_size,
        valid_token_mask=valid,
    )


def build_reset_future_teacher_inputs_from_codes(
    input_codes: torch.Tensor,
    target_codes: torch.Tensor,
    target_times: torch.Tensor,
    *,
    reset_code_windows: Sequence[torch.Tensor],
    horizon_tokens: int,
    past_context_tokens: int,
    mask_token_id: int,
    motion_fps: int,
    frame_size: int,
    reset_prefix_drop_tokens: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[ResetFutureBoundary],
]:
    """Insert independently encoded fixed reset windows into teacher inputs."""

    if input_codes.shape != target_codes.shape or input_codes.dim() != 3:
        raise ValueError(
            "Expected matching input/target codes [B,K,T], got "
            f"{tuple(input_codes.shape)} and {tuple(target_codes.shape)}."
        )
    batch, codebooks, steps = target_codes.shape
    if target_times.shape != (batch,):
        raise ValueError(
            f"Expected target_times shape {(batch,)}, got "
            f"{tuple(target_times.shape)}."
        )
    if len(reset_code_windows) != batch:
        raise ValueError(
            f"Expected {batch} reset windows, got "
            f"{len(reset_code_windows)}."
        )
    if horizon_tokens <= 0 or frame_size <= 0 or motion_fps <= 0:
        raise ValueError(
            "horizon_tokens, frame_size, and motion_fps must be positive."
        )
    if reset_prefix_drop_tokens < 0:
        raise ValueError("reset_prefix_drop_tokens cannot be negative.")

    temporal_codes = build_masked_future_gesture_inputs(
        input_codes,
        target_codes,
        target_times,
        horizon_tokens=horizon_tokens,
        past_context_tokens=past_context_tokens,
        mask_token_id=mask_token_id,
    )
    reset_future_valid = torch.zeros(
        batch,
        steps,
        device=target_codes.device,
        dtype=torch.bool,
    )
    boundaries = []
    for batch_index, (target_time_tensor, reset_window) in enumerate(
        zip(target_times, reset_code_windows)
    ):
        target_token = int(target_time_tensor.item())
        suffix_start_token = target_token + horizon_tokens
        if reset_window.dim() == 3 and reset_window.shape[0] == 1:
            reset_window = reset_window[0]
        if reset_window.dim() != 2:
            raise ValueError(
                "Each cached reset window must be [K,H], got "
                f"{tuple(reset_window.shape)}."
            )
        if reset_window.shape[0] != codebooks:
            raise ValueError(
                f"Reset window has K={reset_window.shape[0]}, expected "
                f"{codebooks}."
            )
        encoded_tokens = min(
            int(reset_window.shape[-1]),
            steps - suffix_start_token,
        )
        if encoded_tokens <= reset_prefix_drop_tokens:
            raise ValueError(
                "Reset window has no visible token after the configured "
                "prefix drop."
            )
        suffix_end_token = suffix_start_token + encoded_tokens
        reset_codes = reset_window[
            :,
            :encoded_tokens,
        ].to(
            device=target_codes.device,
            dtype=target_codes.dtype,
        )

        # Mask the complete future first so no intact token survives outside
        # the fixed reset window.
        temporal_codes[
            batch_index,
            :,
            suffix_start_token:,
        ] = mask_token_id
        original_valid = (
            target_codes[
                batch_index,
                :,
                suffix_start_token:suffix_end_token,
            ]
            != mask_token_id
        )
        temporal_codes[
            batch_index,
            :,
            suffix_start_token:suffix_end_token,
        ] = torch.where(
            original_valid,
            reset_codes,
            torch.full_like(reset_codes, mask_token_id),
        )
        if reset_prefix_drop_tokens:
            temporal_codes[
                batch_index,
                :,
                suffix_start_token:
                suffix_start_token + reset_prefix_drop_tokens,
            ] = mask_token_id

        valid = original_valid.any(dim=0)
        valid[:reset_prefix_drop_tokens] = False
        reset_future_valid[
            batch_index,
            suffix_start_token:suffix_end_token,
        ] = valid
        boundaries.append(
            ResetFutureBoundary(
                target_token=target_token,
                target_frame=target_token * frame_size,
                suffix_start_token=suffix_start_token,
                suffix_start_frame=suffix_start_token * frame_size,
                encoded_tokens=encoded_tokens,
                visible_tokens=int(valid.sum().item()),
                offset_ms=(
                    horizon_tokens
                    * frame_size
                    * 1000.0
                    / motion_fps
                ),
            )
        )
    return temporal_codes, reset_future_valid, boundaries


def build_reset_future_teacher_inputs(
    input_codes: torch.Tensor,
    target_codes: torch.Tensor,
    target_times: torch.Tensor,
    *,
    codec_motion_inputs: Sequence[torch.Tensor],
    gesture_codecs: Sequence[GestureMimiCodec],
    horizon_tokens: int,
    past_context_tokens: int,
    mask_token_id: int,
    motion_fps: int,
    reset_prefix_drop_tokens: int = 0,
    maximum_suffix_frames: int | None = None,
    lower_velocity_feature_slice: tuple[int, int] = (54, 57),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[ResetFutureBoundary],
]:
    """Replace every intact future body-part token with a reset encoding.

    Codebook order follows MIBURI's canonical upper/lower/face layout.  The
    intact view remains responsible for history and labels.  Existing PAD
    positions in ``target_codes`` remain PAD after replacement, which
    preserves unavailable lower/face supervision in mixed datasets.
    """

    if input_codes.shape != target_codes.shape or input_codes.dim() != 3:
        raise ValueError(
            "Expected matching input/target codes [B,K,T], got "
            f"{tuple(input_codes.shape)} and {tuple(target_codes.shape)}."
        )
    batch, codebooks, steps = target_codes.shape
    if target_times.shape != (batch,):
        raise ValueError(
            f"Expected target_times shape {(batch,)}, got "
            f"{tuple(target_times.shape)}."
        )
    if len(codec_motion_inputs) != 3 or len(gesture_codecs) != 3:
        raise ValueError(
            "Reset MIBURI teacher requires exactly upper/lower/face motion "
            "inputs and codecs."
        )
    codec_codebooks = [int(codec.num_codebooks) for codec in gesture_codecs]
    if sum(codec_codebooks) != codebooks:
        raise ValueError(
            f"Codec codebooks {codec_codebooks} do not match K={codebooks}."
        )
    frame_sizes = {int(codec.frame_size) for codec in gesture_codecs}
    if len(frame_sizes) != 1:
        raise ValueError(
            f"All gesture codecs must share a frame size, got {frame_sizes}."
        )
    frame_size = frame_sizes.pop()
    motion_frames = codec_motion_inputs[0].shape[1]
    for part_name, motion in zip(
        ("upper", "lower", "face"),
        codec_motion_inputs,
    ):
        if motion.dim() != 3 or motion.shape[0] != batch:
            raise ValueError(
                f"{part_name} codec input must be [B,F,D], got "
                f"{tuple(motion.shape)}."
            )
        if motion.shape[1] != motion_frames:
            raise ValueError(
                "All body-part motion inputs must share the same frame count."
            )
    if motion_frames != steps * frame_size:
        raise ValueError(
            f"Motion/token alignment mismatch: F={motion_frames}, T={steps}, "
            f"frame_size={frame_size}."
        )

    temporal_codes = build_masked_future_gesture_inputs(
        input_codes,
        target_codes,
        target_times,
        horizon_tokens=horizon_tokens,
        past_context_tokens=past_context_tokens,
        mask_token_id=mask_token_id,
    )
    reset_future_valid = torch.zeros(
        batch,
        steps,
        device=target_codes.device,
        dtype=torch.bool,
    )
    boundaries: list[ResetFutureBoundary] = []

    for batch_index, target_time_tensor in enumerate(target_times):
        target_token = int(target_time_tensor.item())
        suffix_start_token = target_token + horizon_tokens
        suffix_start_frame = suffix_start_token * frame_size
        part_encodings = []
        for part_index, (codec, motion) in enumerate(
            zip(gesture_codecs, codec_motion_inputs)
        ):
            boundary_slice = (
                lower_velocity_feature_slice
                if part_index == 1
                else None
            )
            encoding = encode_reset_suffix(
                codec,
                motion[batch_index:batch_index + 1],
                suffix_start_frame=suffix_start_frame,
                reset_prefix_drop_tokens=reset_prefix_drop_tokens,
                zero_first_frame_feature_slice=boundary_slice,
                maximum_suffix_frames=maximum_suffix_frames,
            )
            part_encodings.append(encoding)

        encoded_lengths = {
            encoding.codes.shape[-1]
            for encoding in part_encodings
        }
        if len(encoded_lengths) != 1:
            raise RuntimeError(
                "Upper/lower/face reset encoders produced different suffix "
                f"lengths: {sorted(encoded_lengths)}."
            )
        encoded_tokens = encoded_lengths.pop()
        reset_codes = torch.cat(
            [encoding.codes for encoding in part_encodings],
            dim=1,
        )[0].to(target_codes.device)
        suffix_end_token = suffix_start_token + encoded_tokens
        if suffix_end_token > steps:
            raise RuntimeError(
                f"Reset suffix ends at token {suffix_end_token}, beyond T={steps}."
            )

        # No intact future token may survive after the reset boundary.
        temporal_codes[
            batch_index,
            :,
            suffix_start_token:,
        ] = mask_token_id
        original_valid = (
            target_codes[
                batch_index,
                :,
                suffix_start_token:suffix_end_token,
            ]
            != mask_token_id
        )
        replacement = torch.where(
            original_valid,
            reset_codes,
            torch.full_like(reset_codes, mask_token_id),
        )
        temporal_codes[
            batch_index,
            :,
            suffix_start_token:suffix_end_token,
        ] = replacement

        valid_mask = part_encodings[0].valid_token_mask.to(
            reset_future_valid.device
        )
        if reset_prefix_drop_tokens:
            dropped_end = min(
                suffix_end_token,
                suffix_start_token + reset_prefix_drop_tokens,
            )
            temporal_codes[
                batch_index,
                :,
                suffix_start_token:dropped_end,
            ] = mask_token_id
        reset_future_valid[
            batch_index,
            suffix_start_token:suffix_end_token,
        ] = valid_mask
        # A token is only usable if at least one canonical codebook exists.
        reset_future_valid[
            batch_index,
            suffix_start_token:suffix_end_token,
        ] &= original_valid.any(dim=0)

        visible_tokens = int(
            reset_future_valid[
                batch_index,
                suffix_start_token:suffix_end_token,
            ].sum().item()
        )
        boundaries.append(
            ResetFutureBoundary(
                target_token=target_token,
                target_frame=target_token * frame_size,
                suffix_start_token=suffix_start_token,
                suffix_start_frame=suffix_start_frame,
                encoded_tokens=encoded_tokens,
                visible_tokens=visible_tokens,
                offset_ms=(
                    horizon_tokens
                    * frame_size
                    * 1000.0
                    / motion_fps
                ),
            )
        )

    return temporal_codes, reset_future_valid, boundaries
