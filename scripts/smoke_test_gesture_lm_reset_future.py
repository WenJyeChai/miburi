"""Focused CPU checks for raw reset-suffix future-teacher construction.

These tests use a deterministic lightweight codec double so they isolate the
information boundary itself.  A remote checkpoint smoke run should additionally
exercise the same helpers with the three released gesture codecs.
"""

from __future__ import annotations

import torch

from miburi.models.gesture_lm_reset_future import (
    build_reset_future_teacher_inputs,
    build_reset_future_teacher_inputs_from_codes,
    encode_reset_suffix,
)


class _DeterministicQuantizer:
    def __init__(self, num_codebooks: int, cardinality: int = 31):
        self.num_codebooks = num_codebooks
        self.cardinality = cardinality

    def encode(self, latent):
        base = torch.round(latent.sum(dim=1) * 100).long()
        return torch.stack(
            [
                (base + codebook_index * 7) % self.cardinality
                for codebook_index in range(self.num_codebooks)
            ],
            dim=1,
        )


class _DeterministicFreshCodec:
    def __init__(self, num_codebooks: int, frame_size: int = 2):
        self.num_codebooks = num_codebooks
        self.frame_size = frame_size
        self.quantizer = _DeterministicQuantizer(num_codebooks)
        self.training = True
        self.active_state = None

    def eval(self):
        self.training = False
        return self

    def get_streaming_state(self):
        return {"": self.active_state}

    def encode_to_latent(self, motion, quantize=False):
        assert not quantize
        batch, frames, features = motion.shape
        assert frames % self.frame_size == 0
        return (
            motion.reshape(
                batch,
                frames // self.frame_size,
                self.frame_size,
                features,
            )
            .mean(dim=2)
            .transpose(1, 2)
        )

    @staticmethod
    def _cast_for_module(latent, module):
        del module
        return latent


def _motion(batch=2, frames=16, features=8):
    values = torch.arange(
        batch * frames * features,
        dtype=torch.float32,
    )
    return values.reshape(batch, frames, features) / 1000


def test_hidden_interval_and_past_invariance():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    motion = _motion(batch=1)
    suffix_start = 8
    reference = encode_reset_suffix(
        codec,
        motion,
        suffix_start_frame=suffix_start,
    )

    variants = []
    zeroed = motion.clone()
    zeroed[:, :suffix_start] = 0
    variants.append(zeroed)
    randomized = motion.clone()
    randomized[:, :suffix_start] = torch.randn_like(
        randomized[:, :suffix_start]
    )
    variants.append(randomized)
    replaced = motion.clone()
    replaced[:, :suffix_start] = _motion(batch=1)[
        :,
        :suffix_start,
    ].flip(1)
    variants.append(replaced)

    for variant in variants:
        encoded = encode_reset_suffix(
            codec,
            variant,
            suffix_start_frame=suffix_start,
        )
        torch.testing.assert_close(
            encoded.prequantized_latent,
            reference.prequantized_latent,
        )
        torch.testing.assert_close(encoded.codes, reference.codes)


def test_future_sensitivity_and_cache_contamination():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    first = _motion(batch=1)
    second = first.clone()
    second[:, 10:] += 0.25

    first_a = encode_reset_suffix(
        codec,
        first,
        suffix_start_frame=8,
    )
    second_encoded = encode_reset_suffix(
        codec,
        second,
        suffix_start_frame=8,
    )
    first_b = encode_reset_suffix(
        codec,
        first,
        suffix_start_frame=8,
    )

    assert not torch.equal(first_a.codes, second_encoded.codes)
    assert not torch.equal(
        first_a.prequantized_latent,
        second_encoded.prequantized_latent,
    )
    torch.testing.assert_close(first_a.codes, first_b.codes)
    torch.testing.assert_close(
        first_a.prequantized_latent,
        first_b.prequantized_latent,
    )


def test_fixed_window_batch_matches_individual_resets():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    motion = _motion(batch=3, frames=12)
    batched = encode_reset_suffix(
        codec,
        motion,
        suffix_start_frame=0,
        maximum_suffix_frames=8,
    )
    assert batched.codes.shape == (3, 2, 4)
    for sample_index in range(motion.shape[0]):
        individual = encode_reset_suffix(
            codec,
            motion[sample_index:sample_index + 1],
            suffix_start_frame=0,
            maximum_suffix_frames=8,
        )
        torch.testing.assert_close(
            batched.codes[sample_index:sample_index + 1],
            individual.codes,
        )
        torch.testing.assert_close(
            batched.prequantized_latent[
                sample_index:sample_index + 1
            ],
            individual.prequantized_latent,
        )


def test_active_streaming_state_is_rejected():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    codec.active_state = object()
    try:
        encode_reset_suffix(
            codec,
            _motion(batch=1),
            suffix_start_frame=8,
        )
    except RuntimeError as exc:
        assert "active streaming state" in str(exc)
    else:
        raise AssertionError("A contaminated codec state was accepted.")


def test_lower_boundary_velocity_is_internal_to_suffix():
    codec = _DeterministicFreshCodec(num_codebooks=2)
    motion = _motion(batch=1)
    changed = motion.clone()
    changed[:, 8, 1:4] += 1000
    reference = encode_reset_suffix(
        codec,
        motion,
        suffix_start_frame=8,
        zero_first_frame_feature_slice=(1, 4),
    )
    encoded = encode_reset_suffix(
        codec,
        changed,
        suffix_start_frame=8,
        zero_first_frame_feature_slice=(1, 4),
    )
    torch.testing.assert_close(
        encoded.prequantized_latent,
        reference.prequantized_latent,
    )
    torch.testing.assert_close(encoded.codes, reference.codes)


def test_all_body_parts_are_reset_and_boundaries_align():
    codecs = (
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(1),
    )
    motions = (
        _motion(features=8),
        _motion(features=8) + 0.1,
        _motion(features=8) + 0.2,
    )
    batch = motions[0].shape[0]
    steps = motions[0].shape[1] // codecs[0].frame_size
    codebooks = sum(codec.num_codebooks for codec in codecs)
    intact = torch.arange(
        batch * codebooks * steps,
        dtype=torch.long,
    ).reshape(batch, codebooks, steps) % 31
    targets = torch.tensor([1, 2])

    temporal, valid, boundaries = build_reset_future_teacher_inputs(
        intact,
        intact,
        targets,
        codec_motion_inputs=motions,
        gesture_codecs=codecs,
        horizon_tokens=2,
        past_context_tokens=3,
        mask_token_id=31,
        motion_fps=25,
        reset_prefix_drop_tokens=0,
        lower_velocity_feature_slice=(1, 4),
    )

    assert temporal.shape == intact.shape
    assert valid.shape == (batch, steps)
    assert len(boundaries) == batch
    for batch_index, boundary in enumerate(boundaries):
        assert boundary.target_frame == boundary.target_token * 2
        assert boundary.suffix_start_token == boundary.target_token + 2
        assert boundary.suffix_start_frame == (
            boundary.suffix_start_token * 2
        )
        assert boundary.offset_ms == 160.0
        start = boundary.suffix_start_token
        assert (temporal[batch_index, :, start:] != 31).any()
        # Every canonical part must differ from the intact future somewhere;
        # otherwise one body-part stream was accidentally left untouched.
        offsets = (0, 2, 4, 5)
        for part_start, part_end in zip(offsets[:-1], offsets[1:]):
            assert not torch.equal(
                temporal[
                    batch_index,
                    part_start:part_end,
                    start:,
                ],
                intact[
                    batch_index,
                    part_start:part_end,
                    start:,
                ],
            )


def test_reset_prefix_drop_masks_first_future_token():
    codecs = (
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(1),
    )
    motions = (
        _motion(batch=1, features=8),
        _motion(batch=1, features=8) + 0.1,
        _motion(batch=1, features=8) + 0.2,
    )
    intact = torch.zeros(1, 5, 8, dtype=torch.long)
    target = torch.tensor([1])
    temporal, valid, boundaries = build_reset_future_teacher_inputs(
        intact,
        intact,
        target,
        codec_motion_inputs=motions,
        gesture_codecs=codecs,
        horizon_tokens=2,
        past_context_tokens=3,
        mask_token_id=31,
        motion_fps=25,
        reset_prefix_drop_tokens=1,
        lower_velocity_feature_slice=(1, 4),
    )
    start = boundaries[0].suffix_start_token
    assert (temporal[:, :, start] == 31).all()
    assert not valid[:, start].any()
    assert valid[:, start + 1:].any()


def test_cached_fixed_window_matches_online_construction():
    codecs = (
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(2),
        _DeterministicFreshCodec(1),
    )
    motions = (
        _motion(batch=2, features=8),
        _motion(batch=2, features=8) + 0.1,
        _motion(batch=2, features=8) + 0.2,
    )
    intact = torch.zeros(2, 5, 8, dtype=torch.long)
    targets = torch.tensor([1, 2])
    online, online_valid, _ = build_reset_future_teacher_inputs(
        intact,
        intact,
        targets,
        codec_motion_inputs=motions,
        gesture_codecs=codecs,
        horizon_tokens=2,
        past_context_tokens=3,
        mask_token_id=31,
        motion_fps=25,
        maximum_suffix_frames=4,
        lower_velocity_feature_slice=(1, 4),
    )

    reset_windows = []
    for sample_index, target in enumerate(targets.tolist()):
        start_frame = (target + 2) * 2
        parts = []
        for part_index, (codec, motion) in enumerate(
            zip(codecs, motions)
        ):
            parts.append(
                encode_reset_suffix(
                    codec,
                    motion[sample_index:sample_index + 1],
                    suffix_start_frame=start_frame,
                    maximum_suffix_frames=4,
                    zero_first_frame_feature_slice=(
                        (1, 4) if part_index == 1 else None
                    ),
                ).codes[0]
            )
        reset_windows.append(torch.cat(parts, dim=0))

    cached, cached_valid, _ = (
        build_reset_future_teacher_inputs_from_codes(
            intact,
            intact,
            targets,
            reset_code_windows=reset_windows,
            horizon_tokens=2,
            past_context_tokens=3,
            mask_token_id=31,
            motion_fps=25,
            frame_size=2,
        )
    )
    torch.testing.assert_close(cached, online)
    torch.testing.assert_close(cached_valid, online_valid)


if __name__ == "__main__":
    test_hidden_interval_and_past_invariance()
    test_future_sensitivity_and_cache_contamination()
    test_fixed_window_batch_matches_individual_resets()
    test_active_streaming_state_is_rejected()
    test_lower_boundary_velocity_is_internal_to_suffix()
    test_all_body_parts_are_reset_and_boundaries_align()
    test_reset_prefix_drop_masks_first_future_token()
    test_cached_fixed_window_matches_online_construction()
    print(
        "Reset-future smoke tests passed "
        "(invariance/sensitivity/batched reset/cache/boundary/all parts)."
    )
