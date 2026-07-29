import torch
from torch import nn

from miburi.models.gesture_lm_c2f import (
    COARSE_SLOTS,
    KINEMATIC_SLOTS,
    GTemporalDepthModel3C2F,
    GestureLMC2FGen,
)


class _FakeCodebook(nn.Module):
    def __init__(self, card: int, dim: int):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Parameter(torch.randn(card, dim))


class _FakeCodecLayer(nn.Module):
    def __init__(self, card: int, dim: int):
        super().__init__()
        self._codebook = _FakeCodebook(card, dim)


def _make_model(card: int = 16) -> GTemporalDepthModel3C2F:
    codebooks = nn.ModuleList(
        [_FakeCodecLayer(card=card, dim=8) for _ in range(20)]
    )
    return GTemporalDepthModel3C2F(
        n_q=20,
        card=card,
        dim=16,
        num_heads=4,
        num_layers=1,
        hidden_scale=2,
        query2mem_scale=2,
        num_temp_classifiers=3,
        norm="layer_norm",
        cond_dim=16,
        context=4,
        memory_context=4,
        causal=True,
        depformer_heads=4,
        depformer_layers=1,
        depformer_dim=16,
        depformer_dim_feedforward=32,
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_pos_emb="none",
        text_procemb=torch.randn(19, 16),
        audio_procemb=[torch.randn(17, 16), torch.randn(17, 16)],
        gesture_codec_layers=codebooks,
        body_parts=3,
        bp_dist=None,
        positional_embedding="none",
        gating="silu",
        depformer_gating="silu",
        dropout=0.0,
    )


def _batch(card: int = 16):
    B, T = 2, 3
    codes = torch.randint(card, (B, 20, T))
    # Simulate a source without lower/face targets in one frame.
    codes[1, 8:, -1] = card
    audio = torch.randint(17, (B, 2, T * 2))
    text = torch.randint(19, (B, 1, T * 2))
    speaker = torch.tensor([1, 2])
    return codes, audio, text, speaker


def test_slot_schedule_is_complete_and_global_c2f():
    assert COARSE_SLOTS == (0, 8, 16)
    assert KINEMATIC_SLOTS[:9] == (1, 9, 17, 2, 10, 18, 3, 11, 19)
    assert sorted(COARSE_SLOTS + KINEMATIC_SLOTS) == list(range(20))


def test_teacher_forced_and_self_forced_forward_keep_canonical_layout():
    torch.manual_seed(7)
    model = _make_model()
    codes, audio, text, speaker = _batch()
    model.train()

    teacher_logits = model(
        codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
        self_force_kinematic=False,
    )
    assert teacher_logits.shape == (2, 20, 3, 17)
    assert model.last_kinematic_input_codes is None

    masked_inputs = codes.clone()
    masked_inputs[0, 8:16] = model.pad_token_id
    self_forced_logits = model(
        masked_inputs,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
        self_force_kinematic=True,
        kinematic_target_codes=codes,
    )
    assert self_forced_logits.shape == (2, 20, 3, 17)
    assert model.temporal_transformer.training
    assert model.depth_transformer.training
    assert model.last_kinematic_input_codes.shape == codes.shape
    assert model.last_temporal_rollout_codes.shape == (2, 3, 3)
    assert model.last_temporal_input_codes.shape == (2, 3, 3)
    assert (
        model.last_temporal_input_codes[:, :, 0]
        == model.initial_token_id
    ).all()
    assert torch.equal(
        model.last_temporal_input_codes[:, :, 1:],
        model.last_temporal_rollout_codes[:, :, :-1],
    )
    assert torch.equal(
        model.last_kinematic_input_codes[:, COARSE_SLOTS],
        model.last_temporal_rollout_codes,
    )
    assert (
        model.last_kinematic_input_codes[1, 8:, -1] == model.pad_token_id
    ).all()
    assert (
        model.last_kinematic_input_codes[0, 8:16] != model.pad_token_id
    ).all()

    recovery = model.soft_recovery_loss(
        self_forced_logits,
        codes,
        model.last_kinematic_input_codes,
        topk=4,
    )
    assert recovery.ndim == 0
    assert torch.isfinite(recovery)
    (self_forced_logits.square().mean() + 0.1 * recovery).backward()
    assert all(head.weight.grad is not None for head in model.temporal_classifier)
    assert all(
        head.weight.grad is not None for head in model.depformer_classifier
    )


def test_temporal_backbone_ignores_fine_token_history():
    torch.manual_seed(8)
    model = _make_model().eval()
    codes, audio, text, speaker = _batch()
    alternate = codes.clone()
    fine_slots = [
        slot for slot in range(model.n_q) if slot not in COARSE_SLOTS
    ]
    alternate[:, fine_slots] = torch.randint(
        model.card, alternate[:, fine_slots].shape
    )
    audio_condition, text_condition = model.process_conditions(audio, text)
    audio_condition = audio_condition.squeeze(1)
    text_condition = text_condition.squeeze(1)
    temporal_speaker = speaker.unsqueeze(1).expand(-1, codes.shape[-1])
    hidden, logits = model.forward_temporal(
        codes,
        audio_condition,
        text_condition,
        temporal_speaker,
    )
    alternate_hidden, alternate_logits = model.forward_temporal(
        alternate,
        audio_condition,
        text_condition,
        temporal_speaker,
    )
    assert torch.equal(hidden, alternate_hidden)
    assert torch.equal(logits, alternate_logits)
    for slot in KINEMATIC_SLOTS:
        assert not any(
            parameter.requires_grad
            for parameter in model.temporal_gemb[slot].parameters()
        )
        assert not any(
            parameter.requires_grad
            for parameter in model.temporal_gproj[slot].parameters()
        )


def test_self_forced_q0_never_uses_pad_class():
    torch.manual_seed(9)
    model = _make_model().eval()
    codes, audio, text, speaker = _batch()
    B, _, T = codes.shape
    audio_condition, text_condition = model.process_conditions(audio, text)
    audio_condition = audio_condition.squeeze(1)
    text_condition = text_condition.squeeze(1)
    temporal_speaker = speaker.unsqueeze(1).expand(-1, T)

    # Make PAD the unambiguous winner of the unmasked temporal classifier.
    # The self-forcing rollout must still select only real codec IDs.
    temporal_logits = torch.zeros(B, 3, T, model.card + 1)
    temporal_logits[..., model.pad_token_id] = 100.0
    generated = model._greedy_self_forced_codes(
        codes,
        transformer_out=torch.randn(B, T, model.dim),
        temporal_logits=temporal_logits,
        audio_condition=audio_condition,
        text_condition=text_condition,
        sum_condition=temporal_speaker,
        ca_query_padding_mask=None,
    )
    valid_q0 = codes[:, COARSE_SLOTS] != model.pad_token_id
    assert (generated[:, COARSE_SLOTS][valid_q0] < model.card).all()


def test_greedy_cfg1_generator_matches_training_rollout():
    torch.manual_seed(10)
    model = _make_model().eval()
    codes, audio, text, speaker = _batch()
    codes = codes[:1]
    audio = audio[:1]
    text = text[:1]
    speaker = speaker[:1]
    _, _, T = codes.shape

    audio_condition, text_condition = model.process_conditions(audio, text)
    audio_condition = audio_condition.squeeze(1)
    text_condition = text_condition.squeeze(1)
    temporal_speaker = speaker.unsqueeze(1).expand(-1, T)
    coarse, transformer_out = (
        model._greedy_temporal_self_forced_coarse_codes(
            codes,
            audio_condition,
            text_condition,
            temporal_speaker,
        )
    )
    temporal_input = torch.full_like(codes, model.initial_token_id)
    if T > 1:
        temporal_input[:, COARSE_SLOTS, 1:] = coarse[:, :, :-1]
    _, temporal_logits = model.forward_temporal(
        temporal_input,
        audio_condition,
        text_condition,
        temporal_speaker,
    )
    training_rollout = model._greedy_self_forced_codes(
        codes,
        transformer_out,
        temporal_logits,
        audio_condition,
        text_condition,
        temporal_speaker,
        ca_query_padding_mask=None,
        coarse_codes=coarse,
    )

    generator = GestureLMC2FGen(
        model,
        use_sampling=False,
        cfg_coef=1.0,
        condition_tensors=speaker.unsqueeze(1),
    )
    generated_frames = []
    with generator.streaming(1):
        for time_index in range(T):
            memory_start = time_index * model.query2mem_scale
            memory_end = memory_start + model.query2mem_scale
            condition = torch.cat(
                [
                    text[:, :, memory_start:memory_end],
                    audio[:, :, memory_start:memory_end],
                ],
                dim=1,
            )
            generated_frames.append(generator.step(condition).clone())
    evaluation_rollout = torch.cat(generated_frames, dim=-1)
    assert torch.equal(training_rollout, evaluation_rollout), (
        "Greedy CFG=1 evaluation diverged from training rollout at "
        f"{(training_rollout != evaluation_rollout).nonzero().tolist()}; "
        f"training={training_rollout.tolist()}, "
        f"evaluation={evaluation_rollout.tolist()}"
    )


def test_streaming_generator_outputs_all_20_canonical_slots():
    torch.manual_seed(11)
    model = _make_model().eval()
    generator = GestureLMC2FGen(
        model,
        use_sampling=False,
        cfg_coef=2.0,
        condition_tensors=torch.tensor([[1]]),
    )
    # One gesture frame consumes query2mem_scale=2 text/audio frames.
    condition = torch.cat(
        [
            torch.randint(19, (1, 1, 2)),
            torch.randint(17, (1, 2, 2)),
        ],
        dim=1,
    )
    with generator.streaming(1):
        output = generator.step(condition)
    assert output.shape == (1, 20, 1)
    assert (output >= 0).all()
    assert (output < model.card).all()


if __name__ == "__main__":
    test_slot_schedule_is_complete_and_global_c2f()
    test_teacher_forced_and_self_forced_forward_keep_canonical_layout()
    test_temporal_backbone_ignores_fine_token_history()
    test_self_forced_q0_never_uses_pad_class()
    test_greedy_cfg1_generator_matches_training_rollout()
    test_streaming_generator_outputs_all_20_canonical_slots()
    print("C2F smoke tests passed (forward/backward/CFG streaming).")
