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
    assert model.last_kinematic_input_codes.shape == codes.shape
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
    test_streaming_generator_outputs_all_20_canonical_slots()
    print("C2F smoke tests passed (forward/backward/CFG streaming).")
