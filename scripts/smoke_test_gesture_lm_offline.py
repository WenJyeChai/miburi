import torch
from torch import nn

from miburi.models.gesture_lm import GTemporalDepthModel3
from miburi.models.gesture_lm_offline import (
    GTemporalDepthModel3Offline,
    GestureLMOfflineGen,
    StaticMemoryCrossAttention,
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


def _model_kwargs(card: int = 16):
    return dict(
        n_q=20,
        card=card,
        dim=16,
        num_heads=4,
        num_layers=1,
        hidden_scale=2,
        query2mem_scale=1,
        num_temp_classifiers=1,
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
        depformer_pos_emb="rope",
        text_procemb=torch.randn(19, 16),
        audio_procemb=[
            torch.randn(17, 16),
            torch.randn(17, 16),
        ],
        gesture_codec_layers=nn.ModuleList(
            [_FakeCodecLayer(card=card, dim=8) for _ in range(20)]
        ),
        body_parts=3,
        bp_dist=None,
        positional_embedding="rope",
        gating="silu",
        depformer_gating="silu",
        dropout=0.0,
    )


def _make_model(card: int = 16):
    return GTemporalDepthModel3Offline(**_model_kwargs(card))


def _batch(card: int = 16):
    batch_size, steps = 2, 4
    codes = torch.randint(card, (batch_size, 20, steps))
    audio = torch.randint(17, (batch_size, 2, steps))
    text = torch.randint(19, (batch_size, 1, steps))
    speaker = torch.tensor([1, 2])
    return codes, audio, text, speaker


def _temporal_inputs(model, codes):
    initial = model._get_initial_token().expand(
        codes.shape[0],
        model.n_q,
        -1,
    )
    return torch.cat([initial, codes], dim=-1)[:, :, :-1]


def test_only_temporal_conditioning_is_noncausal():
    model = _make_model()
    assert model.causal
    for layer in model.temporal_transformer.layers:
        assert layer.self_attn.causal
        assert all(
            isinstance(attention, StaticMemoryCrossAttention)
            for attention in layer.cross_attns
        )
        assert all(
            not attention.causal for attention in layer.cross_attns
        )
    for layer in model.depth_transformer.layers:
        assert layer.self_attn.causal


def test_parameter_count_matches_original_miburi():
    torch.manual_seed(2)
    original = GTemporalDepthModel3(**_model_kwargs())
    offline = _make_model()
    original_count = sum(p.numel() for p in original.parameters())
    offline_count = sum(p.numel() for p in offline.parameters())
    assert original_count == offline_count


def test_future_gesture_is_hidden_but_future_condition_is_visible():
    torch.manual_seed(3)
    model = _make_model().eval()
    codes, audio, text, speaker = _batch()
    temporal_input = _temporal_inputs(model, codes)
    audio_condition, text_condition = model.process_conditions(audio, text)
    audio_condition = audio_condition.squeeze(1)
    text_condition = text_condition.squeeze(1)
    temporal_speaker = speaker[:, None].expand(-1, codes.shape[-1])

    _, reference_logits = model.forward_temporal(
        temporal_input,
        audio_condition,
        text_condition,
        temporal_speaker,
    )

    changed_future_gesture = temporal_input.clone()
    changed_future_gesture[:, :, -1] = (
        changed_future_gesture[:, :, -1] + 1
    ) % model.card
    _, future_gesture_logits = model.forward_temporal(
        changed_future_gesture,
        audio_condition,
        text_condition,
        temporal_speaker,
    )
    torch.testing.assert_close(
        reference_logits[:, :, :-1],
        future_gesture_logits[:, :, :-1],
    )

    changed_future_audio = audio_condition.clone()
    changed_future_audio[:, -1] += 10.0
    _, future_audio_logits = model.forward_temporal(
        temporal_input,
        changed_future_audio,
        text_condition,
        temporal_speaker,
    )
    assert not torch.allclose(
        reference_logits[:, :, 0],
        future_audio_logits[:, :, 0],
    )


def test_forward_backward_keeps_original_20_token_layout():
    torch.manual_seed(4)
    model = _make_model().train()
    codes, audio, text, speaker = _batch()
    logits = model(
        codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
    )
    assert logits.shape == (2, 20, 4, 17)
    logits.square().mean().backward()
    assert model.temporal_classifier.weight.grad is not None
    assert all(
        classifier.weight.grad is not None
        for classifier in model.depformer_classifier
    )


def test_offline_generator_uses_one_static_full_memory():
    torch.manual_seed(5)
    model = _make_model().eval()
    _, audio, text, speaker = _batch()
    audio = audio[:1]
    text = text[:1]
    speaker = speaker[:1]
    full_condition = torch.cat([text, audio], dim=1)
    generator = GestureLMOfflineGen(
        model,
        use_sampling=False,
        cfg_coef=1.0,
        condition_tensors=speaker[:, None],
        check=True,
    )
    generator.set_full_condition(full_condition)

    generated = []
    with generator.streaming(1):
        for _ in range(full_condition.shape[-1]):
            output = generator.step()
            generated.append(output)
        for layer in model.temporal_transformer.layers:
            for attention in layer.cross_attns:
                state = attention._streaming_state
                assert state.memory_length == full_condition.shape[-1]
                assert state.query_offset.item() == full_condition.shape[-1]

    generated = torch.cat(generated, dim=-1)
    assert generated.shape == (1, 20, full_condition.shape[-1])
    assert (generated >= 0).all()
    assert (generated < model.card).all()


def test_offline_generator_supports_classifier_free_guidance():
    torch.manual_seed(6)
    model = _make_model().eval()
    _, audio, text, speaker = _batch()
    full_condition = torch.cat([text[:1], audio[:1]], dim=1)
    generator = GestureLMOfflineGen(
        model,
        use_sampling=False,
        cfg_coef=2.0,
        condition_tensors=speaker[:1, None],
        check=True,
    )
    generator.set_full_condition(full_condition)
    with generator.streaming(1):
        output = generator.step()
        for layer in model.temporal_transformer.layers:
            for attention in layer.cross_attns:
                state = attention._streaming_state
                assert state.batch_size == 2
                assert state.memory_length == full_condition.shape[-1]
    assert output.shape == (1, 20, 1)
    assert (output >= 0).all()
    assert (output < model.card).all()


if __name__ == "__main__":
    test_only_temporal_conditioning_is_noncausal()
    test_parameter_count_matches_original_miburi()
    test_future_gesture_is_hidden_but_future_condition_is_visible()
    test_forward_backward_keeps_original_20_token_layout()
    test_offline_generator_uses_one_static_full_memory()
    test_offline_generator_supports_classifier_free_guidance()
    print(
        "Offline MIBURI smoke tests passed "
        "(mask separation/full memory/forward/backward/generation)."
    )
