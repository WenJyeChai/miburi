"""CPU smoke checks for Condition C; no checkpoints or datasets required.

Run with ``python -B -m scripts.smoke_test_gesture_lm_condition_c``.
"""

from unittest.mock import patch

import torch
import torch.nn.functional as F

from miburi.models.gesture_lm import GTemporalDepthModel3
from miburi.models.gesture_lm_condition_c import (
    ConditionCMemoryCrossAttention,
    GTemporalDepthModel3ConditionC,
    build_condition_c_attention_bias,
    forward_condition_c_teacher_view,
)
from miburi.models.gesture_lm_offline import (
    GTemporalDepthModel3Offline,
    GestureLMOfflineGen,
    StaticMemoryCrossAttention,
)
from miburi.modules.rope import RotaryEmbedding
from miburi.modules.transformer import StreamingMultiheadCrossAttention
from miburi.utils.compile import no_compile
from scripts.smoke_test_gesture_lm_offline import (
    _model_kwargs,
    _temporal_inputs,
)


def _kwargs(scale=1, checkpointing=False):
    kwargs = _model_kwargs()
    kwargs.update(
        num_layers=2,
        context=3,
        memory_context=3 * scale,
        query2mem_scale=scale,
        gradient_checkpointing=checkpointing,
    )
    return kwargs


def _batch(scale=1, batch=2, steps=7):
    return (
        torch.randint(16, (batch, 20, steps)),
        torch.randint(17, (batch, 2, steps * scale)),
        torch.randint(19, (batch, 1, steps * scale)),
        torch.arange(1, batch + 1),
    )


def _teacher_kwargs(batch):
    codes, audio, text, speaker = batch
    return dict(
        input_codes=codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
    )


def _model_forward(model, batch, **kwargs):
    codes, audio, text, speaker = batch
    return model(
        codes,
        audio_codes=audio,
        text_codes=text,
        sum_condition=speaker,
        **kwargs,
    )


def test_mask_matches_causal_boundary_and_admits_all_future():
    # Capture the actual base attention mask instead of only comparing two
    # locally implemented formulas. Include floor-division boundary cases.
    for scale, query_scale, context in (
        (1, 1, 3), (2, 1, 5), (3, 2, 7), (2, 2, 1), (2, 1, None),
    ):
        query = torch.randn(2, 8, 16)
        memory = torch.randn(2, 12, 16)
        source = StreamingMultiheadCrossAttention(
            embed_dim=16,
            num_heads=4,
            causal=True,
            context=context,
            upsample_factor=scale,
            query_upsample_factor=query_scale,
        )
        captured = []
        sdpa = F.scaled_dot_product_attention

        def capture(q, k, v, attn_mask=None, **kwargs):
            captured.append(attn_mask.detach().clone())
            return sdpa(q, k, v, attn_mask, **kwargs)

        with patch.object(F, "scaled_dot_product_attention", side_effect=capture):
            source(query, memory, memory)
        causal_mask = captured[0]
        c_mask = build_condition_c_attention_bias(
            8, 12, context=context, device="cpu",
            upsample_factor=scale, query_upsample_factor=query_scale,
        )
        q = torch.arange(8).view(-1, 1) // query_scale
        k = torch.arange(12).view(1, -1) // scale
        future = (k > q)[None, None]
        torch.testing.assert_close(c_mask.expand_as(causal_mask), causal_mask | future)
        assert bool(c_mask[future].all())
        torch.testing.assert_close(c_mask & ~future, causal_mask[:1])


def test_mask_uses_absolute_batched_offsets_and_original_past_window():
    for context in (None, 0, 1, 5, 8):
        mask = build_condition_c_attention_bias(
            3, 12, context=context, device="cpu", upsample_factor=2,
            query_upsample_factor=2, query_offset=torch.tensor([0, 7]),
            key_offset=torch.tensor([-2, 2]),
        )
        for batch, (q_start, k_start) in enumerate(((0, -2), (7, 2))):
            for q in range(3):
                for k in range(12):
                    aligned_key = (k_start + k) // 2
                    delta = (q_start + q) // 2 - aligned_key
                    past = delta >= 0 and (context is None or delta < context // 2)
                    expected = aligned_key >= 0 and (past or delta < 0)
                    assert bool(mask[batch, 0, q, k]) == expected


def test_direct_attention_hides_old_memory_and_sees_future_with_padding():
    torch.manual_seed(22)
    attention = ConditionCMemoryCrossAttention(
        embed_dim=16, num_heads=4, causal=False, context=2,
    ).eval()
    query, memory = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
    reference = attention(query, memory, memory)
    changed = memory.clone()
    changed[:, 0] += torch.randn_like(changed[:, 0]) * 10
    old_changed = attention(query, changed, changed)
    torch.testing.assert_close(reference[:, 5], old_changed[:, 5])
    changed = memory.clone()
    changed[:, -1] += torch.randn_like(changed[:, -1]) * 10
    future_changed = attention(query, changed, changed)
    assert not torch.allclose(reference[:, 0], future_changed[:, 0])

    key_padding = torch.zeros(2, 7, dtype=torch.bool)
    key_padding[:, -1] = True
    query_padding = torch.zeros_like(key_padding)
    query_padding[:, 1] = True
    extra = torch.ones(1, 1, 7, 7, dtype=torch.bool)
    extra[..., 0] = False
    masked = attention(
        query, memory, memory, query_padding_mask=query_padding,
        key_padding_mask=key_padding, extra_attn_bias=extra,
    )
    masked_changed = attention(
        query, changed, changed, query_padding_mask=query_padding,
        key_padding_mask=key_padding, extra_attn_bias=extra,
    )
    torch.testing.assert_close(masked, masked_changed)
    torch.testing.assert_close(masked[:, 1], torch.zeros_like(masked[:, 1]))


def test_parameter_layout_and_motion_causality_are_unchanged():
    torch.manual_seed(23)
    teacher = GTemporalDepthModel3ConditionC(**_kwargs()).eval()
    offline = GTemporalDepthModel3Offline(**_kwargs()).eval()
    offline.load_state_dict(teacher.state_dict(), strict=True)
    assert {
        name: parameter.shape for name, parameter in teacher.named_parameters()
    } == {
        name: parameter.shape for name, parameter in offline.named_parameters()
    }
    for layer in teacher.temporal_transformer.layers:
        assert layer.self_attn.causal
        assert all(isinstance(a, ConditionCMemoryCrossAttention) for a in layer.cross_attns)
    assert all(layer.self_attn.causal for layer in teacher.depth_transformer.layers)

    batch = _batch()
    reference = _model_forward(teacher, batch)
    changed_codes = batch[0].clone()
    changed_codes[:, :, 3:] = (changed_codes[:, :, 3:] + 1) % teacher.card
    changed = _model_forward(teacher, (changed_codes, *batch[1:]))
    torch.testing.assert_close(reference[:, :, :3], changed[:, :, :3])
    # q0 at frame 3 also cannot see frame 3's target/prefix.
    torch.testing.assert_close(reference[:, :1, :4], changed[:, :1, :4])
    assert reference.shape == (2, 20, 7, 17)


def test_shared_teacher_matches_supervised_teacher_for_padded_rvq_prefixes():
    for scale in (1, 2):
        torch.manual_seed(24 + scale)
        student = GTemporalDepthModel3(**_kwargs(scale)).eval()
        teacher = GTemporalDepthModel3ConditionC(**_kwargs(scale)).eval()
        teacher.load_state_dict(student.state_dict(), strict=True)
        # Different per-layer/per-memory contexts must survive the view.
        for layer_index, (s_layer, t_layer) in enumerate(zip(
            student.temporal_transformer.layers, teacher.temporal_transformer.layers,
        )):
            for memory_index, (s_attn, t_attn) in enumerate(zip(
                s_layer.cross_attns, t_layer.cross_attns,
            )):
                s_attn.context = t_attn.context = (2 + layer_index + memory_index) * scale
        batch = _batch(scale)
        codes = batch[0]
        codes[0, 8:16] = student.pad_token_id
        codes[1, :, -1] = student.pad_token_id
        stochastic_prefix = torch.where(
            codes == student.pad_token_id, codes,
            (codes + torch.randint(1, 5, codes.shape)) % student.card,
        )
        lower_mask = torch.zeros_like(codes, dtype=torch.bool)
        lower_mask[:, 8:16] = True
        kwargs = dict(
            depth_input_codes=stochastic_prefix,
            ca_depth_padding_mask=lower_mask,
        )
        temporal, depth = forward_condition_c_teacher_view(
            student, **_teacher_kwargs(batch), **kwargs,
        )
        expected = _model_forward(teacher, batch, **kwargs)
        torch.testing.assert_close(torch.cat([temporal, depth], dim=1), expected)
        assert not temporal.requires_grad and not depth.requires_grad
        q0, absent_depth = forward_condition_c_teacher_view(
            student, **_teacher_kwargs(batch), include_depth_levels=False,
        )
        torch.testing.assert_close(q0, temporal)
        assert absent_depth is None


def test_teacher_ce_is_gradient_enabled_and_shared_view_is_detached():
    torch.manual_seed(27)
    model = GTemporalDepthModel3ConditionC(**_kwargs()).train()
    batch = _batch()
    logits = _model_forward(model, batch)
    logits.square().mean().backward()
    assert model.temporal_classifier.weight.grad.abs().sum() > 0
    assert all(head.weight.grad.abs().sum() > 0 for head in model.depformer_classifier)
    assert model.temporal_transformer.layers[0].cross_attns[0].key_projs[0].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    temporal, depth = forward_condition_c_teacher_view(model, **_teacher_kwargs(batch))
    assert not temporal.requires_grad and not depth.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training


def test_helper_restores_modules_flags_and_streaming_state_on_error():
    torch.manual_seed(28)
    student = GTemporalDepthModel3(**_kwargs()).eval()
    batch = _batch()
    before_modules = [layer.cross_attns for layer in student.temporal_transformer.layers]
    before_parameters = list(student.parameters())
    source_attention = before_modules[0][0]
    view = ConditionCMemoryCrossAttention.shared_view(source_attention)
    assert view.in_projs is source_attention.in_projs
    assert view.causal is False and source_attention.causal is True

    with student.streaming(2):
        audio_condition, text_condition = student.process_conditions(batch[1], batch[2])
        student.forward_temporal(
            _temporal_inputs(student, batch[0])[:, :, :1],
            audio_condition.squeeze(1)[:, :1],
            text_condition.squeeze(1)[:, :1],
            batch[3][:, None],
        )
        before_states = student.get_streaming_state()
        cross_state = source_attention._streaming_state
        old_offset = cross_state.offsetq.clone()
        old_cache = cross_state.kv_cache.cache.clone()
        forward_condition_c_teacher_view(student, **_teacher_kwargs(batch))
        with patch.object(student, "forward_temporal", side_effect=RuntimeError("injected")):
            try:
                forward_condition_c_teacher_view(student, **_teacher_kwargs(batch))
            except RuntimeError as exc:
                assert str(exc) == "injected"
            else:
                raise AssertionError("Expected injected failure.")
        assert all(student.get_streaming_state()[key] is state for key, state in before_states.items())
        torch.testing.assert_close(cross_state.offsetq, old_offset)
        torch.testing.assert_close(cross_state.kv_cache.cache, old_cache)
    assert all(layer.cross_attns is before for layer, before in zip(
        student.temporal_transformer.layers, before_modules,
    ))
    assert all(current is before for current, before in zip(student.parameters(), before_parameters))
    assert all(a.causal for layer in student.temporal_transformer.layers for a in layer.cross_attns)


def test_checkpointed_student_backward_survives_interleaved_teacher_view():
    torch.manual_seed(29)
    reference = GTemporalDepthModel3(**_kwargs(checkpointing=True)).train()
    interleaved = GTemporalDepthModel3(**_kwargs(checkpointing=True)).train()
    interleaved.load_state_dict(reference.state_dict())
    batch = _batch()
    _model_forward(reference, batch).square().mean().backward()
    logits = _model_forward(interleaved, batch)
    forward_condition_c_teacher_view(interleaved, **_teacher_kwargs(batch))
    logits.square().mean().backward()
    for (name, ref), (other_name, other) in zip(
        reference.named_parameters(), interleaved.named_parameters(),
    ):
        assert name == other_name
        if ref.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(ref.grad, other.grad)

    # The supervised C branch's own recomputation also retains its mask.
    c_reference = GTemporalDepthModel3ConditionC(**_kwargs()).train()
    c_checkpointed = GTemporalDepthModel3ConditionC(**_kwargs(checkpointing=True)).train()
    c_checkpointed.load_state_dict(c_reference.state_dict())
    _model_forward(c_reference, batch).square().mean().backward()
    _model_forward(c_checkpointed, batch).square().mean().backward()
    for ref, other in zip(c_reference.parameters(), c_checkpointed.parameters()):
        if ref.grad is not None:
            torch.testing.assert_close(ref.grad, other.grad)


def test_static_streaming_attention_matches_batch_with_rope_and_padding():
    torch.manual_seed(30)
    attention = ConditionCMemoryCrossAttention(
        embed_dim=16, num_heads=4, causal=False, context=5,
        upsample_factor=2, rope=RotaryEmbedding(),
    ).eval()
    queries, memory = torch.randn(2, 7, 16), torch.randn(2, 14, 16)
    key_padding = torch.zeros(2, 14, dtype=torch.bool)
    key_padding[0, -2:] = True
    query_padding = torch.zeros(2, 7, dtype=torch.bool)
    query_padding[1, 2] = True
    reference = attention(
        queries, memory, memory, query_padding_mask=query_padding,
        key_padding_mask=key_padding,
    )
    outputs = []
    with attention.streaming(2):
        initial_keys = None
        for t in range(7):
            outputs.append(attention(
                queries[:, t:t + 1], memory, memory,
                query_padding_mask=query_padding[:, t:t + 1],
                key_padding_mask=key_padding,
            ))
            state = attention._streaming_state
            if initial_keys is None:
                initial_keys = state.projected_keys
            assert state.projected_keys is initial_keys
        torch.testing.assert_close(state.query_offset, torch.full((2,), 7))
    torch.testing.assert_close(torch.cat(outputs, dim=1), reference, atol=1e-6, rtol=1e-5)


def test_temporal_streaming_matches_complete_sequence_beyond_past_window():
    for scale in (1, 2):
        torch.manual_seed(31 + scale)
        model = GTemporalDepthModel3ConditionC(**_kwargs(scale)).eval()
        codes, audio, text, speaker = _batch(scale)
        sequence = _temporal_inputs(model, codes)
        a, t = model.process_conditions(audio, text)
        a, t = a.squeeze(1), t.squeeze(1)
        reference_hidden, reference_logits = model.forward_temporal(
            sequence, a, t, speaker[:, None].expand(-1, codes.shape[-1]),
        )
        audio_memory, text_memory = model.project_temporal_conditions(a, t)
        hidden, logits = [], []
        with model.streaming(2):
            for step in range(codes.shape[-1]):
                h, out = model.forward_temporal_projected(
                    sequence[:, :, step:step + 1], audio_memory, text_memory,
                    speaker[:, None],
                )
                hidden.append(h)
                logits.append(out)
        torch.testing.assert_close(torch.cat(hidden, dim=1), reference_hidden, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(torch.cat(logits, dim=2), reference_logits, atol=2e-6, rtol=2e-5)


def test_generator_reuses_full_memory_and_matches_greedy_q0_with_cfg():
    for scale, cfg, per_step_weights in ((1, 1.0, True), (1, 2.0, True), (2, 2.0, False)):
        torch.manual_seed(34 + scale)
        kwargs = _kwargs(scale)
        # The released depth path's per-step K/V schedule assumes scale=1.
        # Exercise scaled C generation with its supported shared-weight path.
        kwargs["depformer_weights_per_step"] = per_step_weights
        model = GTemporalDepthModel3ConditionC(**kwargs).eval()
        _, audio, text, speaker = _batch(scale, batch=1, steps=5)
        generator = GestureLMOfflineGen(
            model, use_sampling=False, cfg_coef=cfg,
            condition_tensors=speaker[:, None], check=True,
        )
        generator.set_full_condition(torch.cat([text, audio], dim=1))
        generated = []
        with generator.streaming(1):
            for _ in range(5):
                generated.append(generator.step())
            for layer in model.temporal_transformer.layers:
                for attention in layer.cross_attns:
                    state = attention._streaming_state
                    assert state.memory_length == 5 * scale
                    assert state.batch_size == (1 if cfg == 1 else 2)
                    assert bool((state.query_offset == 5).all())
        generated = torch.cat(generated, dim=-1)
        logits = _model_forward(model, (generated, audio, text, speaker))
        if cfg != 1:
            null_logits = _model_forward(model, (
                generated, torch.full_like(audio, -1),
                torch.full_like(text, -1), speaker,
            ))
            logits = null_logits + cfg * (logits - null_logits)
        torch.testing.assert_close(
            logits[:, :1, :, :model.card].argmax(dim=-1), generated[:, :1],
        )
        assert generated.shape == (1, 20, 5)
        assert bool(((generated >= 0) & (generated < model.card)).all())


def test_complete_generation_matches_offline_when_past_masks_coincide():
    # Scope regression: Condition C leaves the released depth generator
    # untouched, even though that generator's depth logits do not generally
    # equal batched teacher-forced depth logits (its existing KV behavior).
    for cfg in (1.0, 2.0):
        torch.manual_seed(38)
        kwargs = _kwargs()
        kwargs["memory_context"] = 10
        teacher = GTemporalDepthModel3ConditionC(**kwargs).eval()
        offline = GTemporalDepthModel3Offline(**kwargs).eval()
        offline.load_state_dict(teacher.state_dict())
        _, audio, text, speaker = _batch(batch=1, steps=5)
        results = []
        for model in (teacher, offline):
            generator = GestureLMOfflineGen(
                model, use_sampling=False, cfg_coef=cfg,
                condition_tensors=speaker[:, None], check=True,
            )
            generator.set_full_condition(torch.cat([text, audio], dim=1))
            with generator.streaming(1):
                results.append(torch.cat([generator.step() for _ in range(5)], dim=-1))
        torch.testing.assert_close(results[0], results[1])


if __name__ == "__main__":
    torch.set_num_threads(1)
    tests = [value for name, value in globals().copy().items() if name.startswith("test_")]
    with no_compile():
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
    print(f"Condition C model smoke tests passed ({len(tests)} checks).")
