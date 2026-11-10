"""Condition C: full future speech with the causal past window preserved.

Gesture self-attention and the depth transformer's aligned speech slices
are unchanged. Only temporal cross-attention gains future speech keys.
Unlike unrestricted offline/GlobalRegret attention, old speech keys do not
become directly visible merely because future conditioning is enabled.

The same explicit mask is used for complete-sequence training, static-memory
autoregressive generation, and the detached shared-weight teacher view.
No parameters or persistent state-dict entries are added.
"""

from contextlib import contextmanager
import copy

import torch

from ..modules.streaming import StreamingModule
from ..modules.transformer import StreamingMultiheadCrossAttention
from .gesture_lm_offline import (
    GTemporalDepthModel3Offline,
    StaticMemoryCrossAttention,
)
from .gesture_lm_shared_regret import (
    _forward_depth_branch,
    _prepend_initial_and_shift,
    _process_conditions_squeezed,
)


def build_condition_c_attention_bias(
    query_length: int,
    memory_length: int,
    *,
    context: int | None,
    device: torch.device | str,
    upsample_factor: int = 1,
    query_upsample_factor: int = 1,
    query_offset: int | torch.Tensor = 0,
    key_offset: int | torch.Tensor = 0,
) -> torch.Tensor:
    """Return a boolean ``[B or 1, 1, Tq, Tk]`` mask (True == visible).

    Coordinates exactly match ``StreamingMultiheadCrossAttention``:
    ``delta = floor(query / query_upsample_factor)
    - floor(key / upsample_factor)``. Its causal past is retained with the
    same strict ``delta < context // upsample_factor`` boundary. Every
    available future key (``delta < 0``) is additionally admitted. Offsets
    refer to absolute token coordinates; static full speech memory starts
    at key offset zero even while streaming gesture queries advance.

    The explicit union also handles contexts shorter than one aligned
    condition group: the original empty past remains empty, while future
    groups are still available. ``context=None`` admits the entire prefix.
    """

    if query_length <= 0 or memory_length <= 0:
        raise ValueError("Condition C requires nonempty queries and memory.")
    if upsample_factor < 1 or query_upsample_factor < 1:
        raise ValueError("Attention upsample factors must be positive.")
    if memory_length % upsample_factor:
        raise ValueError(
            "Condition memory length must be divisible by upsample_factor: "
            f"{memory_length} vs {upsample_factor}."
        )

    q_offset = torch.as_tensor(
        query_offset, dtype=torch.long, device=device,
    ).reshape(-1, 1, 1)
    k_offset = torch.as_tensor(
        key_offset, dtype=torch.long, device=device,
    ).reshape(-1, 1, 1)
    if (
        q_offset.shape[0] != k_offset.shape[0]
        and q_offset.shape[0] != 1
        and k_offset.shape[0] != 1
    ):
        raise ValueError("Query/key offset batches must match or broadcast.")

    pos_q = q_offset + torch.arange(
        query_length, device=device,
    ).view(1, -1, 1)
    pos_k = k_offset + torch.arange(
        memory_length, device=device,
    ).view(1, 1, -1)
    pos_q = pos_q // query_upsample_factor
    pos_k = pos_k // upsample_factor
    delta = pos_q - pos_k
    causal_past = delta >= 0
    if context is not None:
        causal_past = causal_past & (delta < context // upsample_factor)
    return ((pos_k >= 0) & (causal_past | (delta < 0))).unsqueeze(1)


class ConditionCMemoryCrossAttention(StaticMemoryCrossAttention):
    """Static full speech memory with an immutable Condition C policy.

    ``causal=False`` disables the parent's upper cutoff, but an explicit
    per-call mask preserves its original lower past boundary. The policy is
    part of this module's forward, so checkpoint recomputation sees exactly
    the same mask without depending on temporarily toggled causal flags.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_padding_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        extra_attn_bias: torch.Tensor | None = None,
    ):
        state = self._streaming_state
        query_offset = 0 if state is None else state.query_offset
        attention_bias = build_condition_c_attention_bias(
            query.shape[1],
            key.shape[1],
            context=self.context,
            device=query.device,
            upsample_factor=self.upsample_factor,
            query_upsample_factor=self.query_upsample_factor,
            query_offset=query_offset,
        )
        if extra_attn_bias is not None:
            attention_bias = attention_bias & extra_attn_bias
        return super().forward(
            query,
            key,
            value,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            extra_attn_bias=attention_bias,
        )

    @classmethod
    def shared_view(cls, source: StreamingMultiheadCrossAttention):
        """Create a temporary module view without copying any parameters.

        A shallow module copy retains the exact projections, hooks, dtype,
        device, and train/eval state. Its registries and streaming pointer
        are independent; only this view's class/policy changes. The source
        attention and its parameters remain the student's original objects.
        """

        view = copy.copy(source)
        view.__class__ = cls
        view._parameters = source._parameters.copy()
        view._buffers = source._buffers.copy()
        view._modules = source._modules.copy()
        view.causal = False
        view._streaming_state = None
        return view


class GTemporalDepthModel3ConditionC(GTemporalDepthModel3Offline):
    """Gradient-enabled teacher with C visibility and Offline's layout.

    Inherits full forward/depth behavior and is directly compatible with
    ``GestureLMOfflineGen``. Each cross-attention layer retains its own
    configured context and alignment while reusing the full-memory KV cache.
    """

    temporal_attention_class = ConditionCMemoryCrossAttention


@contextmanager
def _condition_c_teacher_attention(model):
    """Install C views for a detached pass and restore all original state.

    Original modules are restored before any later causal-student backward
    (including checkpoint recomputation). Existing streaming state objects
    are temporarily suspended, so this complete-sequence teacher pass does
    not advance or overwrite an active generator's caches.
    """

    streaming_states = [
        (module, module._streaming_state)
        for module in model.modules()
        if isinstance(module, StreamingModule)
    ]
    original_memories = []
    try:
        for module, _ in streaming_states:
            module._streaming_state = None
        for layer in model.temporal_transformer.layers:
            original_memories.append((layer, layer.cross_attns))
            layer.cross_attns = torch.nn.ModuleList(
                [
                    ConditionCMemoryCrossAttention.shared_view(attention)
                    for attention in layer.cross_attns
                ]
            )
        yield
    finally:
        for layer, cross_attns in original_memories:
            layer.cross_attns = cross_attns
        for module, state in streaming_states:
            module._streaming_state = state


@torch.no_grad()
def forward_condition_c_teacher_view(
    model,
    *,
    input_codes: torch.Tensor,
    audio_codes: torch.Tensor,
    text_codes: torch.Tensor,
    sum_condition: torch.Tensor,
    ca_depth_padding_mask: torch.Tensor | None = None,
    include_depth_levels: bool = True,
    depth_input_codes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Detached shared-weight C teacher; same interface as GlobalRegret.

    Both paths see the same teacher-forced gesture history and, when given,
    the same stochastic RVQ depth prefix. Only future speech visibility
    changes. Train/eval mode is preserved; this helper does not alter dropout
    policy or permanently change any module, flag, parameter, or cache.
    """

    temporal_sequence = _prepend_initial_and_shift(model, input_codes)
    temporal_sum_condition = sum_condition.unsqueeze(1).expand(
        -1, temporal_sequence.shape[-1],
    )
    audio_condition, text_condition = _process_conditions_squeezed(
        model, audio_codes, text_codes,
    )
    with _condition_c_teacher_attention(model):
        teacher_transformer_out, temporal_logits = model.forward_temporal(
            temporal_sequence,
            audio_condition=audio_condition,
            text_condition=text_condition,
            sum_condition=temporal_sum_condition,
        )
        depth_logits = None
        if include_depth_levels:
            depth_logits = _forward_depth_branch(
                model,
                teacher_transformer_out=teacher_transformer_out,
                input_codes=input_codes,
                depth_input_codes=depth_input_codes,
                audio_condition=audio_condition,
                text_condition=text_condition,
                sum_condition=sum_condition,
                ca_depth_padding_mask=ca_depth_padding_mask,
            )
    return temporal_logits, depth_logits
