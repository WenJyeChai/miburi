"""Utilities for future-aware q0 regret distillation.

The frozen privileged teacher only needs the temporal path that produces the
upper-body q0 distribution.  The kinematic/depth path is intentionally
discarded after loading the teacher checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .gesture_lm_future_gesture import (
    GTemporalDepthModel3FutureGestureFullCondition,
)


TEMPORAL_Q0_STATE_PREFIXES = (
    "audio_procemb.",
    "text_procemb.",
    "temporal_gemb.",
    "temporal_gproj.",
    "spk_tempemb.",
    "temporal_transformer.",
    "out_norm.",
    "temporal_classifier.",
    "temp_condproj.",
)


def is_temporal_q0_state_key(name: str) -> bool:
    """Return whether a checkpoint tensor is required by the q0 teacher."""

    return name.startswith(TEMPORAL_Q0_STATE_PREFIXES)


def retain_temporal_q0_only(
    model: GTemporalDepthModel3FutureGestureFullCondition,
) -> GTemporalDepthModel3FutureGestureFullCondition:
    """Remove every learned module unused by temporal q0 inference.

    The operation is performed only after the full teacher checkpoint has
    been loaded.  Keeping the original class lets us reuse its carefully
    tested masked-target forward path while avoiding GPU storage for the
    kinematic transformer and auxiliary heads.
    """

    unused_modules = (
        "gesture_codec_layers",
        "depformer_in",
        "body_part_emb",
        "spk_depemb",
        "depformer_gemb",
        "depformer_gproj",
        "depth_transformer",
        "depformer_classifier",
        "dep_condproj",
        "vad_predictor",
        "vad_face_proj",
    )
    for name in unused_modules:
        if hasattr(model, name):
            setattr(model, name, None)
    return model


def forward_q0_regret_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute forward KL ``KL(stopgrad(teacher) || student)``.

    Both tensors must contain logits over real gesture tokens only.  The
    caller must remove MIBURI's final PAD class before invoking this helper.
    The result is the mean KL per selected target.  ``temperature**2`` keeps
    the gradient scale comparable when distillation temperature is changed.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher q0 logits must have identical shapes, got "
            f"{tuple(student_logits.shape)} and "
            f"{tuple(teacher_logits.shape)}."
        )
    if student_logits.dim() != 2:
        raise ValueError(
            "Expected selected q0 logits [B,V], got "
            f"{tuple(student_logits.shape)}."
        )
    if temperature <= 0:
        raise ValueError("Regret temperature must be positive.")

    student_log_probs = F.log_softmax(
        student_logits.float() / temperature,
        dim=-1,
    )
    with torch.no_grad():
        teacher_probs = F.softmax(
            teacher_logits.float() / temperature,
            dim=-1,
        )
    return (
        F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )
        * (temperature**2)
    )
