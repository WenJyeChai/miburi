"""Condition C: future speech is the teacher's only attention privilege.

Both experiments retain the student's rolling past-speech boundary and
causal gesture history. The teacher can additionally attend to every future
speech token in the supplied clip, independently of sentence boundaries.
The ForeMotion run trains a causal student against its detached shared-weight
teacher. The teacher-only run directly supervises the condition-C forward.
"""

from loguru import logger

from miburi.models import (
    GTemporalDepthModel3ConditionC,
    forward_condition_c_teacher_view,
)

from .uflgtdm3_offline_trainer import UpperFaceLowerGTDM3OfflineTrainer
from .uflgtdm3_shared_regret_rvq_trainer import (
    StochasticRVQTrainingMixin,
    UpperFaceLowerGTDM3SharedRegretRVQTrainer,
)


def _validate_condition_c_args(args, *, teacher_only=False):
    if float(getattr(args, "dense_future_gesture_weight", 0.0)) != 0:
        raise ValueError(
            "Condition C privileges future speech only: "
            "dense_future_gesture_weight must be zero."
        )
    if getattr(args, "regret_teacher_ckpt", None):
        raise ValueError(
            "Condition C uses shared weights or a directly trained teacher, "
            "not regret_teacher_ckpt. To resume a run, use continue_ckpt."
        )
    if teacher_only and (
        float(getattr(args, "regret_weight", 0.0)) != 0
        or float(getattr(args, "regret_initial_weight", 0.0)) != 0
    ):
        raise ValueError(
            "The condition-C teacher-only trainer has no student or KL loss. "
            "Set regret_weight and regret_initial_weight to zero."
        )


class UpperFaceLowerGTDM3ConditionCRegretRVQTrainer(
    UpperFaceLowerGTDM3SharedRegretRVQTrainer
):
    """ForeMotion: causal student + detached shared-weight condition C."""

    _REGRET_VIEW_NAME = "condition-C future-only regret"

    def __init__(self, args):
        _validate_condition_c_args(args)
        super().__init__(args)

    def _forward_regret_teacher_view(
        self,
        *,
        split,
        input_codes,
        audio_codes,
        text_codes,
        sum_condition,
        ca_depth_padding_mask,
        depth_input_codes,
        sample_ids=None,
    ):
        del split, sample_ids
        return forward_condition_c_teacher_view(
            self._student_model(),
            input_codes=input_codes,
            audio_codes=audio_codes,
            text_codes=text_codes,
            sum_condition=sum_condition,
            ca_depth_padding_mask=ca_depth_padding_mask,
            include_depth_levels=self.regret_include_depth_levels,
            depth_input_codes=depth_input_codes,
        )


class UpperFaceLowerGTDM3ConditionCTeacherRVQTrainer(
    StochasticRVQTrainingMixin, UpperFaceLowerGTDM3OfflineTrainer
):
    """One gradient-enabled C forward; token/RVQ and existing auxiliaries.

    There is no SharedRegret ancestor, student pass, KL loss, or EMA model.
    Training and deterministic validation both use condition C. Generation
    caches the full supplied speech memory and applies the same C boundary
    at each autoregressive gesture step.
    """

    model_class = GTemporalDepthModel3ConditionC

    def __init__(self, args):
        _validate_condition_c_args(args, teacher_only=True)
        super().__init__(args)
        logger.info(
            "Condition-C teacher-only training: gradients through temporal "
            "and depth token/RVQ losses plus configured auxiliaries; "
            "rolling past speech + all supplied future speech; "
            "causal gesture history; no student/KL pass."
        )
