"""Evaluate MIBURI with one correct speaker ID per batch sample.

This is an opt-in companion to ``scripts/test.py``.  The released evaluator
keeps its historical singleton/first-sample speaker broadcast so existing
commands remain reproducible.  This entry point enables the corrected batched
speaker path while otherwise using the same CLI, trainer, sampling policy,
metrics, and checkpoint loading code.
"""

import os

from test import main_worker
from trainers.utils import config


if __name__ == "__main__":
    args = config.parse_args()
    args.is_train = False
    args.ddp = False
    args.name = os.path.dirname(args.test_ckpt)
    args.eval_correct_speaker_batch = True

    main_worker(0, 1, args)
