"""Evaluate MIBURI with one correct speaker ID per batch sample.

This is an opt-in companion to ``scripts/test.py``.  The released evaluator
keeps its historical singleton/first-sample speaker broadcast so existing
commands remain reproducible.  This entry point enables the corrected batched
speaker path while otherwise using the same CLI, trainer, sampling policy,
metrics, and checkpoint loading code.
"""

import os
import sys
from pathlib import Path


# ``python scripts/<entrypoint>.py`` puts ``scripts/`` rather than the
# repository root first on sys.path.  On machines with a non-editable MIBURI
# installation, that can silently import a stale site-packages copy.  This
# evaluator depends on the matching per-sample-CFG generator implementation,
# so resolve the repository package explicitly before importing the trainer.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
if _REPOSITORY_ROOT in sys.path:
    sys.path.remove(_REPOSITORY_ROOT)
sys.path.insert(0, _REPOSITORY_ROOT)

import miburi

print(
    "[test_correct_speaker] Using miburi package from "
    f"{Path(miburi.__file__).resolve()}"
)

from test import main_worker
from trainers.utils import config


if __name__ == "__main__":
    args = config.parse_args()
    args.is_train = False
    args.ddp = False
    args.name = os.path.dirname(args.test_ckpt)
    args.eval_correct_speaker_batch = True

    main_worker(0, 1, args)
