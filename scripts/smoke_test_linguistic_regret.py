"""Small CPU smoke test for word/sentence teacher boundary masks.

Run from the repository root:

    python scripts/smoke_test_linguistic_regret.py
"""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import torch

_MODULE_PATH = (
    Path(__file__).resolve().parent
    / "trainers"
    / "utils"
    / "linguistic_boundaries.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "linguistic_boundaries_smoke", _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_BOUNDARIES = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BOUNDARIES
_SPEC.loader.exec_module(_BOUNDARIES)
WhisperLinguisticBoundaryIndex = (
    _BOUNDARIES.WhisperLinguisticBoundaryIndex
)
split_beatx_filechunk_id = _BOUNDARIES.split_beatx_filechunk_id


def _write_transcript(root: str) -> None:
    transcript_dir = os.path.join(root, "whisper_transcription")
    os.makedirs(transcript_dir, exist_ok=True)
    payload = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "hello weekend",
                "words": [
                    {"word": "hello", "start": 0.2, "end": 0.6},
                    {"word": "weekend", "start": 0.8, "end": 1.4},
                ],
            }
        ]
    }
    with open(
        os.path.join(transcript_dir, "1_scott_0_1_1.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle)


def _make_index(root: str, mode: str, cap: float = 0.0):
    return WhisperLinguisticBoundaryIndex(
        root,
        mode=mode,
        motion_fps=10,
        pose_length=20,
        condition_frame_rate=10,
        gesture_frame_rate=10,
        max_lookahead_seconds=cap,
    )


def main() -> None:
    assert split_beatx_filechunk_id("1_scott_0_1_1_C3") == (
        "1_scott_0_1_1",
        3,
    )
    assert split_beatx_filechunk_id("1_scott_0_1_1") == (
        "1_scott_0_1_1",
        0,
    )

    with tempfile.TemporaryDirectory() as root:
        _write_transcript(root)
        word_index = _make_index(root, "word")
        word_batch = word_index.build_cross_attention_bias(
            ["1_scott_0_1_1_C0"],
            query_length=20,
            condition_length=20,
            device=torch.device("cpu"),
        )
        # q=3 is at 0.35 s, inside hello [0.2,0.6): keys through 5.
        assert word_batch.lookahead_tokens[0, 3].item() == 2
        assert word_batch.attention_bias[0, 0, 3, :6].all()
        assert not word_batch.attention_bias[0, 0, 3, 6:].any()
        # q=6 is outside either word and therefore remains causal.
        assert word_batch.lookahead_tokens[0, 6].item() == 0
        assert word_batch.attention_bias[0, 0, 6, :7].all()
        assert not word_batch.attention_bias[0, 0, 6, 7:].any()

        sentence = _make_index(root, "sentence").build_cross_attention_bias(
            ["1_scott_0_1_1_C0"],
            query_length=20,
            condition_length=20,
            device=torch.device("cpu"),
        )
        # Every query inside the Whisper segment sees through 1.5 s (key 14).
        assert sentence.lookahead_tokens[0, 3].item() == 11
        assert sentence.attention_bias[0, 0, 3, :15].all()
        assert not sentence.attention_bias[0, 0, 3, 15:].any()

        capped = _make_index(root, "sentence", cap=0.2)
        capped_batch = capped.build_cross_attention_bias(
            ["1_scott_0_1_1_C0"],
            query_length=20,
            condition_length=20,
            device=torch.device("cpu"),
        )
        assert capped_batch.lookahead_tokens.max().item() <= 2

        # C1 starts at 2 s, after all transcript spans: entirely causal.
        later_chunk = word_index.build_cross_attention_bias(
            ["1_scott_0_1_1_C1"],
            query_length=20,
            condition_length=20,
            device=torch.device("cpu"),
        )
        assert later_chunk.lookahead_tokens.count_nonzero().item() == 0

    print("linguistic regret boundary smoke test passed")


if __name__ == "__main__":
    main()
