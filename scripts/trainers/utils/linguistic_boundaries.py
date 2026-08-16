"""Whisper-aligned attention cutoffs for linguistic regret teachers.

The BEATX HDF5 intentionally stores the expensive Moshi audio/text tokens
but not the word-level timestamps used to create the text stream.  This
module reconstructs a tiny boundary view directly from the source Whisper
JSON and caches parsed files in each training process.  Existing HDF5 files
therefore remain read-only and do not need to be rebuilt.

For every gesture query, ``build_cross_attention_bias`` returns the final
audio/text key that a privileged teacher may inspect.  Queries inside a word
can look through that word's end; sentence mode uses the enclosing Whisper
segment.  Queries outside the requested span remain strictly causal.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
from typing import Literal, Sequence

import torch


BoundaryMode = Literal["word", "sentence"]


@dataclass(frozen=True)
class TimedSpan:
    start: float
    end: float


@dataclass(frozen=True)
class LinguisticBoundaryBatch:
    attention_bias: torch.Tensor
    lookahead_tokens: torch.Tensor
    active_positions: torch.Tensor


def split_beatx_filechunk_id(filechunk_id: str) -> tuple[str, int]:
    """Return ``(source_file_id, chunk_index)`` for a BEATX HDF5 key."""

    match = re.match(r"^(?P<file_id>.+)_C(?P<chunk_index>\d+)$", filechunk_id)
    if match is None:
        # Full-sequence HDF5 entries deliberately have no ``_C<n>`` suffix.
        return filechunk_id, 0
    return match.group("file_id"), int(match.group("chunk_index"))


def _finite_interval(start, end) -> TimedSpan | None:
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start_f) or not math.isfinite(end_f):
        return None
    if end_f <= start_f:
        return None
    return TimedSpan(start_f, end_f)


class WhisperLinguisticBoundaryIndex:
    """Lazy, process-local cache of BEATX Whisper timing spans."""

    def __init__(
        self,
        beatx_data_path: str,
        *,
        mode: BoundaryMode,
        motion_fps: float,
        pose_length: int,
        condition_frame_rate: float,
        gesture_frame_rate: float,
        max_lookahead_seconds: float = 0.0,
        transcript_subdir: str = "whisper_transcription",
    ):
        if mode not in ("word", "sentence"):
            raise ValueError(f"Unsupported linguistic boundary mode: {mode!r}")
        if not beatx_data_path:
            raise ValueError(
                "--beatx_data_path is required for linguistic regret training."
            )
        if motion_fps <= 0 or pose_length <= 0:
            raise ValueError("motion_fps and pose_length must be positive.")
        if condition_frame_rate <= 0 or gesture_frame_rate <= 0:
            raise ValueError("Condition/gesture frame rates must be positive.")
        if max_lookahead_seconds < 0:
            raise ValueError(
                "linguistic_regret_max_lookahead_seconds cannot be negative."
            )

        self.transcript_dir = os.path.join(
            beatx_data_path, transcript_subdir,
        )
        self.mode = mode
        self.motion_fps = float(motion_fps)
        self.pose_length = int(pose_length)
        self.condition_frame_rate = float(condition_frame_rate)
        self.gesture_frame_rate = float(gesture_frame_rate)
        self.max_lookahead_seconds = float(max_lookahead_seconds)
        self._span_cache: dict[str, tuple[TimedSpan, ...]] = {}
        self._cutoff_cache: dict[
            tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @property
    def chunk_duration_seconds(self) -> float:
        return self.pose_length / self.motion_fps

    def _transcript_path(self, file_id: str) -> str:
        return os.path.join(self.transcript_dir, f"{file_id}.json")

    def _load_spans(self, file_id: str) -> tuple[TimedSpan, ...]:
        cached = self._span_cache.get(file_id)
        if cached is not None:
            return cached

        transcript_path = self._transcript_path(file_id)
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(
                "Linguistic regret requires the Whisper transcript for "
                f"{file_id!r}, but it was not found at {transcript_path!r}."
            )
        with open(transcript_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        segments = payload.get("segments", [])
        if not isinstance(segments, list):
            raise ValueError(
                f"Expected a list at segments in {transcript_path!r}."
            )

        spans: list[TimedSpan] = []
        if self.mode == "word":
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                words = segment.get("words", [])
                if not isinstance(words, list):
                    continue
                for word in words:
                    if not isinstance(word, dict):
                        continue
                    span = _finite_interval(word.get("start"), word.get("end"))
                    if span is not None:
                        spans.append(span)
        else:
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                span = _finite_interval(
                    segment.get("start"), segment.get("end"),
                )
                if span is None:
                    valid_words = []
                    words = segment.get("words", [])
                    if isinstance(words, list):
                        valid_words = [
                            word_span
                            for word in words
                            if isinstance(word, dict)
                            for word_span in [
                                _finite_interval(
                                    word.get("start"), word.get("end"),
                                )
                            ]
                            if word_span is not None
                        ]
                    if valid_words:
                        span = TimedSpan(
                            min(item.start for item in valid_words),
                            max(item.end for item in valid_words),
                        )
                if span is not None:
                    spans.append(span)

        spans.sort(key=lambda item: (item.start, item.end))
        result = tuple(spans)
        self._span_cache[file_id] = result
        return result

    @staticmethod
    def _enclosing_span(
        spans: Sequence[TimedSpan], timestamp: float,
    ) -> TimedSpan | None:
        # Whisper timings are sorted and normally non-overlapping.  Retain
        # the narrowest enclosing interval if malformed metadata overlaps.
        candidates = [
            span for span in spans if span.start <= timestamp < span.end
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.end - item.start)

    def build_cross_attention_bias(
        self,
        filechunk_ids: Sequence[str],
        *,
        query_length: int,
        condition_length: int,
        device: torch.device,
    ) -> LinguisticBoundaryBatch:
        """Construct ``[B,1,Tq,Tk]`` mask; ``True`` means attention allowed."""

        if query_length <= 0 or condition_length <= 0:
            raise ValueError("Query and condition lengths must be positive.")
        batch = len(filechunk_ids)
        if batch == 0:
            raise ValueError("filechunk_ids cannot be empty.")

        # This reproduces the causal student's final visible condition key
        # for aligned streams.  It also remains correct if a future model has
        # an integer number of condition steps per gesture step.
        scale = condition_length / query_length
        current_cutoffs = torch.ceil(
            (torch.arange(query_length, dtype=torch.float64) + 1.0) * scale
        ).to(torch.long) - 1
        current_cutoffs.clamp_(0, condition_length - 1)
        cutoffs = current_cutoffs.unsqueeze(0).expand(batch, -1).clone()
        active = torch.zeros((batch, query_length), dtype=torch.bool)

        for batch_index, filechunk_id in enumerate(filechunk_ids):
            cache_key = (
                str(filechunk_id), query_length, condition_length,
            )
            cached_cutoffs = self._cutoff_cache.get(cache_key)
            if cached_cutoffs is not None:
                cutoffs[batch_index] = cached_cutoffs[0]
                active[batch_index] = cached_cutoffs[1]
                continue

            source_id, chunk_index = split_beatx_filechunk_id(
                str(filechunk_id),
            )
            chunk_start = chunk_index * self.chunk_duration_seconds
            spans = self._load_spans(source_id)
            for query_index in range(query_length):
                query_time = (
                    chunk_start
                    # Match Interleaver.build_token_stream: condition step q
                    # contains alignments whose start is before the right
                    # edge of that step (q + 1), not merely its center.
                    + (query_index + 1.0) / self.gesture_frame_rate
                    - 1e-9
                )
                enclosing = self._enclosing_span(spans, query_time)
                if enclosing is None:
                    continue

                boundary_cutoff = (
                    math.ceil(
                        (enclosing.end - chunk_start)
                        * self.condition_frame_rate
                    )
                    - 1
                )
                boundary_cutoff = max(
                    int(current_cutoffs[query_index]), boundary_cutoff,
                )
                boundary_cutoff = min(condition_length - 1, boundary_cutoff)
                if self.max_lookahead_seconds > 0:
                    max_extra = math.ceil(
                        self.max_lookahead_seconds
                        * self.condition_frame_rate
                    )
                    boundary_cutoff = min(
                        boundary_cutoff,
                        int(current_cutoffs[query_index]) + max_extra,
                    )
                cutoffs[batch_index, query_index] = boundary_cutoff
                active[batch_index, query_index] = (
                    boundary_cutoff > current_cutoffs[query_index]
                )
            self._cutoff_cache[cache_key] = (
                cutoffs[batch_index].clone(),
                active[batch_index].clone(),
            )

        keys = torch.arange(condition_length, dtype=torch.long)
        attention_bias = keys.view(1, 1, 1, -1) <= cutoffs[:, None, :, None]
        lookahead = cutoffs - current_cutoffs.unsqueeze(0)
        return LinguisticBoundaryBatch(
            attention_bias=attention_bias.to(device=device),
            lookahead_tokens=lookahead.to(device=device),
            active_positions=active.to(device=device),
        )
