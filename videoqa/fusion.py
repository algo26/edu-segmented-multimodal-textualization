"""Deterministic temporal fusion using the two policies in the paper."""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from typing import List, Sequence

from .common_types import Caption, FusedContent, StageCache, TranscriptSegment
from .config import FusionConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Window:
    start: float
    end: float
    kind: str


class MultimodalFusion:
    """Group raw perception outputs without summarizing or deduplicating them."""

    def __init__(self, config: FusionConfig, cache: StageCache):
        self.config = config
        self.cache = cache

    def fuse(
        self,
        captions: List[Caption],
        transcripts: List[TranscriptSegment],
        video_id: str,
        *,
        video_duration: float | None = None,
    ) -> FusedContent:
        cache_path = self.cache.get_path(video_id, "fusion", "pkl")
        if self.cache.enabled and self.config.cache_fusion and cache_path.exists():
            logger.info("Loading fused representation from artifacts: %s", cache_path)
            with open(cache_path, "rb") as file:
                return pickle.load(file)

        duration = self._duration(captions, transcripts, video_duration)
        if self.config.windowing_strategy == "fixed":
            windows = self._fixed_windows(duration)
        elif self.config.windowing_strategy == "transcript_segments":
            windows = self._variable_windows(transcripts, duration)
        else:
            raise ValueError(
                f"Unsupported windowing strategy: {self.config.windowing_strategy}"
            )

        timeline = self._render(windows, captions, transcripts, include_visual=True, include_audio=True)
        captions_text = self._render(
            windows,
            captions,
            transcripts,
            include_visual=True,
            include_audio=False,
        )
        transcripts_text = self._render(
            windows,
            captions,
            transcripts,
            include_visual=False,
            include_audio=True,
        )
        fused = FusedContent(
            timeline_text=timeline,
            narrative_text=timeline,
            captions_text=captions_text,
            transcripts_text=transcripts_text,
            metadata={
                "windowing_strategy": self.config.windowing_strategy,
                "window_count": len(windows),
                "num_captions": len(captions),
                "num_transcripts": len(transcripts),
            },
        )

        if self.cache.enabled and self.config.cache_fusion:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as file:
                pickle.dump(fused, file)
        return fused

    def _fixed_windows(self, duration: float) -> List[_Window]:
        if duration <= 0:
            return []
        size = self.config.timeline_window_seconds
        count = math.ceil(duration / size)
        return [
            _Window(index * size, min((index + 1) * size, duration), "fixed")
            for index in range(count)
        ]

    def _variable_windows(
        self,
        transcripts: Sequence[TranscriptSegment],
        duration: float,
    ) -> List[_Window]:
        ordered = sorted(transcripts, key=lambda segment: (segment.start_time, segment.end_time))
        windows = [
            _Window(segment.start_time, segment.end_time, "speech")
            for segment in ordered
        ]
        for left, right in zip(ordered, ordered[1:]):
            if right.start_time - left.end_time >= 8.0:
                windows.append(_Window(left.end_time, right.start_time, "silence"))
        if not ordered and duration >= 8.0:
            windows.append(_Window(0.0, duration, "silence"))
        return sorted(windows, key=lambda window: (window.start, window.end, window.kind))

    def _render(
        self,
        windows: Sequence[_Window],
        captions: Sequence[Caption],
        transcripts: Sequence[TranscriptSegment],
        *,
        include_visual: bool,
        include_audio: bool,
    ) -> str:
        blocks = []
        for index, window in enumerate(windows):
            is_last = index == len(windows) - 1
            entries: list[tuple[float, int, str]] = []

            if include_visual:
                for caption in captions:
                    if self._contains(caption.timestamp, window, is_last):
                        entries.append(
                            (
                                caption.timestamp,
                                0,
                                f"VISUAL {self._format_timestamp(caption.timestamp)}: {caption.text.strip()}",
                            )
                        )

            if include_audio:
                for transcript in transcripts:
                    midpoint = (transcript.start_time + transcript.end_time) / 2.0
                    if self._contains(midpoint, window, is_last):
                        entries.append(
                            (
                                transcript.start_time,
                                1,
                                "SPEECH "
                                f"{self._format_timestamp(transcript.start_time)}-"
                                f"{self._format_timestamp(transcript.end_time)}: "
                                f"{transcript.text.strip()}",
                            )
                        )

            entries.sort(key=lambda entry: (entry[0], entry[1]))
            header = (
                f"[{self._format_timestamp(window.start)}-"
                f"{self._format_timestamp(window.end)}]"
            )
            blocks.append("\n".join([header, *(entry[2] for entry in entries)]))
        return " | ".join(blocks)

    @staticmethod
    def _contains(timestamp: float, window: _Window, is_last: bool) -> bool:
        if window.kind in {"speech", "silence"}:
            return window.start <= timestamp <= window.end
        if is_last:
            return window.start <= timestamp <= window.end
        return window.start <= timestamp < window.end

    @staticmethod
    def _duration(
        captions: Sequence[Caption],
        transcripts: Sequence[TranscriptSegment],
        video_duration: float | None,
    ) -> float:
        if video_duration is not None:
            return max(0.0, video_duration)
        return max(
            [caption.timestamp for caption in captions]
            + [segment.end_time for segment in transcripts]
            + [0.0]
        )

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        milliseconds = int(round(max(0.0, seconds) * 1000))
        whole_seconds, millis = divmod(milliseconds, 1000)
        minutes, secs = divmod(whole_seconds, 60)
        return f"{minutes:02d}:{secs:02d}.{millis:03d}"

    def truncate_to_context_length(self, text: str, max_length: int | None = None) -> str:
        """Retain the complete representation; the method applies no compression."""
        return text
