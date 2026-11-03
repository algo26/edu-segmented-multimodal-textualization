"""Video decoding, paper-defined frame sampling, and audio extraction."""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import cv2
import ffmpeg
from PIL import Image

from .common_types import FrameData, StageCache, TranscriptSegment
from .config import AudioConfig, VideoConfig

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Extract only the frames requested by the two sampling regimes."""

    def __init__(
        self,
        video_config: VideoConfig,
        audio_config: AudioConfig,
        cache: StageCache,
    ):
        self.video_config = video_config
        self.audio_config = audio_config
        self.cache = cache

    def extract_frames(self, video_path: Path, video_id: str) -> List[FrameData]:
        """Extract the complete video uniformly at exactly 0.5 fps, without a cap."""
        duration = self.get_video_duration(video_path)
        if duration <= 0:
            return []

        period = 1.0 / self.video_config.fps
        timestamps: List[float] = []
        timestamp = 0.0
        while timestamp < duration:
            timestamps.append(timestamp)
            timestamp += period

        return self.extract_frames_at_timestamps(
            video_path=video_path,
            video_id=video_id,
            timestamps=timestamps,
            cache_stage="frames_uniform",
        )

    def extract_transcript_anchor_frames(
        self,
        video_path: Path,
        video_id: str,
        transcripts: List[TranscriptSegment],
    ) -> List[FrameData]:
        """Decode frames at the exact speech-anchored timestamps from the method."""
        timestamps = self.build_transcript_anchor_timestamps(
            transcripts=transcripts,
            video_duration=self.get_video_duration(video_path),
        )
        return self.extract_frames_at_timestamps(
            video_path=video_path,
            video_id=video_id,
            timestamps=timestamps,
            cache_stage="frames_transcript_segments",
        )

    def build_transcript_anchor_timestamps(
        self,
        transcripts: List[TranscriptSegment],
        video_duration: float,
    ) -> List[float]:
        """Apply the 1/2/3 frame rule to speech and qualifying silent gaps."""
        ordered = sorted(transcripts, key=lambda segment: (segment.start_time, segment.end_time))
        for segment in ordered:
            if segment.start_time < 0 or segment.end_time <= segment.start_time:
                raise ValueError(f"Invalid transcript segment: {segment}")

        timestamps: List[float] = []
        for segment in ordered:
            timestamps.extend(
                self._segment_sample_timestamps(segment.start_time, segment.end_time)
            )

        threshold = self.video_config.silence_gap_seconds
        for left, right in zip(ordered, ordered[1:]):
            gap_start = max(0.0, left.end_time)
            gap_end = min(video_duration, right.start_time)
            if gap_end - gap_start >= threshold:
                timestamps.extend(self._segment_sample_timestamps(gap_start, gap_end))

        # Eight benchmark videos contain no detected speech. Treat such a video
        # as one silent interval so the visual channel remains defined.
        if not ordered and video_duration >= threshold:
            timestamps.extend(self._segment_sample_timestamps(0.0, video_duration))

        return self._dedupe_timestamps(timestamps, video_duration)

    def extract_frames_at_timestamps(
        self,
        video_path: Path,
        video_id: str,
        timestamps: List[float],
        cache_stage: str,
    ) -> List[FrameData]:
        """Decode requested frames and retain both anchor and discrete frame times."""
        if not timestamps:
            return []

        cache_path = self.cache.get_path(video_id, cache_stage, "pkl")
        if self.cache.enabled and self.video_config.cache_frames and cache_path.exists():
            logger.info("Loading frames from artifacts: %s", cache_path)
            with open(cache_path, "rb") as file:
                return pickle.load(file)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0 or total_frames <= 0:
            capture.release()
            raise ValueError(f"Could not determine frame timing for video: {video_path}")

        frame_dir = self.cache.cache_dir / cache_stage / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames: List[FrameData] = []

        for requested_timestamp in timestamps:
            frame_index = int(round(requested_timestamp * source_fps))
            frame_index = max(0, min(total_frames - 1, frame_index))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                logger.warning("Skipping unreadable frame at index %s", frame_index)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, self.video_config.frame_size)
            image_path = self._save_frame_image(
                frame_dir=frame_dir,
                frame_rgb=frame_rgb,
                requested_timestamp=requested_timestamp,
                frame_index=frame_index,
            )
            frames.append(
                FrameData(
                    timestamp=frame_index / source_fps,
                    frame_idx=frame_index,
                    image_path=image_path,
                    requested_timestamp=requested_timestamp,
                )
            )

        capture.release()
        if self.cache.enabled and self.video_config.cache_frames:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as file:
                pickle.dump(frames, file)
        return frames

    def extract_audio(self, video_path: Path, video_id: str) -> Optional[Path]:
        """Extract a mono 16 kHz PCM track for Whisper."""
        if not self.audio_config.extract_audio:
            return None

        audio_path = self.cache.cache_dir / "audio" / f"{video_id}.wav"
        if self.cache.enabled and self.audio_config.cache_audio and audio_path.exists():
            return audio_path

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(audio_path),
                acodec="pcm_s16le",
                ar=self.audio_config.sample_rate,
                ac=1,
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return audio_path
        except ffmpeg.Error as error:
            logger.error("Audio extraction failed: %s", error)
            return None

    def get_video_duration(self, video_path: Path) -> float:
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return frame_count / fps if fps > 0 else 0.0

    def _segment_sample_timestamps(self, start_time: float, end_time: float) -> List[float]:
        duration = end_time - start_time
        if duration <= 0:
            return []
        if duration < self.video_config.transcript_segment_medium_seconds:
            count = 1
        elif duration < self.video_config.transcript_segment_long_seconds:
            count = 2
        else:
            count = 3
        return [
            start_time + (index * duration) / (count + 1)
            for index in range(1, count + 1)
        ]

    @staticmethod
    def _dedupe_timestamps(timestamps: List[float], duration: float) -> List[float]:
        valid = {
            round(timestamp, 6)
            for timestamp in timestamps
            if 0.0 <= timestamp < duration
        }
        return sorted(valid)

    @staticmethod
    def _save_frame_image(
        *,
        frame_dir: Path,
        frame_rgb,
        requested_timestamp: float,
        frame_index: int,
    ) -> Path:
        milliseconds = int(round(requested_timestamp * 1000))
        image_path = frame_dir / f"frame_{milliseconds:012d}ms_{frame_index:09d}.jpg"
        Image.fromarray(frame_rgb).save(image_path)
        return image_path
