"""
Automatic Speech Recognition (ASR) module using Whisper.
"""
import logging
import pickle
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

from .common_types import StageCache, TranscriptSegment
from .config import ASRConfig
from .device_utils import detect_device

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """Transcribes audio using Whisper models."""

    def __init__(self, config: ASRConfig, cache: StageCache):
        self.config = config
        self.cache = cache
        self.device = self._map_device(detect_device(config.device))

        # Load Whisper only if the transcript cache cannot satisfy a request.
        self.model = None

    def _ensure_model_loaded(self):
        """Load Whisper on first transcription request."""
        if self.model is None:
            self._load_model()

    def _map_device(self, device: str) -> str:
        """Map device names for faster-whisper."""
        # faster-whisper uses "cpu" or "cuda"
        if device == "mps":
            logger.warning("MPS not supported by faster-whisper, using CPU")
            return "cpu"
        return device

    def _load_model(self):
        """Load the Whisper model."""
        logger.info(f"Loading Whisper model: {self.config.model_name}")

        try:
            self.model = WhisperModel(
                self.config.model_name,
                device=self.device,
                compute_type=self.config.compute_type
            )
            logger.info(f"Whisper model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def transcribe_audio(self, audio_path: Path, video_id: str) -> List[TranscriptSegment]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            video_id: Video identifier for caching

        Returns:
            List of TranscriptSegment objects
        """
        if not audio_path or not audio_path.exists():
            logger.warning("No audio file provided or file does not exist")
            return []

        # Check cache
        cache_path = self.cache.get_path(video_id, "transcripts", "pkl")
        if self.cache.enabled and self.config.cache_transcripts and cache_path.exists():
            logger.info(f"Loading transcripts from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        logger.info(f"Transcribing audio: {audio_path}")
        self._ensure_model_loaded()

        # Transcribe
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            beam_size=self.config.beam_size,
        )

        logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

        # Convert to TranscriptSegment objects
        transcripts = []
        for segment in segments:
            transcripts.append(TranscriptSegment(
                start_time=segment.start,
                end_time=segment.end,
                text=segment.text.strip(),
                confidence=segment.avg_logprob  # Use avg log probability as confidence
            ))

        logger.info(f"Generated {len(transcripts)} transcript segments")

        # Cache transcripts
        if self.cache.enabled and self.config.cache_transcripts:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(transcripts, f)

        return transcripts

    def get_full_transcript(self, transcripts: List[TranscriptSegment]) -> str:
        """
        Get full transcript as a single string.

        Args:
            transcripts: List of transcript segments

        Returns:
            Full transcript text
        """
        return " ".join([seg.text for seg in transcripts])
