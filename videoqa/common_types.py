"""
Common type definitions for the video QA pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FrameData:
    """Represents a single video frame."""
    timestamp: float  # actual decoded-frame timestamp, in seconds
    frame_idx: int
    image_path: Optional[Path] = None
    image_array: Optional[Any] = None  # numpy array or PIL Image
    requested_timestamp: Optional[float] = None  # sampling anchor associated with this frame


@dataclass
class AudioSegment:
    """Represents an audio segment."""
    start_time: float  # seconds
    end_time: float  # seconds
    audio_path: Optional[Path] = None
    audio_array: Optional[Any] = None


@dataclass
class Caption:
    """Represents a frame caption."""
    timestamp: float
    frame_idx: int
    text: str
    confidence: Optional[float] = None
    decoded_frame_timestamp: Optional[float] = None


@dataclass
class TranscriptSegment:
    """Represents a transcript segment."""
    start_time: float
    end_time: float
    text: str
    confidence: Optional[float] = None


@dataclass
class FusedContent:
    """Represents fused multimodal content."""
    timeline_text: str  # Chronologically ordered fusion
    narrative_text: str  # Semantic/narrative fusion
    captions_text: str = ""  # Visual-only prompt context
    transcripts_text: str = ""  # Audio-only prompt context
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoQASample:
    """Represents a video QA sample."""
    video_id: str
    video_path: Path
    question: str
    answer: Optional[str] = None  # Ground truth
    question_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Represents a prediction result."""
    question_id: str
    video_id: str
    question: str
    predicted_answer: str
    ground_truth: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedVideo:
    """Represents a fully processed video."""
    video_id: str
    frames: List[FrameData]
    audio_segments: List[AudioSegment]
    captions: List[Caption]
    transcripts: List[TranscriptSegment]
    fused_content: Optional[FusedContent] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageCache:
    """Cache information for a processing stage."""
    cache_dir: Path
    enabled: bool = True

    def get_path(self, video_id: str, stage_name: str, extension: str = "pkl") -> Path:
        """
        Get cache file path for a specific video and stage.
        Organizes cache into subfolders by stage.

        Structure:
        cache/
        ├── captions/
        │   └── {video_id}.pkl
        ├── transcripts/
        │   └── {video_id}.pkl
        ├── fusion/
        │   └── {video_id}.pkl
        ├── frames/
        │   └── {video_id}/
        └── audio/
            └── {video_id}.wav
        """
        # Create stage subfolder
        stage_dir = self.cache_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Return path in subfolder (no need to repeat stage_name in filename)
        return stage_dir / f"{video_id}.{extension}"
