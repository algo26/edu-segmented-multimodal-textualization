"""
Configuration management for the video QA pipeline.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Video preprocessing configuration."""
    fps: float = 0.5  # Paper protocol: one frame every two seconds
    max_frames: Optional[int] = None  # Retained for compatibility; must remain null
    frame_size: tuple = (224, 224)  # Resize frames to this size
    keyframe_method: str = "uniform"
    sampling_strategy: str = "uniform"  # uniform, transcript_segments
    transcript_segment_medium_seconds: float = 8.0
    transcript_segment_long_seconds: float = 16.0
    silence_gap_seconds: float = 8.0
    cache_frames: bool = False


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    extract_audio: bool = True
    cache_audio: bool = False


@dataclass
class CaptioningConfig:
    """Image captioning configuration."""
    model_name: str = "Salesforce/blip-image-captioning-base"
    batch_size: int = 1
    max_length: int = 50
    num_beams: int = 1
    device: str = "auto"
    min_length: int = 0
    cache_captions: bool = False

@dataclass
class ASRConfig:
    """Automatic speech recognition configuration."""
    model_name: str = "base"  # tiny, base, small, medium, large
    device: str = "cpu"
    compute_type: str = "int8"  # int8, float16, float32
    language: Optional[str] = None  # Auto-detect if None
    beam_size: int = 5
    cache_transcripts: bool = False


@dataclass
class FusionConfig:
    """Fusion strategy configuration."""
    timeline_window_seconds: float = 30.0  # Window for timeline fusion
    windowing_strategy: str = "fixed"  # fixed, transcript_segments
    cache_fusion: bool = False


@dataclass
class ReasoningConfig:
    """LLM reasoning configuration."""
    provider: str = "openai"  # openai, local, anthropic
    model_name: str = "gpt-4o-mini-2024-07-18"
    api_key: Optional[str] = None
    context_source: str = "multimodal"
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass
class CacheConfig:
    """Runtime artifact location; stage-cache reuse is disabled in the paper profile."""
    enabled: bool = False
    cache_dir: Path = Path("./artifacts")
    cache_frames: bool = False
    cache_audio: bool = False
    cache_captions: bool = False
    cache_transcripts: bool = False
    cache_fusion: bool = False


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    device: str = "auto"  # Global device override
    verbose: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        cache_section = config_dict.get('cache', {})

        # Handle nested configs
        video_dict = {**config_dict.get('video', {})}
        audio_dict = {**config_dict.get('audio', {})}
        captioning_dict = {**config_dict.get('captioning', {})}
        asr_dict = {**config_dict.get('asr', {})}
        fusion_dict = {**config_dict.get('fusion', {})}

        if 'cache_frames' in cache_section:
            video_dict.setdefault('cache_frames', cache_section['cache_frames'])
        if 'cache_audio' in cache_section:
            audio_dict.setdefault('cache_audio', cache_section['cache_audio'])
        if 'cache_captions' in cache_section:
            captioning_dict.setdefault('cache_captions', cache_section['cache_captions'])
        if 'cache_transcripts' in cache_section:
            asr_dict.setdefault('cache_transcripts', cache_section['cache_transcripts'])
        if 'cache_fusion' in cache_section:
            fusion_dict.setdefault('cache_fusion', cache_section['cache_fusion'])

        video_config = VideoConfig(**video_dict)
        audio_config = AudioConfig(**audio_dict)
        captioning_config = CaptioningConfig(**captioning_dict)
        asr_config = ASRConfig(**asr_dict)
        fusion_config = FusionConfig(**fusion_dict)
        reasoning_config = ReasoningConfig(**config_dict.get('reasoning', {}))
        cache_config = CacheConfig(**cache_section)

        # Handle Path conversion
        if 'cache_dir' in config_dict.get('cache', {}):
            cache_config.cache_dir = Path(config_dict['cache']['cache_dir'])

        config = cls(
            video=video_config,
            audio=audio_config,
            captioning=captioning_config,
            asr=asr_config,
            fusion=fusion_config,
            reasoning=reasoning_config,
            cache=cache_config,
            device=config_dict.get('device', 'auto'),
            verbose=config_dict.get('verbose', True)
        )
        config.validate_method()
        return config

    def validate_method(self) -> None:
        """Reject configuration changes that would alter the paper methodology."""
        expected = {
            "video.fps": (self.video.fps, 0.5),
            "video.max_frames": (self.video.max_frames, None),
            "video.keyframe_method": (self.video.keyframe_method, "uniform"),
            "video.transcript_segment_medium_seconds": (
                self.video.transcript_segment_medium_seconds,
                8.0,
            ),
            "video.transcript_segment_long_seconds": (
                self.video.transcript_segment_long_seconds,
                16.0,
            ),
            "video.silence_gap_seconds": (self.video.silence_gap_seconds, 8.0),
            "captioning.model_name": (
                self.captioning.model_name,
                "Salesforce/blip-image-captioning-base",
            ),
            "captioning.max_length": (self.captioning.max_length, 50),
            "captioning.num_beams": (self.captioning.num_beams, 1),
            "captioning.min_length": (self.captioning.min_length, 0),
            "asr.model_name": (self.asr.model_name, "base"),
            "asr.device": (self.asr.device, "cpu"),
            "asr.compute_type": (self.asr.compute_type, "int8"),
            "fusion.timeline_window_seconds": (
                self.fusion.timeline_window_seconds,
                30.0,
            ),
            "reasoning.provider": (self.reasoning.provider, "openai"),
            "reasoning.model_name": (
                self.reasoning.model_name,
                "gpt-4o-mini-2024-07-18",
            ),
            "reasoning.temperature": (self.reasoning.temperature, 0.0),
            "cache.enabled": (self.cache.enabled, False),
            "cache.cache_frames": (self.cache.cache_frames, False),
            "cache.cache_audio": (self.cache.cache_audio, False),
            "cache.cache_captions": (self.cache.cache_captions, False),
            "cache.cache_transcripts": (self.cache.cache_transcripts, False),
            "cache.cache_fusion": (self.cache.cache_fusion, False),
        }
        mismatches = [
            f"{name} must be {wanted!r}, got {actual!r}"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]

        combination = (
            self.video.sampling_strategy,
            self.fusion.windowing_strategy,
            self.reasoning.context_source,
        )
        allowed_combinations = {
            ("uniform", "fixed", "multimodal"),
            ("uniform", "fixed", "captions_only"),
            ("none", "fixed", "transcripts_only"),
            ("transcript_segments", "fixed", "multimodal"),
            ("transcript_segments", "transcript_segments", "multimodal"),
        }
        if combination not in allowed_combinations:
            mismatches.append(
                "sampling, windows, and context must match one of configurations C1-C5; "
                f"got {combination!r}"
            )
        if mismatches:
            raise ValueError("Configuration differs from the paper method:\n- " + "\n- ".join(mismatches))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'video': self.video.__dict__,
            'audio': self.audio.__dict__,
            'captioning': self.captioning.__dict__,
            'asr': self.asr.__dict__,
            'fusion': self.fusion.__dict__,
            'reasoning': self.reasoning.__dict__,
            'cache': {**self.cache.__dict__, 'cache_dir': str(self.cache.cache_dir)},
            'device': self.device,
            'verbose': self.verbose
        }

    def save_yaml(self, yaml_path: Path):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
