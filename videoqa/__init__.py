"""
VideoQA Pipeline - Modular Multimodal Video Understanding System
"""

from .benchmarking import (
    BenchmarkRunner,
    MLVUDatasetAdapter,
    MultipleChoiceSample,
    SummaryDatasetAdapter,
    SummaryJudge,
    SummarySample,
)
from .common_types import (
    Caption,
    FrameData,
    FusedContent,
    PredictionResult,
    ProcessedVideo,
    TranscriptSegment,
    VideoQASample,
)
from .config import PipelineConfig
from .evaluation import DatasetAdapter, Evaluator
from .pipeline import VideoQAPipeline

__version__ = "0.1.0"

__all__ = [
    "VideoQAPipeline",
    "PipelineConfig",
    "VideoQASample",
    "PredictionResult",
    "ProcessedVideo",
    "FrameData",
    "Caption",
    "TranscriptSegment",
    "FusedContent",
    "Evaluator",
    "DatasetAdapter",
    "BenchmarkRunner",
    "MLVUDatasetAdapter",
    "MultipleChoiceSample",
    "SummaryDatasetAdapter",
    "SummaryJudge",
    "SummarySample",
]
