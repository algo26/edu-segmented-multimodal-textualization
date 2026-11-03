"""
Main video QA pipeline orchestration.
"""
import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .asr import SpeechRecognizer
from .captioning import ImageCaptioner
from .common_types import PredictionResult, ProcessedVideo, StageCache, VideoQASample
from .config import PipelineConfig
from .device_utils import get_device_info
from .evaluation import Evaluator
from .fusion import MultimodalFusion
from .keyframes import KeyframeSelector
from .reasoning import get_reasoner
from .video_io import VideoProcessor

logger = logging.getLogger(__name__)


class VideoQAPipeline:
    """
    End-to-end video question answering pipeline.

    Stages:
    1. Video preprocessing (frame extraction, audio extraction)
    2. Visual understanding (frame captioning)
    3. Audio understanding (speech recognition)
    4. Multimodal fusion
    5. Question answering with LLM
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        config.validate_method()
        self.config = config

        # Setup logging
        log_level = logging.INFO if config.verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Log device info
        device_info = get_device_info()
        logger.info(f"Device info: {device_info}")

        # Create cache directory
        self.config.cache.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache
        self.cache = StageCache(
            cache_dir=self.config.cache.cache_dir,
            enabled=self.config.cache.enabled
        )

        # Initialize pipeline components
        self._init_components()

    def _init_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")

        # Video processor
        self.video_processor = VideoProcessor(
            self.config.video,
            self.config.audio,
            self.cache
        )

        # Keyframe selector
        self.keyframe_selector = KeyframeSelector(
            method=self.config.video.keyframe_method
        )

        # Image captioner
        self.captioner = ImageCaptioner(
            self.config.captioning,
            self.cache
        )

        # Speech recognizer
        self.asr = SpeechRecognizer(
            self.config.asr,
            self.cache
        )

        # Multimodal fusion
        self.fusion = MultimodalFusion(
            self.config.fusion,
            self.cache
        )

        # The external reasoner is loaded only when a question is issued. This
        # keeps representation generation independent of API credentials.
        self.reasoner = None

        # Evaluator
        self.evaluator = Evaluator()

        logger.info("Pipeline components initialized")

    def process_video(self, video_path: Path, video_id: str) -> ProcessedVideo:
        """
        Process a single video through all stages.

        Args:
            video_path: Path to video file
            video_id: Unique video identifier

        Returns:
            ProcessedVideo object with all extracted information
        """
        logger.info(f"Processing video: {video_id}")
        processing_cache_id = self._build_processing_cache_id(video_id)

        transcripts = []
        captions = []
        frames = []
        keyframes = []
        sampling_strategy = (self.config.video.sampling_strategy or "uniform").lower()
        context_source = self.config.reasoning.context_source
        needs_transcript = context_source != "captions_only" or sampling_strategy == "transcript_segments"
        needs_captions = context_source != "transcripts_only"

        # Stage 1: Extract audio
        audio_path = None
        if needs_transcript:
            logger.info("Stage 1: Extracting audio...")
            audio_path = self.video_processor.extract_audio(video_path, video_id)

        # Stage 2: Transcribe audio
        if needs_transcript and audio_path:
            logger.info("Stage 2: Transcribing audio...")
            transcripts = self.asr.transcribe_audio(audio_path, video_id)

        video_duration = self.video_processor.get_video_duration(video_path)

        # Stage 3: Extract only the requested frames.
        if needs_captions and sampling_strategy == "transcript_segments":
            logger.info("Stage 3: Extracting speech-anchored frames...")
            frames = self.video_processor.extract_transcript_anchor_frames(
                video_path=video_path,
                video_id=processing_cache_id,
                transcripts=transcripts,
            )
        elif needs_captions and sampling_strategy == "uniform":
            logger.info("Stage 3: Extracting uniform frames...")
            frames = self.video_processor.extract_frames(video_path, video_id)
        if needs_captions:
            keyframes = self.keyframe_selector.select_keyframes(frames)

        # Stage 4: Generate captions
        if needs_captions:
            logger.info("Stage 4: Generating captions at sampled timestamps...")
            captions = self.captioner.caption_frames(keyframes, processing_cache_id)

        # Stage 5: Fuse modalities
        logger.info("Stage 5: Fusing modalities...")
        fused_content = self.fusion.fuse(
            captions,
            transcripts,
            processing_cache_id,
            video_duration=video_duration,
        )

        processed_video = ProcessedVideo(
            video_id=video_id,
            frames=keyframes,
            audio_segments=[],
            captions=captions,
            transcripts=transcripts,
            fused_content=fused_content,
            metadata={
                "sampling_strategy": sampling_strategy,
                "num_frames": len(frames),
                "num_keyframes": len(keyframes),
                "num_captions": len(captions),
                "num_transcripts": len(transcripts)
            }
        )

        logger.info(f"Video processing complete: {video_id}")
        return processed_video

    def _build_processing_cache_id(self, video_id: str) -> str:
        """Separate deterministic artifacts for each sampling/window policy."""
        sampling_strategy = (self.config.video.sampling_strategy or "uniform").lower()
        return (
            f"{video_id}__sampling-{sampling_strategy}"
            f"__windows-{self.config.fusion.windowing_strategy}"
            f"__context-{self.config.reasoning.context_source}"
        )

    def answer_question(
        self,
        processed_video: ProcessedVideo,
        question: str,
        question_id: Optional[str] = None
    ) -> PredictionResult:
        """
        Answer a question about a processed video.

        Args:
            processed_video: Processed video data
            question: Question to answer
            question_id: Optional question identifier

        Returns:
            PredictionResult with answer
        """
        logger.info(f"Answering question for video: {processed_video.video_id}")

        # Get answer from LLM
        if self.reasoner is None:
            self.reasoner = get_reasoner(self.config.reasoning)
        answer = self.reasoner.answer_question(
            processed_video.fused_content,
            question
        )

        result = PredictionResult(
            question_id=question_id or f"{processed_video.video_id}_q",
            video_id=processed_video.video_id,
            question=question,
            predicted_answer=answer,
            metadata=processed_video.metadata
        )

        return result

    def run_qa_batch(
        self,
        samples: List[VideoQASample],
        output_path: Optional[Path] = None
    ) -> List[PredictionResult]:
        """
        Run QA on a batch of video samples.

        Args:
            samples: List of VideoQASample objects
            output_path: Optional path to save predictions

        Returns:
            List of PredictionResult objects
        """
        logger.info(f"Running QA on {len(samples)} samples")

        predictions = []

        for sample in tqdm(samples, desc="Processing videos"):
            try:
                # Process video
                processed_video = self.process_video(
                    sample.video_path,
                    sample.video_id
                )

                # Answer question
                result = self.answer_question(
                    processed_video,
                    sample.question,
                    question_id=sample.question_id
                )

                # Add ground truth if available
                result.ground_truth = sample.answer

                predictions.append(result)

            except Exception as e:
                logger.error(f"Error processing {sample.video_id}: {e}")
                # Add error result
                predictions.append(PredictionResult(
                    question_id=sample.question_id or f"{sample.video_id}_q",
                    video_id=sample.video_id,
                    question=sample.question,
                    predicted_answer=f"Error: {str(e)}",
                    ground_truth=sample.answer
                ))

        # Evaluate
        metrics = self.evaluator.evaluate(predictions)
        logger.info(f"Evaluation metrics: {metrics}")

        # Save predictions
        if output_path:
            self.evaluator.save_predictions(predictions, output_path)

        return predictions

    def run_single_qa(
        self,
        video_path: Path,
        question: str,
        video_id: Optional[str] = None
    ) -> str:
        """
        Run QA on a single video (convenience method).

        Args:
            video_path: Path to video file
            question: Question to answer
            video_id: Optional video identifier

        Returns:
            Answer string
        """
        if video_id is None:
            video_id = video_path.stem

        # Process video
        processed_video = self.process_video(video_path, video_id)

        # Answer question
        result = self.answer_question(processed_video, question)

        return result.predicted_answer
