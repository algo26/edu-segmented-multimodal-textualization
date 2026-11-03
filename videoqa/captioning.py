"""
Image captioning module using pre-trained models.
"""
import logging
import pickle
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from .common_types import Caption, FrameData, StageCache
from .config import CaptioningConfig
from .device_utils import detect_device

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """Generates captions for video frames using pre-trained models."""

    def __init__(self, config: CaptioningConfig, cache: StageCache):
        self.config = config
        self.cache = cache
        self.device = detect_device(config.device)
        self.hf_cache_dir = self.cache.cache_dir / "huggingface"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model weights only when a caption must be generated.
        self.processor = None
        self.model = None

    def _ensure_model_loaded(self):
        """Load the captioning model on first use."""
        if self.processor is None or self.model is None:
            self._load_model()

    def _load_model(self):
        """Load the captioning model."""
        try:
            logger.info(f"Loading captioning model: {self.config.model_name}")

            model_source = self._resolve_model_source()
            load_kwargs = {}
            if model_source == self.config.model_name:
                load_kwargs["cache_dir"] = self.hf_cache_dir
            else:
                load_kwargs["local_files_only"] = True

            self.processor = self._hf_load_with_retry(
                BlipProcessor.from_pretrained,
                model_source,
                **load_kwargs,
            )
            self.model = self._hf_load_with_retry(
                BlipForConditionalGeneration.from_pretrained,
                model_source,
                **load_kwargs,
            )

            # Move to device
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            elif self.device == "mps":
                # MPS support
                self.model = self.model.to(self.device)
            else:
                self.model = self.model.to("cpu")

            self.model.eval()
            logger.info(f"Model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _resolve_model_source(self) -> str:
        """Prefer a complete local Hugging Face snapshot when one is cached."""
        configured = Path(self.config.model_name)
        if configured.exists():
            return str(configured.resolve())

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return self.config.model_name

        # First inspect the configured stage cache, then a sibling canonical
        # repository cache, and finally Hugging Face's standard user cache.
        cache_candidates = [
            self.hf_cache_dir,
            self.cache.cache_dir.parent / "huggingface",
            None,
        ]
        for cache_dir in cache_candidates:
            try:
                kwargs = {"local_files_only": True}
                if cache_dir is not None:
                    kwargs["cache_dir"] = cache_dir
                snapshot = snapshot_download(self.config.model_name, **kwargs)
                logger.info("Using local caption-model snapshot: %s", snapshot)
                return snapshot
            except (OSError, FileNotFoundError):
                continue

        return self.config.model_name

    def _hf_load_with_retry(self, load_fn, model_name: str, **kwargs):
        """Load a Hugging Face component without destructively clearing caches."""
        return load_fn(model_name, **kwargs)

    def caption_frames(self, frames: List[FrameData], video_id: str) -> List[Caption]:
        """
        Generate captions for a list of frames.

        Args:
            frames: List of frames to caption
            video_id: Video identifier for caching

        Returns:
            List of Caption objects
        """
        # Check cache
        cache_path = self.cache.get_path(video_id, "captions", "pkl")
        if self.cache.enabled and self.config.cache_captions and cache_path.exists():
            logger.info(f"Loading captions from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        logger.info(f"Generating captions for {len(frames)} frames")

        captions = []

        # Process in batches
        batch_size = self.config.batch_size

        for i in tqdm(range(0, len(frames), batch_size), desc="Captioning frames"):
            batch_frames = frames[i:i + batch_size]
            batch_captions = self._process_batch(batch_frames)
            captions.extend(batch_captions)

        # Cache captions
        if self.cache.enabled and self.config.cache_captions:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(captions, f)

        logger.info(f"Generated {len(captions)} captions")
        return captions

    def _process_batch(self, frames: List[FrameData]) -> List[Caption]:
        """Process a batch of frames."""
        self._ensure_model_loaded()

        # Load images
        images = []
        for frame in frames:
            img = Image.open(frame.image_path).convert('RGB')
            images.append(img)

        # Prepare inputs
        inputs = self.processor(images=images, return_tensors="pt", padding=True)

        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate captions
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_beams=1,
                min_length=self.config.min_length,
                do_sample=False,
            )

        # Decode captions
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Create Caption objects
        captions = []
        for frame, text in zip(frames, generated_texts):
            captions.append(Caption(
                timestamp=(
                    frame.requested_timestamp
                    if frame.requested_timestamp is not None
                    else frame.timestamp
                ),
                frame_idx=frame.frame_idx,
                text=text.strip(),
                decoded_frame_timestamp=frame.timestamp,
            ))

        return captions

    def caption_single_image(self, image_path: Path) -> str:
        """
        Generate caption for a single image.
        Utility method for testing.
        """
        self._ensure_model_loaded()
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_beams=1,
                min_length=self.config.min_length,
                do_sample=False,
            )

        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()
