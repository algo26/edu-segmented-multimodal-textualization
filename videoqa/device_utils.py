"""
Device detection and management utilities.
"""
import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DeviceType = Literal["cpu", "cuda", "mps"]


def detect_device(prefer: str = "auto") -> str:
    """
    Detect the best available device.

    Args:
        prefer: "auto", "cpu", "cuda", or "mps"

    Returns:
        Device string: "cpu", "cuda", or "mps"
    """
    if prefer != "auto":
        if prefer == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU/MPS")
        elif prefer == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
        else:
            return prefer

    # Auto-detect
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple MPS (Metal) detected")
    else:
        device = "cpu"
        logger.info("Using CPU")

    return device


def get_device_info() -> dict:
    """Get detailed device information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def get_optimal_batch_size(device: str, model_size: str = "base") -> int:
    """
    Get optimal batch size based on device and model size.

    Args:
        device: "cpu", "cuda", or "mps"
        model_size: "small", "base", "large"

    Returns:
        Recommended batch size
    """
    # Default batch sizes
    batch_sizes = {
        "cpu": {"small": 1, "base": 1, "large": 1},
        "mps": {"small": 4, "base": 2, "large": 1},
        "cuda": {"small": 16, "base": 8, "large": 4},
    }

    return batch_sizes.get(device, {}).get(model_size, 1)


def move_to_device(obj, device: str):
    """Move PyTorch object to device safely."""
    if hasattr(obj, 'to'):
        return obj.to(device)
    return obj
