"""Keyframe selection retained as the pipeline's sampling boundary."""

from typing import List

from .common_types import FrameData


class KeyframeSelector:
    """Return every frame requested by the configured paper sampling regime.

    Frame positions are determined during extraction.  This class remains in the
    original architecture, but it no longer applies a second cap or an adaptive
    scene-selection strategy.
    """

    def __init__(self, method: str = "uniform"):
        if method != "uniform":
            raise ValueError("The paper method supports only uniform keyframe selection")
        self.method = method

    def select_keyframes(
        self,
        frames: List[FrameData],
        max_keyframes: int | None = None,
    ) -> List[FrameData]:
        if max_keyframes is not None:
            raise ValueError("The paper method does not impose a frame cap")
        return list(frames)
