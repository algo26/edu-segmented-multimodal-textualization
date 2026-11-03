from pathlib import Path

import cv2
import numpy as np
import pytest

from videoqa.captioning import ImageCaptioner
from videoqa.common_types import FrameData, StageCache, TranscriptSegment
from videoqa.config import AudioConfig, VideoConfig
from videoqa.keyframes import KeyframeSelector
from videoqa.video_io import VideoProcessor


def processor(tmp_path: Path) -> VideoProcessor:
    return VideoProcessor(
        VideoConfig(),
        AudioConfig(),
        StageCache(tmp_path, enabled=False),
    )


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0.0, 4.0, [2.0]),
        (0.0, 8.0, pytest.approx([8.0 / 3.0, 16.0 / 3.0])),
        (0.0, 16.0, pytest.approx([4.0, 8.0, 12.0])),
    ],
)
def test_segment_frame_rule(tmp_path, start, end, expected):
    assert processor(tmp_path)._segment_sample_timestamps(start, end) == expected


def test_speech_anchors_include_only_qualifying_internal_silence(tmp_path):
    transcripts = [
        TranscriptSegment(5.0, 9.0, "first"),
        TranscriptSegment(17.0, 21.0, "second"),
    ]
    timestamps = processor(tmp_path).build_transcript_anchor_timestamps(transcripts, 30.0)
    assert timestamps == pytest.approx([7.0, 35.0 / 3.0, 43.0 / 3.0, 19.0])
    assert all(timestamp > 5.0 for timestamp in timestamps)
    assert all(timestamp < 21.0 for timestamp in timestamps)


def test_keyframe_boundary_never_caps_frames():
    frames = [FrameData(float(index), index) for index in range(601)]
    assert KeyframeSelector().select_keyframes(frames) == frames
    with pytest.raises(ValueError, match="frame cap"):
        KeyframeSelector().select_keyframes(frames, max_keyframes=600)


def test_uniform_sampling_covers_full_duration_without_cap(tmp_path, monkeypatch):
    video_processor = processor(tmp_path)
    captured = {}
    monkeypatch.setattr(video_processor, "get_video_duration", lambda _: 6.1)

    def capture_requests(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(video_processor, "extract_frames_at_timestamps", capture_requests)
    video_processor.extract_frames(Path("lecture.mp4"), "lecture")
    assert captured["timestamps"] == [0.0, 2.0, 4.0, 6.0]


def test_nearest_caption_reuse_api_was_removed():
    assert not hasattr(ImageCaptioner, "select_cached_captions")
    assert not hasattr(ImageCaptioner, "select_nearest_captions")


def test_decoding_preserves_anchor_and_actual_frame_timestamp(tmp_path, monkeypatch):
    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FPS:
                return 2.0
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 10.0
            return 0.0

        def set(self, property_id, value):
            return True

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            return None

    video_processor = processor(tmp_path)
    monkeypatch.setattr("videoqa.video_io.cv2.VideoCapture", lambda _: FakeCapture())
    monkeypatch.setattr(
        video_processor,
        "_save_frame_image",
        lambda **_: tmp_path / "frame.jpg",
    )

    frames = video_processor.extract_frames_at_timestamps(
        Path("lecture.mp4"),
        "lecture",
        [0.26],
        "frames_transcript_segments",
    )

    assert frames[0].requested_timestamp == pytest.approx(0.26)
    assert frames[0].timestamp == pytest.approx(0.5)
    assert frames[0].frame_idx == 1
