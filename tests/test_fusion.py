from videoqa.common_types import Caption, StageCache, TranscriptSegment
from videoqa.config import FusionConfig
from videoqa.fusion import MultimodalFusion


def test_fixed_windows_use_caption_timestamp_and_speech_midpoint(tmp_path):
    fusion = MultimodalFusion(
        FusionConfig(windowing_strategy="fixed", cache_fusion=False),
        StageCache(tmp_path, enabled=False),
    )
    result = fusion.fuse(
        [Caption(2.0, 1, "slide one"), Caption(32.0, 2, "slide two")],
        [TranscriptSegment(28.0, 32.0, "boundary speech")],
        "video",
        video_duration=60.0,
    )
    first, second = result.timeline_text.split(" | ")
    assert "slide one" in first
    assert "boundary speech" not in first
    assert "slide two" in second
    assert "boundary speech" in second


def test_variable_windows_preserve_raw_entries(tmp_path):
    fusion = MultimodalFusion(
        FusionConfig(windowing_strategy="transcript_segments", cache_fusion=False),
        StageCache(tmp_path, enabled=False),
    )
    result = fusion.fuse(
        [Caption(2.0, 1, "first visual"), Caption(8.0, 2, "gap visual")],
        [
            TranscriptSegment(0.0, 4.0, "first speech"),
            TranscriptSegment(12.0, 16.0, "second speech"),
        ],
        "video",
        video_duration=16.0,
    )
    blocks = result.timeline_text.split(" | ")
    assert len(blocks) == 3
    assert "first visual" in blocks[0]
    assert "gap visual" in blocks[1]
    assert "first speech" in blocks[0]
    assert "second speech" in blocks[2]
