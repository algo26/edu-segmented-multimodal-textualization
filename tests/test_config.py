from pathlib import Path

import pytest

from videoqa.config import PipelineConfig

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name", ["c1", "c2", "c3", "c4", "c5"])
def test_paper_configurations_load(name):
    config = PipelineConfig.from_yaml(ROOT / "configs" / f"{name}.yaml")
    assert config.cache.enabled is False
    assert config.video.cache_frames is False
    assert config.audio.cache_audio is False
    assert config.captioning.cache_captions is False
    assert config.asr.cache_transcripts is False
    assert config.fusion.cache_fusion is False


def test_frame_cap_is_rejected():
    config = PipelineConfig.from_yaml(ROOT / "configs" / "c1.yaml")
    config.video.max_frames = 600
    with pytest.raises(ValueError, match="frame cap|video.max_frames"):
        config.validate_method()


def test_stage_cache_reuse_is_rejected():
    config = PipelineConfig.from_yaml(ROOT / "configs" / "c1.yaml")
    config.cache.enabled = True
    with pytest.raises(ValueError, match="cache.enabled"):
        config.validate_method()


def test_only_reported_configuration_combinations_are_allowed():
    config = PipelineConfig.from_yaml(ROOT / "configs" / "c1.yaml")
    config.fusion.windowing_strategy = "transcript_segments"
    with pytest.raises(ValueError, match="C1-C5"):
        config.validate_method()
