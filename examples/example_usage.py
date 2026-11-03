"""Minimal examples using the retained VideoQAPipeline API."""

from pathlib import Path

from videoqa import PipelineConfig, VideoQAPipeline


def textualize(video_path: Path, configuration: str = "c1"):
    """Generate and store one of the five paper-defined representations."""
    config = PipelineConfig.from_yaml(Path("configs") / f"{configuration}.yaml")
    pipeline = VideoQAPipeline(config)
    return pipeline.process_video(video_path, video_path.stem)


def ask(video_path: Path, question: str, configuration: str = "c1") -> str:
    """Textualize a video and issue one question to GPT-4o-mini."""
    config = PipelineConfig.from_yaml(Path("configs") / f"{configuration}.yaml")
    pipeline = VideoQAPipeline(config)
    return pipeline.run_single_qa(video_path, question)


if __name__ == "__main__":
    processed = textualize(Path("lecture.mp4"), configuration="c5")
    print(processed.fused_content.timeline_text)
