# Methods traceability

| Paper requirement | Existing-code implementation |
|---|---|
| Uniform sampling at 0.5 fps with no cap | `videoqa.video_io.VideoProcessor.extract_frames` |
| Speech-segment 1/2/3 frame rule at 8 s and 16 s | `VideoProcessor._segment_sample_timestamps` |
| Evenly spaced interior timestamps | `VideoProcessor._segment_sample_timestamps` |
| Reference frames for internal silent gaps of at least 8 s | `VideoProcessor.build_transcript_anchor_timestamps` |
| Decode and caption the requested frames directly, preserving requested and decoded timestamps | `VideoProcessor.extract_frames_at_timestamps`, `FrameData`, `Caption`, and `ImageCaptioner.caption_frames` |
| BLIP-base, greedy decoding, maximum length 50 | `videoqa.captioning` and `CaptioningConfig` |
| Whisper-base on CPU with int8 quantization | `videoqa.asr` and `ASRConfig` |
| Fixed 30-second windows; speech assigned by midpoint | `MultimodalFusion._fixed_windows` and `_render` |
| Variable Whisper-segment and silent-gap windows | `MultimodalFusion._variable_windows` |
| No summarization, compression, paraphrase, or deduplication | `MultimodalFusion.fuse` |
| C1-C5 only | `PipelineConfig.validate_method` and `configs/c1.yaml` through `configs/c5.yaml` |
| Shared prompt and `gpt-4o-mini-2024-07-18` at temperature zero | `videoqa.reasoning` |
| Generate representations before benchmark queries | `BenchmarkRunner._prepare_representations` |

Nearest-neighbour caption selection from the legacy uniform cache has been removed. Canonical C1-C5
runs disable stage-cache reuse so settings cannot resolve to stale perception output. Runtime model
weights, frames, audio, perception outputs, representations, benchmark runs, caches, and databases
are excluded from version control.
