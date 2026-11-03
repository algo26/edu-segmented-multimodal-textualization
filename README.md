# Segmented Multimodal Textualization

Reference implementation of the paper's training-free pipeline for converting long-form video into
an inspectable segmented text representation before language-model reasoning. This repository keeps
the original `videoqa` architecture and narrows its behavior to the published method.

## Method

The pipeline retains its existing stages:

1. `VideoProcessor` extracts audio and only the requested frames.
2. `SpeechRecognizer` runs frozen Whisper-base with int8 quantization, while `ImageCaptioner` runs
   frozen BLIP-base with greedy decoding and a maximum caption length of 50 tokens.
3. `MultimodalFusion` groups the raw timestamped outputs without summarizing, deduplicating,
   compressing, or paraphrasing them.
4. `OpenAIReasoner` inserts the representation and query into the paper's shared prompt and makes
   one call to `gpt-4o-mini-2024-07-18` at temperature zero.

Uniform sampling covers the complete video at 0.5 fps without a frame cap. Speech-anchored sampling
uses Whisper segment boundaries and requests one interior frame below 8 seconds, two from 8 to less
than 16 seconds, and three from 16 seconds upward. Inter-segment silent gaps of at least 8 seconds
receive frames under the same rule. Each caption retains its requested sampling anchor and the actual
decoded-frame timestamp; there is no nearest-neighbour caption reuse or timestamp retiming.

Fixed fusion uses contiguous 30-second windows and assigns speech segments by midpoint. Variable
fusion uses each Whisper segment as a window and adds qualifying silent-gap windows. Blocks are
joined with `|` in chronological order.

## Configurations

| Config | Frame sampling | Windows | Context |
|---|---|---|---|
| `c1.yaml` | Uniform 0.5 fps | Fixed 30 s | Captions and transcript |
| `c2.yaml` | Uniform 0.5 fps | Fixed 30 s | Captions only |
| `c3.yaml` | None | Fixed 30 s | Transcript only |
| `c4.yaml` | Speech-anchored | Fixed 30 s | Captions and transcript |
| `c5.yaml` | Speech-anchored | Whisper segments | Captions and transcript |

The configuration loader rejects settings outside these five combinations or changes to the paper's
model, sampling, window, and reasoning constants.

## Installation

Python 3.12 and FFmpeg are required.

```bash
brew install ffmpeg
uv sync --extra dev
```

Set `OPENAI_API_KEY` only for the reasoning stage. Perception and representation generation do not
require it.

## Usage

Validate setup without loading model weights:

```bash
uv run python run.py test --config configs/c1.yaml
```

Process one video and answer one question:

```bash
uv run python run.py single \
  --config configs/c1.yaml \
  --video /path/to/video.mp4 \
  --question "What is the main topic?"
```

Run Topic Reasoning and Needle Question Answering on MLVU:

```bash
uv run python run.py benchmark-mlvu \
  --config configs/c1.yaml \
  --dataset-root /path/to/MLVU \
  --question-types topic_reasoning,needle_qa
```

Run Video Summarization with the official-style GPT-4 judge prompt retained in
`videoqa/benchmarking.py`:

```bash
uv run python run.py benchmark-summary \
  --config configs/c1.yaml \
  --annotations /path/to/test_vs_gt.json \
  --videos-dir /path/to/videos
```

The benchmark runner textualizes every unique video before issuing the first query, then reuses the
stored deterministic representation for all questions concerning that video.

## Runtime artifacts

Downloaded model weights, decoded frames, audio, transcripts, captions, fused representations,
benchmark predictions, checkpoints, datasets, caches, databases, and media are runtime artifacts.
They are excluded from version control by `.gitignore`; no such files are included in this repository.
The canonical C1-C5 configurations disable stage-cache reuse to prevent stale outputs after a change
to a source video or any output-affecting setting.

## Project layout

```text
videoqa/
  asr.py             Whisper transcription
  benchmarking.py    MLVU adapters and evaluation runner
  captioning.py       BLIP frame captioning
  config.py           paper-locked configuration schema
  fusion.py           fixed and variable deterministic fusion
  keyframes.py        retained uncapped selection boundary
  pipeline.py         existing pipeline orchestration
  reasoning.py        shared prompt and GPT-4o-mini call
  video_io.py         uniform and speech-anchored extraction
configs/              C1-C5 configurations
tests/                method-level regression tests
run.py                command-line interface
```

## Development

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

The source repository's exploratory notebooks were not published because they depend on old cached
caption strategies and alternative model/fusion variants rather than the final method.
