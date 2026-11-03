"""
Benchmark runners and dataset adapters for structured evaluation.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from tqdm import tqdm

from .common_types import PredictionResult
from .evaluation import DatasetAdapter

logger = logging.getLogger(__name__)

CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _normalize_question_type(value: Any) -> str:
    """Return a stable task identifier for CLI filters and annotation labels."""
    question_type = str(value or "unknown").strip()
    question_type = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", question_type)
    question_type = re.sub(r"[\s-]+", "_", question_type)
    question_type = re.sub(r"_+", "_", question_type).strip("_").lower()
    return question_type


OFFICIAL_SUMMARY_JUDGE_PROMPT = """
##TASK DESCRIPTION:
You are required to evaluate the performance of the respondent in the video summarization task based on the standard answer and the respondent's answer. You should provide two scores. The first is the COMPLETENESS score, which should range from 1 to 5. The second is the RELIABILITY score, which should also range from 1 to 5. Below are the criteria for each scoring category:
##COMPLETENESS Scoring Criteria:
The completeness score focuses on whether the summary covers all key points and main information from the video.
Score 1: The summary hardly covers any of the main content or key points of the video.
Score 2: The summary covers some of the main content and key points but misses many.
Score 3: The summary covers most of the main content and key points.
Score 4: The summary is very comprehensive, covering most to nearly all of the main content and key points.
Score 5: The summary completely covers all the main content and key points of the video.
##RELIABILITY Scoring Criteria:
The reliability score evaluates the correctness and clarity of the video summary. It checks for factual errors, misleading statements, and contradictions with the video content. If the respondent's answer includes details that are not present in the standard answer, as long as these details do not conflict with the correct answer and are reasonable, points should not be deducted.
Score 1: Contains multiple factual errors and contradictions; presentation is confusing.
Score 2: Includes several errors and some contradictions; needs clearer presentation.
Score 3: Generally accurate with minor errors; minimal contradictions; reasonably clear presentation.
Score 4: Very accurate with negligible inaccuracies; no contradictions; clear and fluent presentation.
Score 5: Completely accurate with no errors or contradictions; presentation is clear and easy to understand.
----
##INSTRUCTION:
1. Evaluate COMPLETENESS: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence.
2. Evaluate RELIABILITY: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence.
3. Output Scores in JSON Format: Present the scores in JSON format as follows:
{'score_completeness': score_comp, 'score_reliability': score_reli, 'total_score': score_comp + score_reli}
""".strip()


def _load_env_if_available() -> None:
    """Load variables from a local .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


@dataclass
class MultipleChoiceSample:
    """Represents a close-ended benchmark sample."""

    question_id: str
    video_id: str
    video_path: Path
    question: str
    candidates: List[str]
    answer: str
    question_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SummarySample:
    """Represents an open-ended summarization benchmark sample."""

    question_id: str
    video_id: str
    video_path: Path
    question: str
    answer: str
    question_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLVUDatasetAdapter(DatasetAdapter):
    """
    Loads MLVU-style multiple-choice samples from either local files or Hugging Face.

    Expected row shape:
      {
        "video": "AWA-6.mp4",
        "question": "...",
        "candidates": ["..."],
        "answer": "...",
        "question_type": "topic_reasoning"
      }
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        *,
        videos_dir: Optional[Path] = None,
        annotations_path: Optional[Path] = None,
        hf_dataset: Optional[str] = None,
        hf_split: Optional[str] = None,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        question_types: Optional[Sequence[str]] = None,
    ):
        super().__init__(dataset_path or Path("."))
        self.dataset_path = dataset_path
        self.videos_dir = videos_dir or dataset_path
        self.annotations_path = annotations_path
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split
        self.hf_revision = hf_revision
        self.hf_token = hf_token
        self.question_types = (
            {_normalize_question_type(question_type) for question_type in question_types}
            if question_types
            else None
        )

    def load_samples(
        self,
        *,
        limit: Optional[int] = None,
        video_ids: Optional[Sequence[str]] = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> List[MultipleChoiceSample]:
        rows = self._load_rows()
        rows = [row for row in rows if self._row_matches_question_types(row)]
        samples = [self._row_to_sample(row) for row in rows]
        samples = self._filter_samples(samples, video_ids=video_ids)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)

        if limit is not None:
            samples = samples[:limit]

        return samples

    def _row_matches_question_types(self, row: Dict[str, Any]) -> bool:
        """Filter rows before sample conversion so mixed annotation dirs are safe."""
        if not self.question_types:
            return True

        question_type = _normalize_question_type(
            row.get("question_type") or row.get("task_type")
        )
        return question_type in self.question_types

    def _load_rows(self) -> List[Dict[str, Any]]:
        if self.hf_dataset:
            return self._load_rows_from_hf()
        return self._load_rows_from_local()

    def _load_rows_from_hf(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        if not self.hf_split:
            raise ValueError("hf_split is required when loading MLVU from Hugging Face")

        logger.info(
            "Loading benchmark dataset from Hugging Face: %s split=%s revision=%s",
            self.hf_dataset,
            self.hf_split,
            self.hf_revision or "main",
        )
        dataset = load_dataset(
            self.hf_dataset,
            split=self.hf_split,
            revision=self.hf_revision,
            token=self.hf_token,
        )
        return [dict(row) for row in dataset]

    def _load_rows_from_local(self) -> List[Dict[str, Any]]:
        annotation_paths = self._discover_annotation_files()
        if not annotation_paths:
            raise FileNotFoundError(
                "No annotation files found. Provide --annotations or a dataset root containing JSON/JSONL files."
            )

        rows: List[Dict[str, Any]] = []
        for annotation_path in annotation_paths:
            rows.extend(self._load_annotation_file(annotation_path))
        return rows

    def _discover_annotation_files(self) -> List[Path]:
        if self.annotations_path:
            if self.annotations_path.is_file():
                return [self.annotations_path]
            return sorted(self.annotations_path.rglob("*.json")) + sorted(self.annotations_path.rglob("*.jsonl"))

        if not self.dataset_path:
            return []

        return sorted(self.dataset_path.rglob("*.json")) + sorted(self.dataset_path.rglob("*.jsonl"))

    def _load_annotation_file(self, annotation_path: Path) -> List[Dict[str, Any]]:
        if annotation_path.suffix == ".jsonl":
            with open(annotation_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        with open(annotation_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                return payload["data"]
            if isinstance(payload.get("samples"), list):
                return payload["samples"]
        raise ValueError(f"Unsupported annotation format in {annotation_path}")

    def _filter_samples(
        self,
        samples: List[MultipleChoiceSample],
        *,
        video_ids: Optional[Sequence[str]] = None,
    ) -> List[MultipleChoiceSample]:
        target_video_ids = {v.strip() for v in video_ids} if video_ids else None
        filtered: List[MultipleChoiceSample] = []

        for sample in samples:
            if self.question_types and sample.question_type not in self.question_types:
                continue
            if target_video_ids and sample.video_id not in target_video_ids:
                continue
            filtered.append(sample)

        return filtered

    def _row_to_sample(self, row: Dict[str, Any]) -> MultipleChoiceSample:
        raw_video = row.get("video") or row.get("video_name") or row.get("video_id")
        video_name = self._video_name_from_row(raw_video)
        if not video_name:
            raise ValueError(f"Missing video field in row: {row}")

        candidates = list(row.get("candidates") or row.get("options") or [])
        if not candidates:
            raise ValueError(f"Missing candidates/options in row for {video_name}")

        answer = str(row.get("answer", "")).strip()
        question = str(row.get("question", "")).strip()
        question_type = _normalize_question_type(
            row.get("question_type") or row.get("task_type")
        )
        video_path = self._resolve_video_path(video_name, raw_video)

        metadata = {
            k: v
            for k, v in row.items()
            if k not in {"video", "video_name", "video_id", "question", "candidates", "options", "answer", "question_type", "task_type"}
        }
        question_id = str(
            row.get("question_id")
            or row.get("id")
            or f"{Path(video_name).stem}:{question_type}:{_stable_text_id(question)}"
        )

        return MultipleChoiceSample(
            question_id=question_id,
            video_id=video_name,
            video_path=video_path,
            question=question,
            candidates=candidates,
            answer=answer,
            question_type=question_type,
            metadata=metadata,
        )

    def _resolve_video_path(self, video_name: str, raw_video: Any = None) -> Path:
        if isinstance(raw_video, dict):
            raw_path = raw_video.get("path")
            if raw_path:
                candidate = Path(raw_path)
                if candidate.exists():
                    return candidate
            raw_bytes = raw_video.get("bytes")
            if raw_bytes:
                cache_dir = (self.dataset_path or Path(".")) / ".hf_video_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_path = cache_dir / Path(video_name).name
                if not cached_path.exists():
                    cached_path.write_bytes(raw_bytes)
                return cached_path

        direct = Path(video_name)
        if direct.exists():
            return direct

        search_roots = [root for root in [self.videos_dir, self.dataset_path] if root]
        for root in search_roots:
            candidate = root / video_name
            if candidate.exists():
                return candidate
            matches = list(root.rglob(video_name))
            if matches:
                return matches[0]

        if self.hf_dataset:
            return Path(video_name)
        raise FileNotFoundError(f"Could not resolve video path for {video_name}")

    @staticmethod
    def _video_name_from_row(raw_video: Any) -> str:
        if isinstance(raw_video, dict):
            raw_path = raw_video.get("path")
            if raw_path:
                return Path(raw_path).name
            filename = raw_video.get("filename")
            if filename:
                return str(filename)
        if raw_video is None:
            return ""
        return os.fspath(raw_video) if hasattr(raw_video, "__fspath__") else str(raw_video)


class SummaryDatasetAdapter(MLVUDatasetAdapter):
    """
    Loads summarization benchmark samples from local files or Hugging Face.

    Expected row shape:
      {
        "video": "test_AWC-4.mp4",
        "question": "Please summarize the main content of this video.",
        "answer": "...",
        "question_type": "summary"
      }
    """

    def load_samples(
        self,
        *,
        limit: Optional[int] = None,
        video_ids: Optional[Sequence[str]] = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> List[SummarySample]:
        rows = self._load_rows()
        samples = [self._row_to_sample(row) for row in rows]
        samples = self._filter_samples(samples, video_ids=video_ids)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)

        if limit is not None:
            samples = samples[:limit]

        return samples

    def _row_to_sample(self, row: Dict[str, Any]) -> SummarySample:
        raw_video = row.get("video") or row.get("video_name") or row.get("video_id")
        video_name = self._video_name_from_row(raw_video)
        if not video_name:
            raise ValueError(f"Missing video field in row: {row}")

        answer = str(row.get("answer", "")).strip()
        question = str(row.get("question", "")).strip()
        question_type = _normalize_question_type(
            row.get("question_type") or row.get("task_type") or "summary"
        )
        if not question:
            raise ValueError(f"Missing question in row for {video_name}")
        if not answer:
            raise ValueError(f"Missing answer in row for {video_name}")

        video_path = self._resolve_video_path(video_name, raw_video)
        metadata = {
            k: v
            for k, v in row.items()
            if k not in {"video", "video_name", "video_id", "question", "answer", "question_type", "task_type"}
        }
        question_id = str(
            row.get("question_id")
            or row.get("id")
            or f"{Path(video_name).stem}:{question_type}:{_stable_text_id(question)}"
        )

        return SummarySample(
            question_id=question_id,
            video_id=video_name,
            video_path=video_path,
            question=question,
            answer=answer,
            question_type=question_type,
            metadata=metadata,
        )


class SummaryJudge:
    """GPT-based summary evaluator aligned with the official script."""

    def __init__(
        self,
        *,
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        prompt: str = OFFICIAL_SUMMARY_JUDGE_PROMPT,
    ):
        _load_env_if_available()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI package not installed. Run: pip install openai") from exc

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key not found in config or environment")

        self.client = OpenAI(api_key=resolved_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt = prompt

    def judge(self, *, question: str, answer: str, prediction: str) -> Dict[str, Any]:
        user_prompt = (
            "Please score the respondent's answer according to the steps in the Instructions. "
            "You must end with a JSON dict to store the scores.\n"
            f"Question: {question}\n"
            f"Standard Answer: {answer}\n"
            f"Respondent's Answer: {prediction}"
        )
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response_message = (completion.choices[0].message.content or "").strip()
        scores = _parse_summary_scores(response_message)
        scores["explain"] = response_message
        return scores


class BenchmarkRunner:
    """Runs benchmark evaluations against the current pipeline."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run_mlvu(
        self,
        *,
        samples: Sequence[MultipleChoiceSample],
        run_root: Path,
        benchmark_name: str = "mlvu_mc",
        run_name: Optional[str] = None,
        config_path: Optional[Path] = None,
        dataset_source: Optional[Dict[str, Any]] = None,
        selection: Optional[Dict[str, Any]] = None,
        checkpoint_every: int = 5,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        run_dir = self._prepare_run_dir(run_root, benchmark_name, run_name)
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_manifest(
            run_dir,
            benchmark_name=benchmark_name,
            config_path=config_path,
            dataset_source=dataset_source or {},
            selection=selection or {},
            checkpoint_every=checkpoint_every,
        )

        processed_videos = self._prepare_representations(samples, show_progress=show_progress)
        predictions: List[Dict[str, Any]] = []
        evaluation_rows: List[PredictionResult] = []
        chunk_rows: List[Dict[str, Any]] = []
        chunk_index = 0

        iterator = tqdm(samples, desc="Benchmark videos", unit="video") if show_progress else samples
        for sample in iterator:
            started_at = time.perf_counter()
            benchmark_question = build_multiple_choice_prompt(sample.question, sample.candidates)
            processed_video = processed_videos[Path(sample.video_id).stem]
            predicted_raw = self.pipeline.answer_question(
                processed_video,
                benchmark_question,
            ).predicted_answer
            elapsed_s = time.perf_counter() - started_at
            parse = parse_multiple_choice_prediction(predicted_raw, sample.candidates)
            expected = normalize_choice_answer(sample.answer, sample.candidates)

            prediction_record = {
                "question_id": sample.question_id,
                "video_id": sample.video_id,
                "video_path": str(sample.video_path),
                "question_type": sample.question_type,
                "question": sample.question,
                "options": sample.candidates,
                "ground_truth": sample.answer,
                "ground_truth_normalized": expected["normalized_answer"],
                "predicted_raw": predicted_raw,
                "predicted_normalized": parse["normalized_answer"],
                "predicted_option_letter": parse["letter"],
                "predicted_option_index": parse["index"],
                "parse_method": parse["method"],
                "is_correct": parse["normalized_answer"] == expected["normalized_answer"],
                "elapsed_seconds": round(elapsed_s, 3),
                "metadata": sample.metadata,
            }
            predictions.append(prediction_record)
            chunk_rows.append(prediction_record)
            evaluation_rows.append(
                PredictionResult(
                    question_id=sample.question_id,
                    video_id=sample.video_id,
                    question=sample.question,
                    predicted_answer=prediction_record["predicted_normalized"] or predicted_raw,
                    ground_truth=expected["normalized_answer"],
                    metadata={
                        "question_type": sample.question_type,
                        "parse_method": parse["method"],
                        "predicted_raw": predicted_raw,
                    },
                )
            )

            if show_progress:
                accuracy = sum(1 for row in predictions if row["is_correct"]) / len(predictions)
                iterator.set_postfix(acc=f"{accuracy:.1%}", last_s=f"{elapsed_s:.1f}")

            if checkpoint_every > 0 and len(chunk_rows) >= checkpoint_every:
                chunk_index += 1
                self._write_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    chunk_index=chunk_index,
                    rows=chunk_rows,
                )
                self._write_json(run_dir / "progress.json", build_benchmark_progress(predictions, checkpoint_every))
                chunk_rows = []

        if chunk_rows:
            chunk_index += 1
            self._write_checkpoint(
                checkpoints_dir=checkpoints_dir,
                chunk_index=chunk_index,
                rows=chunk_rows,
            )

        metrics = evaluate_multiple_choice_predictions(predictions)
        self._write_predictions(run_dir / "predictions.jsonl", predictions)
        self._write_json(run_dir / "metrics.json", metrics)
        self._write_json(run_dir / "summary.json", build_benchmark_summary(metrics, predictions))
        self._write_json(run_dir / "progress.json", build_benchmark_progress(predictions, checkpoint_every))

        return {
            "run_dir": run_dir,
            "metrics": metrics,
            "predictions": predictions,
            "evaluation_rows": evaluation_rows,
        }

    def run_summary(
        self,
        *,
        samples: Sequence[SummarySample],
        run_root: Path,
        judge: SummaryJudge,
        benchmark_name: str = "mlvu_summary",
        run_name: Optional[str] = None,
        config_path: Optional[Path] = None,
        dataset_source: Optional[Dict[str, Any]] = None,
        selection: Optional[Dict[str, Any]] = None,
        checkpoint_every: int = 5,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        run_dir = self._prepare_run_dir(run_root, benchmark_name, run_name)
        checkpoints_dir = run_dir / "checkpoints"
        judge_outputs_dir = run_dir / "judge_outputs"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        judge_outputs_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_manifest(
            run_dir,
            benchmark_name=benchmark_name,
            config_path=config_path,
            dataset_source=dataset_source or {},
            selection=selection or {},
            checkpoint_every=checkpoint_every,
        )

        processed_videos = self._prepare_representations(samples, show_progress=show_progress)
        predictions: List[Dict[str, Any]] = []
        combined_judgments: Dict[str, Any] = {}
        evaluation_rows: List[PredictionResult] = []
        chunk_rows: List[Dict[str, Any]] = []
        chunk_index = 0
        video_id_counts: Dict[str, int] = {}

        iterator = tqdm(samples, desc="Benchmark videos", unit="video") if show_progress else samples
        for sample in iterator:
            instance_index = video_id_counts.get(sample.video_id, 0)
            video_id_counts[sample.video_id] = instance_index + 1
            sample_instance_id = f"{sample.video_id}_{instance_index}"

            generation_started_at = time.perf_counter()
            processed_video = processed_videos[Path(sample.video_id).stem]
            predicted_summary = self.pipeline.answer_question(
                processed_video,
                sample.question,
            ).predicted_answer
            generation_elapsed_s = time.perf_counter() - generation_started_at

            judge_started_at = time.perf_counter()
            judge_result = judge.judge(
                question=sample.question,
                answer=sample.answer,
                prediction=predicted_summary,
            )
            judge_elapsed_s = time.perf_counter() - judge_started_at

            prediction_record = {
                "question_id": sample.question_id,
                "sample_instance_id": sample_instance_id,
                "video_id": sample.video_id,
                "video_path": str(sample.video_path),
                "question_type": sample.question_type,
                "question": sample.question,
                "ground_truth": sample.answer,
                "predicted_summary": predicted_summary,
                "score_completeness": judge_result["score_completeness"],
                "score_reliability": judge_result["score_reliability"],
                "total_score": judge_result["total_score"],
                "judge_explanation": judge_result["explain"],
                "judge_model": judge.model_name,
                "generation_elapsed_seconds": round(generation_elapsed_s, 3),
                "judge_elapsed_seconds": round(judge_elapsed_s, 3),
                "metadata": sample.metadata,
            }
            predictions.append(prediction_record)
            chunk_rows.append(prediction_record)
            evaluation_rows.append(
                PredictionResult(
                    question_id=sample.question_id,
                    video_id=sample.video_id,
                    question=sample.question,
                    predicted_answer=predicted_summary,
                    ground_truth=sample.answer,
                    metadata={
                        "question_type": sample.question_type,
                        "score_completeness": judge_result["score_completeness"],
                        "score_reliability": judge_result["score_reliability"],
                        "total_score": judge_result["total_score"],
                    },
                )
            )

            judge_file_stem = _slugify(sample_instance_id)
            judge_payload = [
                {"explain": judge_result["explain"]},
                {"q": sample.question, "a": sample.answer, "pred": predicted_summary},
            ]
            self._write_json(judge_outputs_dir / f"{judge_file_stem}.json", judge_payload)
            combined_judgments[judge_file_stem] = judge_payload

            if show_progress:
                metrics_so_far = evaluate_summary_predictions(predictions)
                iterator.set_postfix(
                    total=f"{metrics_so_far['average_total_score']:.2f}",
                    comp=f"{metrics_so_far['average_completeness']:.2f}",
                    reli=f"{metrics_so_far['average_reliability']:.2f}",
                )

            if checkpoint_every > 0 and len(chunk_rows) >= checkpoint_every:
                chunk_index += 1
                self._write_summary_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    chunk_index=chunk_index,
                    rows=chunk_rows,
                )
                self._write_json(
                    run_dir / "progress.json",
                    build_summary_benchmark_progress(predictions, checkpoint_every),
                )
                self._write_json(run_dir / "combined_judgments.json", combined_judgments)
                chunk_rows = []

        if chunk_rows:
            chunk_index += 1
            self._write_summary_checkpoint(
                checkpoints_dir=checkpoints_dir,
                chunk_index=chunk_index,
                rows=chunk_rows,
            )

        metrics = evaluate_summary_predictions(predictions)
        self._write_predictions(run_dir / "predictions.jsonl", predictions)
        self._write_json(run_dir / "metrics.json", metrics)
        self._write_json(run_dir / "summary.json", build_summary_benchmark_summary(metrics, predictions))
        self._write_json(run_dir / "progress.json", build_summary_benchmark_progress(predictions, checkpoint_every))
        self._write_json(run_dir / "combined_judgments.json", combined_judgments)

        return {
            "run_dir": run_dir,
            "metrics": metrics,
            "predictions": predictions,
            "evaluation_rows": evaluation_rows,
        }

    def _prepare_representations(
        self,
        samples: Sequence[MultipleChoiceSample | SummarySample],
        *,
        show_progress: bool,
    ) -> Dict[str, Any]:
        """Generate every unique representation before issuing the first query."""
        unique: Dict[str, Path] = {}
        for sample in samples:
            unique.setdefault(Path(sample.video_id).stem, sample.video_path)

        items = unique.items()
        iterator = (
            tqdm(items, total=len(unique), desc="Textualizing videos", unit="video")
            if show_progress
            else items
        )
        return {
            video_id: self.pipeline.process_video(video_path, video_id)
            for video_id, video_path in iterator
        }

    def _prepare_run_dir(self, run_root: Path, benchmark_name: str, run_name: Optional[str]) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        label = _slugify(run_name or benchmark_name)
        run_dir = run_root / benchmark_name / f"{timestamp}_{label}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _write_run_manifest(
        self,
        run_dir: Path,
        *,
        benchmark_name: str,
        config_path: Optional[Path],
        dataset_source: Dict[str, Any],
        selection: Dict[str, Any],
        checkpoint_every: int,
    ) -> None:
        manifest = {
            "benchmark_name": benchmark_name,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "videoqa_version": self._videoqa_version(),
            "git_commit": _git_commit(),
            "config_path": str(config_path) if config_path else None,
            "config": self.pipeline.config.to_dict(),
            "dataset_source": dataset_source,
            "selection": selection,
            "checkpoint_every": checkpoint_every,
        }
        self._write_json(run_dir / "manifest.json", manifest)
        with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.pipeline.config.to_dict(), f, sort_keys=False)

    def _write_checkpoint(
        self,
        *,
        checkpoints_dir: Path,
        chunk_index: int,
        rows: Sequence[Dict[str, Any]],
    ) -> None:
        prefix = f"chunk_{chunk_index:04d}"
        self._write_predictions(checkpoints_dir / f"{prefix}_predictions.jsonl", rows)
        metrics = evaluate_multiple_choice_predictions(rows)
        self._write_json(checkpoints_dir / f"{prefix}_metrics.json", metrics)
        self._write_json(checkpoints_dir / f"{prefix}_summary.json", build_benchmark_summary(metrics, rows))

    def _write_summary_checkpoint(
        self,
        *,
        checkpoints_dir: Path,
        chunk_index: int,
        rows: Sequence[Dict[str, Any]],
    ) -> None:
        prefix = f"chunk_{chunk_index:04d}"
        self._write_predictions(checkpoints_dir / f"{prefix}_predictions.jsonl", rows)
        metrics = evaluate_summary_predictions(rows)
        self._write_json(checkpoints_dir / f"{prefix}_metrics.json", metrics)
        self._write_json(checkpoints_dir / f"{prefix}_summary.json", build_summary_benchmark_summary(metrics, rows))

    def _write_predictions(self, path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @staticmethod
    def _videoqa_version() -> Optional[str]:
        try:
            from . import __version__

            return __version__
        except Exception:
            return None


def build_multiple_choice_prompt(question: str, candidates: Sequence[str]) -> str:
    """Format a strict multiple-choice prompt for the reasoner."""
    option_lines = [f"- {candidate}" for candidate in candidates]
    return (
        f"{question}\n\n"
        "Options:\n"
        f"{chr(10).join(option_lines)}\n\n"
        "Return exactly one option text from the list above.\n"
        "Do not use option letters.\n"
        "Do not explain your choice.\n"
        "Do not output any text that is not one of the options."
    )


def parse_multiple_choice_prediction(prediction: str, candidates: Sequence[str]) -> Dict[str, Any]:
    """Map raw model output to one of the provided options when possible."""
    text = (prediction or "").strip()
    if not text:
        return {"index": None, "letter": None, "normalized_answer": None, "method": "empty"}

    normalized_text = _normalize_text(text)
    for idx, candidate in enumerate(candidates):
        candidate_norm = _normalize_text(candidate)
        if normalized_text == candidate_norm:
            return _parsed_choice(idx, candidates, "exact_candidate")

    anchored = _parse_anchored_candidate(text, candidates)
    if anchored is not None:
        return anchored

    letter_patterns = [
        r"^\(?([A-Z])\)?$",
        r"^\(?([A-Z])\)?[\.\:\- ]",
        r"option\s+\(?([A-Z])\)?",
        r"answer\s*[:\-]?\s*\(?([A-Z])\)?",
    ]
    for pattern in letter_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            idx = CHOICE_LETTERS.find(match.group(1).upper())
            if 0 <= idx < len(candidates):
                return _parsed_choice(idx, candidates, "letter")

    for idx, candidate in enumerate(candidates):
        candidate_norm = _normalize_text(candidate)
        if candidate_norm and candidate_norm in normalized_text:
            return _parsed_choice(idx, candidates, "candidate_substring")

    best_idx = None
    best_score = 0.0
    for idx, candidate in enumerate(candidates):
        score = _token_overlap_score(normalized_text, _normalize_text(candidate))
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx is not None and best_score >= 0.6:
        return _parsed_choice(best_idx, candidates, "token_overlap")

    return {"index": None, "letter": None, "normalized_answer": None, "method": "unparsed"}


def normalize_choice_answer(answer: str, candidates: Sequence[str]) -> Dict[str, Any]:
    """Normalize the benchmark answer into the same choice format as parsed predictions."""
    parsed = parse_multiple_choice_prediction(answer, candidates)
    if parsed["normalized_answer"] is not None:
        return parsed

    normalized_answer = _normalize_text(answer)
    for idx, candidate in enumerate(candidates):
        if _normalize_text(candidate) == normalized_answer:
            return _parsed_choice(idx, candidates, "ground_truth_exact")

    return {"index": None, "letter": None, "normalized_answer": answer.strip(), "method": "ground_truth_raw"}


def evaluate_multiple_choice_predictions(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy and parsing statistics for multiple-choice runs."""
    total = len(predictions)
    correct = sum(1 for row in predictions if row["is_correct"])
    parsed = sum(1 for row in predictions if row["predicted_normalized"] is not None)

    by_question_type: Dict[str, Dict[str, Any]] = {}
    for row in predictions:
        bucket = by_question_type.setdefault(
            row["question_type"],
            {"total": 0, "correct": 0, "parsed": 0},
        )
        bucket["total"] += 1
        bucket["correct"] += int(row["is_correct"])
        bucket["parsed"] += int(row["predicted_normalized"] is not None)

    for bucket in by_question_type.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
        bucket["parse_rate"] = bucket["parsed"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parsed": parsed,
        "parse_rate": parsed / total if total else 0.0,
        "by_question_type": by_question_type,
    }


def build_benchmark_summary(metrics: Dict[str, Any], predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a compact summary for quick inspection."""
    failures = [row for row in predictions if not row["is_correct"]][:10]
    return {
        "headline": {
            "accuracy": metrics["accuracy"],
            "correct": metrics["correct"],
            "total": metrics["total"],
            "parse_rate": metrics["parse_rate"],
        },
        "by_question_type": metrics["by_question_type"],
        "sample_failures": failures,
    }


def evaluate_summary_predictions(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate official-style summary judge scores."""
    total = len(predictions)
    scored_rows = [
        row
        for row in predictions
        if all(isinstance(row.get(key), (int, float)) for key in ("score_completeness", "score_reliability", "total_score"))
    ]
    scored = len(scored_rows)

    comp_sum = sum(float(row["score_completeness"]) for row in scored_rows)
    reli_sum = sum(float(row["score_reliability"]) for row in scored_rows)
    total_sum = sum(float(row["total_score"]) for row in scored_rows)

    by_question_type: Dict[str, Dict[str, Any]] = {}
    for row in predictions:
        bucket = by_question_type.setdefault(
            row["question_type"],
            {
                "total": 0,
                "scored": 0,
                "score_completeness_sum": 0.0,
                "score_reliability_sum": 0.0,
                "total_score_sum": 0.0,
            },
        )
        bucket["total"] += 1
        if all(isinstance(row.get(key), (int, float)) for key in ("score_completeness", "score_reliability", "total_score")):
            bucket["scored"] += 1
            bucket["score_completeness_sum"] += float(row["score_completeness"])
            bucket["score_reliability_sum"] += float(row["score_reliability"])
            bucket["total_score_sum"] += float(row["total_score"])

    for bucket in by_question_type.values():
        denom = bucket["scored"]
        bucket["average_completeness"] = bucket["score_completeness_sum"] / denom if denom else 0.0
        bucket["average_reliability"] = bucket["score_reliability_sum"] / denom if denom else 0.0
        bucket["average_total_score"] = bucket["total_score_sum"] / denom if denom else 0.0
        bucket["score_rate"] = bucket["scored"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "total": total,
        "scored": scored,
        "average_completeness": comp_sum / scored if scored else 0.0,
        "average_reliability": reli_sum / scored if scored else 0.0,
        "average_total_score": total_sum / scored if scored else 0.0,
        "score_rate": scored / total if total else 0.0,
        "by_question_type": by_question_type,
    }


def build_summary_benchmark_summary(metrics: Dict[str, Any], predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a compact summary for official-style summary benchmark runs."""
    lowest = sorted(
        [
            row
            for row in predictions
            if isinstance(row.get("total_score"), (int, float))
        ],
        key=lambda row: row["total_score"],
    )[:10]
    return {
        "headline": {
            "average_completeness": metrics["average_completeness"],
            "average_reliability": metrics["average_reliability"],
            "average_total_score": metrics["average_total_score"],
            "scored": metrics["scored"],
            "total": metrics["total"],
            "score_rate": metrics["score_rate"],
        },
        "by_question_type": metrics["by_question_type"],
        "sample_low_scores": lowest,
    }


def build_benchmark_progress(predictions: Sequence[Dict[str, Any]], checkpoint_every: int) -> Dict[str, Any]:
    """Return current run progress for long benchmark jobs."""
    total = len(predictions)
    correct = sum(1 for row in predictions if row["is_correct"])
    return {
        "processed_videos": total,
        "correct_so_far": correct,
        "accuracy_so_far": correct / total if total else 0.0,
        "checkpoint_every": checkpoint_every,
        "last_updated_utc": datetime.now(UTC).isoformat(),
    }


def build_summary_benchmark_progress(predictions: Sequence[Dict[str, Any]], checkpoint_every: int) -> Dict[str, Any]:
    """Return current run progress for summary benchmark jobs."""
    metrics = evaluate_summary_predictions(predictions)
    return {
        "processed_videos": metrics["total"],
        "scored_so_far": metrics["scored"],
        "average_completeness_so_far": metrics["average_completeness"],
        "average_reliability_so_far": metrics["average_reliability"],
        "average_total_score_so_far": metrics["average_total_score"],
        "checkpoint_every": checkpoint_every,
        "last_updated_utc": datetime.now(UTC).isoformat(),
    }


def benchmark_selection_to_dict(
    *,
    question_types: Sequence[str],
    limit: Optional[int],
    video_ids: Optional[Sequence[str]],
    shuffle: bool,
    seed: int,
) -> Dict[str, Any]:
    return {
        "question_types": list(question_types),
        "limit": limit,
        "video_ids": list(video_ids) if video_ids else None,
        "shuffle": shuffle,
        "seed": seed,
    }


def dataset_source_to_dict(adapter: MLVUDatasetAdapter) -> Dict[str, Any]:
    return {
        "dataset_path": str(adapter.dataset_path) if adapter.dataset_path else None,
        "videos_dir": str(adapter.videos_dir) if adapter.videos_dir else None,
        "annotations_path": str(adapter.annotations_path) if adapter.annotations_path else None,
        "hf_dataset": adapter.hf_dataset,
        "hf_split": adapter.hf_split,
        "hf_revision": adapter.hf_revision,
        "question_types": sorted(adapter.question_types) if adapter.question_types else None,
    }


def summary_dataset_source_to_dict(adapter: SummaryDatasetAdapter) -> Dict[str, Any]:
    return {
        "dataset_path": str(adapter.dataset_path) if adapter.dataset_path else None,
        "videos_dir": str(adapter.videos_dir) if adapter.videos_dir else None,
        "annotations_path": str(adapter.annotations_path) if adapter.annotations_path else None,
        "hf_dataset": adapter.hf_dataset,
        "hf_split": adapter.hf_split,
        "hf_revision": adapter.hf_revision,
        "question_types": sorted(adapter.question_types) if adapter.question_types else None,
    }


def _parsed_choice(index: int, candidates: Sequence[str], method: str) -> Dict[str, Any]:
    return {
        "index": index,
        "letter": CHOICE_LETTERS[index],
        "normalized_answer": candidates[index],
        "method": method,
    }


def _parse_anchored_candidate(text: str, candidates: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Prefer exact option mentions that appear in strong answer-bearing contexts."""
    lowered = text.lower()
    best_idx = None
    best_score = -1

    anchor_patterns = [
        r"(?:final answer|best answer|correct answer|the answer is|answer is|therefore|thus|so)\s*[:\-]?\s*(.*)",
        r"(?:choose|choose option|select|selected)\s*[:\-]?\s*(.*)",
    ]

    for idx, candidate in enumerate(candidates):
        candidate_lower = candidate.lower()
        for pattern in anchor_patterns:
            for match in re.finditer(pattern, lowered, re.IGNORECASE | re.DOTALL):
                remainder = match.group(1)
                pos = remainder.find(candidate_lower)
                if pos >= 0:
                    # Earlier anchored mentions are usually the direct answer,
                    # while later mentions often occur in extra explanation.
                    score = 1000 - pos
                    if score > best_score:
                        best_score = score
                        best_idx = idx

    if best_idx is not None:
        return _parsed_choice(best_idx, candidates, "anchored_candidate")
    return None


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\(\)\[\]\{\}\.,:;!\?\"']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def _stable_text_id(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:48] or "question"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "run"


def _parse_summary_scores(response_message: str) -> Dict[str, Any]:
    """Extract official score fields from a judge response."""
    response_message = (response_message or "").strip()
    candidates = []
    if response_message:
        candidates.append(response_message)

    match = re.search(r"\{.*\}", response_message, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(candidate)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            score_comp = _coerce_int(parsed.get("score_completeness"))
            score_reli = _coerce_int(parsed.get("score_reliability"))
            total_score = _coerce_int(parsed.get("total_score"))
            if total_score is None and score_comp is not None and score_reli is not None:
                total_score = score_comp + score_reli
            return {
                "score_completeness": score_comp,
                "score_reliability": score_reli,
                "total_score": total_score,
            }

    return {
        "score_completeness": None,
        "score_reliability": None,
        "total_score": None,
    }


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _git_commit() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None
