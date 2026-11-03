#!/usr/bin/env python3
"""
Command-line interface for the VideoQA pipeline.
"""
import argparse
import logging
import sys
from pathlib import Path

from videoqa import (
    BenchmarkRunner,
    MLVUDatasetAdapter,
    PipelineConfig,
    SummaryDatasetAdapter,
    SummaryJudge,
    VideoQAPipeline,
    VideoQASample,
)
from videoqa.benchmarking import (
    benchmark_selection_to_dict,
    dataset_source_to_dict,
    summary_dataset_source_to_dict,
)


def setup_logging(verbose: bool = True):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_single_video(args):
    """Run QA on a single video."""
    print(f"\n{'='*80}")
    print("SINGLE VIDEO QA MODE")
    print(f"{'='*80}\n")

    # Load config
    config = PipelineConfig.from_yaml(Path(args.config))

    # Initialize pipeline
    pipeline = VideoQAPipeline(config)

    # Run QA
    answer = pipeline.run_single_qa(
        video_path=Path(args.video),
        question=args.question,
        video_id=args.video_id
    )

    # Print result
    print(f"\n{'='*80}")
    print("RESULT")
    print(f"{'='*80}")
    print(f"\nQuestion: {args.question}")
    print(f"\nAnswer: {answer}\n")
    print(f"{'='*80}\n")


def run_batch(args):
    """Run QA on a batch of videos."""
    print(f"\n{'='*80}")
    print("BATCH QA MODE")
    print(f"{'='*80}\n")

    # Load config
    config = PipelineConfig.from_yaml(Path(args.config))

    # Initialize pipeline
    pipeline = VideoQAPipeline(config)

    # Load samples from JSON/JSONL
    samples = load_samples_from_file(Path(args.input))

    print(f"Loaded {len(samples)} samples\n")

    # Run batch QA
    output_path = Path(args.output) if args.output else None
    predictions = pipeline.run_qa_batch(samples, output_path=output_path)

    # Print summary
    pipeline.evaluator.print_summary(predictions)

    if output_path:
        print(f"\nPredictions saved to: {output_path}")


def load_samples_from_file(input_path: Path):
    """
    Load VideoQA samples from JSON/JSONL file.

    Expected format (JSONL):
    {"video_id": "video1", "video_path": "/path/to/video.mp4", "question": "What happens?", "answer": "..."}
    {"video_id": "video2", "video_path": "/path/to/video2.mp4", "question": "Who appears?", "answer": "..."}
    """
    import json

    samples = []

    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())

            sample = VideoQASample(
                video_id=data['video_id'],
                video_path=Path(data['video_path']),
                question=data['question'],
                answer=data.get('answer'),
                question_id=data.get('question_id'),
                metadata=data.get('metadata', {})
            )

            samples.append(sample)

    return samples


def test_setup(args):
    """Test pipeline setup and dependencies."""
    print(f"\n{'='*80}")
    print("TESTING PIPELINE SETUP")
    print(f"{'='*80}\n")

    # Load config
    config_path = Path(args.config)
    print(f"Loading config from: {config_path}")
    config = PipelineConfig.from_yaml(config_path)
    print("✓ Config loaded successfully\n")

    # Test device detection
    from videoqa.device_utils import detect_device, get_device_info

    device = detect_device(config.device)
    device_info = get_device_info()

    print(f"Detected device: {device}")
    print(f"Device info: {device_info}\n")

    # Initialize pipeline
    print("Initializing pipeline components...")
    VideoQAPipeline(config)
    print("✓ Pipeline initialized successfully\n")

    print(f"{'='*80}")
    print("SETUP TEST COMPLETE")
    print(f"{'='*80}\n")


def run_mlvu_benchmark(args):
    """Run close-ended MLVU benchmark evaluation."""
    print(f"\n{'='*80}")
    print("MLVU BENCHMARK MODE")
    print(f"{'='*80}\n")

    config_path = Path(args.config)
    config = PipelineConfig.from_yaml(config_path)
    pipeline = VideoQAPipeline(config)

    question_types = [q.strip() for q in args.question_types.split(",") if q.strip()]
    video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()] if args.video_ids else None

    adapter = MLVUDatasetAdapter(
        dataset_path=Path(args.dataset_root) if args.dataset_root else None,
        videos_dir=Path(args.videos_dir) if args.videos_dir else None,
        annotations_path=Path(args.annotations) if args.annotations else None,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_revision=args.hf_revision,
        hf_token=args.hf_token,
        question_types=question_types,
    )
    samples = adapter.load_samples(
        limit=args.limit,
        video_ids=video_ids,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not samples:
        raise ValueError("No benchmark samples matched the requested filters")

    print(f"Loaded {len(samples)} benchmark samples")
    runner = BenchmarkRunner(pipeline)
    result = runner.run_mlvu(
        samples=samples,
        run_root=Path(args.output_dir),
        benchmark_name="mlvu_mc",
        run_name=args.run_name,
        config_path=config_path,
        dataset_source=dataset_source_to_dict(adapter),
        selection=benchmark_selection_to_dict(
            question_types=question_types,
            limit=args.limit,
            video_ids=video_ids,
            shuffle=args.shuffle,
            seed=args.seed,
        ),
        checkpoint_every=args.checkpoint_every,
        show_progress=not args.no_progress,
    )

    metrics = result["metrics"]
    print(f"\nAccuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Parse rate: {metrics['parse_rate']:.2%}")
    for question_type, bucket in metrics["by_question_type"].items():
        print(
            f"  {question_type}: accuracy={bucket['accuracy']:.2%} "
            f"({bucket['correct']}/{bucket['total']}) parse_rate={bucket['parse_rate']:.2%}"
        )
    print(f"\nSaved run to: {result['run_dir']}")


def run_summary_benchmark(args):
    """Run summarization benchmark evaluation."""
    print(f"\n{'='*80}")
    print("SUMMARY BENCHMARK MODE")
    print(f"{'='*80}\n")

    config_path = Path(args.config)
    config = PipelineConfig.from_yaml(config_path)
    pipeline = VideoQAPipeline(config)

    question_types = [q.strip() for q in args.question_types.split(",") if q.strip()]
    video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()] if args.video_ids else None

    adapter = SummaryDatasetAdapter(
        dataset_path=Path(args.dataset_root) if args.dataset_root else None,
        videos_dir=Path(args.videos_dir) if args.videos_dir else None,
        annotations_path=Path(args.annotations) if args.annotations else None,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_revision=args.hf_revision,
        hf_token=args.hf_token,
        question_types=question_types,
    )
    samples = adapter.load_samples(
        limit=args.limit,
        video_ids=video_ids,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not samples:
        raise ValueError("No benchmark samples matched the requested filters")

    judge = SummaryJudge(
        model_name=args.judge_model,
        api_key=args.judge_api_key,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
    )

    print(f"Loaded {len(samples)} benchmark samples")
    runner = BenchmarkRunner(pipeline)
    result = runner.run_summary(
        samples=samples,
        run_root=Path(args.output_dir),
        judge=judge,
        benchmark_name="mlvu_summary",
        run_name=args.run_name,
        config_path=config_path,
        dataset_source=summary_dataset_source_to_dict(adapter),
        selection=benchmark_selection_to_dict(
            question_types=question_types,
            limit=args.limit,
            video_ids=video_ids,
            shuffle=args.shuffle,
            seed=args.seed,
        ),
        checkpoint_every=args.checkpoint_every,
        show_progress=not args.no_progress,
    )

    metrics = result["metrics"]
    print(
        f"\nAverage total score: {metrics['average_total_score']:.3f} "
        f"(completeness={metrics['average_completeness']:.3f}, "
        f"reliability={metrics['average_reliability']:.3f})"
    )
    for question_type, bucket in metrics["by_question_type"].items():
        print(
            f"  {question_type}: total={bucket['average_total_score']:.3f} "
            f"completeness={bucket['average_completeness']:.3f} "
            f"reliability={bucket['average_reliability']:.3f} "
            f"scored={bucket['scored']}/{bucket['total']}"
        )
    print(f"\nSaved run to: {result['run_dir']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Segmented multimodal textualization and MLVU evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test setup
  python run.py test --config configs/c1.yaml

  # Single video QA
  python run.py single --config configs/c1.yaml --video /path/to/video.mp4 --question "What happens in the video?"

  # Batch QA
  python run.py batch --config configs/c1.yaml --input samples.jsonl --output predictions.jsonl

  # MLVU benchmark from local files
  python run.py benchmark-mlvu --config configs/c1.yaml \
    --dataset-root /path/to/MLVU --question-types topic_reasoning,needle_qa --limit 5

  # Summary benchmark from local files
  python run.py benchmark-summary --config configs/c1.yaml \
    --annotations /path/to/test_vs_gt.json --videos-dir /path/to/videos --limit 5
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test pipeline setup')
    test_parser.add_argument('--config', required=True, help='Path to config YAML')

    # Single video command
    single_parser = subparsers.add_parser('single', help='Process single video')
    single_parser.add_argument('--config', required=True, help='Path to config YAML')
    single_parser.add_argument('--video', required=True, help='Path to video file')
    single_parser.add_argument('--question', required=True, help='Question to answer')
    single_parser.add_argument('--video-id', help='Video ID (default: filename)')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process batch of videos')
    batch_parser.add_argument('--config', required=True, help='Path to config YAML')
    batch_parser.add_argument('--input', required=True, help='Input JSONL file with samples')
    batch_parser.add_argument('--output', help='Output JSONL file for predictions')

    benchmark_parser = subparsers.add_parser('benchmark-mlvu', help='Run MLVU multiple-choice benchmark')
    benchmark_parser.add_argument('--config', required=True, help='Path to config YAML')
    benchmark_parser.add_argument('--dataset-root', help='Local dataset root containing videos and/or annotations')
    benchmark_parser.add_argument('--videos-dir', help='Directory containing MP4 files')
    benchmark_parser.add_argument('--annotations', help='Annotation JSON/JSONL file or directory')
    benchmark_parser.add_argument('--hf-dataset', help='Hugging Face dataset id for MLVU')
    benchmark_parser.add_argument('--hf-split', help='Hugging Face split name')
    benchmark_parser.add_argument('--hf-revision', help='Dataset revision/commit for reproducibility')
    benchmark_parser.add_argument('--hf-token', help='Optional Hugging Face token')
    benchmark_parser.add_argument(
        '--question-types',
        default='topic_reasoning,needle_qa',
        help='Comma-separated MLVU question types to include',
    )
    benchmark_parser.add_argument('--limit', type=int, help='Limit the number of benchmark samples')
    benchmark_parser.add_argument('--video-ids', help='Comma-separated video file names to evaluate')
    benchmark_parser.add_argument('--shuffle', action='store_true', help='Shuffle filtered samples before limiting')
    benchmark_parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    benchmark_parser.add_argument('--output-dir', default='benchmark_runs', help='Directory for benchmark outputs')
    benchmark_parser.add_argument('--run-name', help='Optional run label')
    benchmark_parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=5,
        help='Write intermediate benchmark artifacts every N processed videos',
    )
    benchmark_parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable the benchmark progress bar',
    )

    summary_parser = subparsers.add_parser('benchmark-summary', help='Run summarization benchmark')
    summary_parser.add_argument('--config', required=True, help='Path to config YAML')
    summary_parser.add_argument('--dataset-root', help='Local dataset root containing videos and/or annotations')
    summary_parser.add_argument('--videos-dir', help='Directory containing MP4 files')
    summary_parser.add_argument('--annotations', help='Annotation JSON/JSONL file or directory')
    summary_parser.add_argument('--hf-dataset', help='Hugging Face dataset id for summary benchmark')
    summary_parser.add_argument('--hf-split', help='Hugging Face split name')
    summary_parser.add_argument('--hf-revision', help='Dataset revision/commit for reproducibility')
    summary_parser.add_argument('--hf-token', help='Optional Hugging Face token')
    summary_parser.add_argument(
        '--question-types',
        default='summary',
        help='Comma-separated summary question types to include',
    )
    summary_parser.add_argument('--limit', type=int, help='Limit the number of benchmark samples')
    summary_parser.add_argument('--video-ids', help='Comma-separated video file names to evaluate')
    summary_parser.add_argument('--shuffle', action='store_true', help='Shuffle filtered samples before limiting')
    summary_parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    summary_parser.add_argument('--output-dir', default='benchmark_runs', help='Directory for benchmark outputs')
    summary_parser.add_argument('--run-name', help='Optional run label')
    summary_parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=5,
        help='Write intermediate benchmark artifacts every N processed videos',
    )
    summary_parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable the benchmark progress bar',
    )
    summary_parser.add_argument('--judge-model', default='gpt-4-turbo', help='OpenAI model used for summary judging')
    summary_parser.add_argument('--judge-api-key', help='Optional OpenAI API key override for the judge')
    summary_parser.add_argument('--judge-temperature', type=float, default=0.0, help='Judge temperature')
    summary_parser.add_argument('--judge-max-tokens', type=int, default=512, help='Judge max tokens')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    setup_logging(verbose=True)

    # Route to appropriate function
    if args.command == 'test':
        test_setup(args)
    elif args.command == 'single':
        run_single_video(args)
    elif args.command == 'batch':
        run_batch(args)
    elif args.command == 'benchmark-mlvu':
        run_mlvu_benchmark(args)
    elif args.command == 'benchmark-summary':
        run_summary_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
