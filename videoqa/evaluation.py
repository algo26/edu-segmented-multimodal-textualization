"""
Evaluation module for video QA benchmarks.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .common_types import PredictionResult

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates video QA predictions."""

    def __init__(self):
        pass

    def evaluate(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of prediction results

        Returns:
            Dictionary of evaluation metrics
        """
        # Filter predictions with ground truth
        predictions_with_gt = [p for p in predictions if p.ground_truth is not None]

        if not predictions_with_gt:
            logger.warning("No predictions with ground truth found")
            return {"error": "No ground truth available"}

        # Calculate metrics
        exact_match = self._exact_match_accuracy(predictions_with_gt)

        metrics = {
            "total_predictions": len(predictions),
            "predictions_with_gt": len(predictions_with_gt),
            "exact_match_accuracy": exact_match,
        }

        logger.info(f"Evaluation results: {metrics}")
        return metrics

    def _exact_match_accuracy(self, predictions: List[PredictionResult]) -> float:
        """
        Calculate exact match accuracy.

        Args:
            predictions: List of predictions with ground truth

        Returns:
            Accuracy as a float between 0 and 1
        """
        if not predictions:
            return 0.0

        correct = 0
        for pred in predictions:
            # Normalize answers for comparison
            pred_answer = self._normalize_answer(pred.predicted_answer)
            gt_answer = self._normalize_answer(pred.ground_truth)

            if pred_answer == gt_answer:
                correct += 1

        accuracy = correct / len(predictions)
        return accuracy

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Lowercase and strip whitespace
        answer = answer.lower().strip()

        # Remove punctuation
        import string
        answer = answer.translate(str.maketrans('', '', string.punctuation))

        return answer

    def save_predictions(self, predictions: List[PredictionResult], output_path: Path):
        """
        Save predictions to JSONL file.

        Args:
            predictions: List of prediction results
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for pred in predictions:
                # Convert to dict
                pred_dict = {
                    "question_id": pred.question_id,
                    "video_id": pred.video_id,
                    "question": pred.question,
                    "predicted_answer": pred.predicted_answer,
                    "ground_truth": pred.ground_truth,
                    "confidence": pred.confidence,
                    "metadata": pred.metadata
                }

                # Write as JSON line
                f.write(json.dumps(pred_dict) + '\n')

        logger.info(f"Saved {len(predictions)} predictions to {output_path}")

    def load_predictions(self, input_path: Path) -> List[PredictionResult]:
        """
        Load predictions from JSONL file.

        Args:
            input_path: Path to input file

        Returns:
            List of PredictionResult objects
        """
        predictions = []

        with open(input_path, 'r') as f:
            for line in f:
                pred_dict = json.loads(line)

                pred = PredictionResult(
                    question_id=pred_dict["question_id"],
                    video_id=pred_dict["video_id"],
                    question=pred_dict["question"],
                    predicted_answer=pred_dict["predicted_answer"],
                    ground_truth=pred_dict.get("ground_truth"),
                    confidence=pred_dict.get("confidence"),
                    metadata=pred_dict.get("metadata", {})
                )

                predictions.append(pred)

        logger.info(f"Loaded {len(predictions)} predictions from {input_path}")
        return predictions

    def print_summary(self, predictions: List[PredictionResult], num_samples: int = 5):
        """
        Print summary of predictions.

        Args:
            predictions: List of predictions
            num_samples: Number of sample predictions to show
        """
        print("\n" + "="*80)
        print("PREDICTION SUMMARY")
        print("="*80)

        # Overall stats
        print(f"\nTotal predictions: {len(predictions)}")

        with_gt = [p for p in predictions if p.ground_truth is not None]
        if with_gt:
            print(f"Predictions with ground truth: {len(with_gt)}")

            # Calculate accuracy
            metrics = self.evaluate(predictions)
            if "exact_match_accuracy" in metrics:
                print(f"Exact match accuracy: {metrics['exact_match_accuracy']:.2%}")

        # Show samples
        print(f"\n{num_samples} Sample Predictions:")
        print("-"*80)

        for i, pred in enumerate(predictions[:num_samples]):
            print(f"\nSample {i+1}:")
            print(f"  Video ID: {pred.video_id}")
            print(f"  Question: {pred.question}")
            print(f"  Predicted: {pred.predicted_answer}")
            if pred.ground_truth:
                print(f"  Ground Truth: {pred.ground_truth}")
                match = "✓" if self._normalize_answer(pred.predicted_answer) == self._normalize_answer(pred.ground_truth) else "✗"
                print(f"  Match: {match}")

        print("\n" + "="*80)


class DatasetAdapter:
    """
    Base class for dataset adapters.
    Subclass this for specific datasets like MovieChat-1K, ActivityNet-QA, etc.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def load_samples(self) -> List[Dict[str, Any]]:
        """
        Load dataset samples.

        Returns:
            List of sample dictionaries
        """
        raise NotImplementedError("Subclass must implement load_samples()")

    def save_results(self, predictions: List[PredictionResult], output_path: Path):
        """
        Save results in dataset-specific format.

        Args:
            predictions: List of predictions
            output_path: Path to output file
        """
        # Default: save as JSONL
        evaluator = Evaluator()
        evaluator.save_predictions(predictions, output_path)
