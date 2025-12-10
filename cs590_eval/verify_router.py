#!/usr/bin/env python
"""
Verify router classification accuracy on both normal and hidden datasets.
Uses the exact same router as MultiModelRouterPipeline in pipelines.py.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from rule_based_router import route_question  # Same router used by MultiModelRouterPipeline._router
from generate import _read_jsonl


# Ground truth mapping: dataset name -> expected task type
TASK_TYPE_MAP = {
    # Normal tasks
    "triviaqa": "factual_qa",
    "arc-c": "reasoning",
    "ifeval": "instruction_following",
    # Hidden tasks
    "hidden_factual_qa": "factual_qa",
    "hidden_reasoning": "reasoning",
    "hidden_instruction_following": "instruction_following",
}


def save_results_to_csv(
    results: List[Dict[str, Any]],
    output_file: str
):
    """Save per-dataset results to CSV."""
    if not results:
        return

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['dataset', 'expected_type', 'total', 'correct', 'accuracy_%',
                     'factual_qa', 'reasoning', 'instruction_following']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_confusion_matrix_csv(
    matrix: Dict[str, Dict[str, int]],
    output_file: str
):
    """Save confusion matrix to CSV."""
    if not matrix:
        return

    all_types = sorted(set(
        list(matrix.keys()) +
        [p for preds in matrix.values() for p in preds.keys()]
    ))

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Expected \\ Predicted'] + all_types)
        # Rows
        for expected in all_types:
            row = [expected]
            for predicted in all_types:
                count = matrix.get(expected, {}).get(predicted, 0)
                row.append(count)
            writer.writerow(row)


def verify_router_on_dataset(
    dataset_name: str,
    data_path: str,
    expected_type: str,
    data_size: int = -1
) -> Tuple[int, int, Dict[str, int]]:
    """
    Verify router accuracy on a single dataset.

    Args:
        dataset_name: Name of the dataset (for display)
        data_path: Path to the JSONL file
        expected_type: Expected task type for this dataset
        data_size: Number of items to check (-1 for all)

    Returns:
        (correct_count, total_count, prediction_counts)
    """
    if not os.path.exists(data_path):
        return 0, 0, {}

    items = _read_jsonl(data_path)
    if data_size > 0:
        items = items[:data_size]

    correct = 0
    prediction_counts = defaultdict(int)

    for item in items:
        # Use same routing logic as MultiModelRouterPipeline._router (pipelines.py:382-384)
        predicted = route_question(item)
        prediction_counts[predicted] += 1
        if predicted == expected_type:
            correct += 1

    total = len(items)
    return correct, total, dict(prediction_counts)


def build_confusion_matrix(
    all_results: List[Tuple[str, str, int]]
) -> Dict[str, Dict[str, int]]:
    """
    Build confusion matrix from results.

    Args:
        all_results: List of (expected, predicted, count) tuples

    Returns:
        Dict[expected][predicted] = count
    """
    matrix = defaultdict(lambda: defaultdict(int))
    for expected, predicted, count in all_results:
        matrix[expected][predicted] += count
    return {k: dict(v) for k, v in matrix.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Verify router classification accuracy on normal and hidden datasets"
    )
    parser.add_argument(
        "--data_triviaqa",
        default="data/triviaqa_test.jsonl",
        help="Path to TriviaQA test data"
    )
    parser.add_argument(
        "--data_arc_c",
        default="data/arc_c_test.jsonl",
        help="Path to ARC-C test data"
    )
    parser.add_argument(
        "--data_ifeval",
        default="data/ifeval_test.jsonl",
        help="Path to IFEval test data"
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=1000,
        help="Number of items to check per dataset (-1 for all)"
    )
    parser.add_argument(
        "--check_hidden",
        action="store_true",
        help="Also check hidden datasets"
    )
    parser.add_argument(
        "--hidden_dir",
        default="hidden_data",
        help="Directory containing hidden datasets"
    )

    args = parser.parse_args()


    # Define datasets to check
    datasets_to_check = {
        "TriviaQA": (args.data_triviaqa, "triviaqa"),
        "ARC-C": (args.data_arc_c, "arc-c"),
        "IFEval": (args.data_ifeval, "ifeval"),
    }

    if args.check_hidden:
        datasets_to_check.update({
            "Hidden Factual QA": (
                os.path.join(args.hidden_dir, "hidden_factual_qa_test.jsonl"),
                "hidden_factual_qa"
            ),
            "Hidden Reasoning": (
                os.path.join(args.hidden_dir, "hidden_reasoning_test.jsonl"),
                "hidden_reasoning"
            ),
            "Hidden Instruction Following": (
                os.path.join(args.hidden_dir, "hidden_instruction_following_test.jsonl"),
                "hidden_instruction_following"
            ),
        })

    # Verify each dataset
    total_correct = 0
    total_items = 0
    all_confusion_data: List[Tuple[str, str, int]] = []
    csv_results: List[Dict[str, Any]] = []

    for display_name, (data_path, dataset_key) in datasets_to_check.items():
        expected_type = TASK_TYPE_MAP[dataset_key]
        correct, total, predictions = verify_router_on_dataset(
            display_name, data_path, expected_type, args.data_size
        )
        total_correct += correct
        total_items += total

        # Add to confusion matrix data
        for predicted_type, count in predictions.items():
            all_confusion_data.append((expected_type, predicted_type, count))

        # Collect for CSV
        if total > 0:
            csv_results.append({
                'dataset': display_name,
                'expected_type': expected_type,
                'total': total,
                'correct': correct,
                'accuracy_%': round(correct / total * 100, 2),
                'factual_qa': predictions.get('factual_qa', 0),
                'reasoning': predictions.get('reasoning', 0),
                'instruction_following': predictions.get('instruction_following', 0),
            })

    # Print simple results
    if total_items > 0:
        # Print per-task accuracy
        for result in csv_results:
            print(f"{result['dataset']}: {result['accuracy_%']:.2f}%")

        # Print overall
        overall_accuracy = total_correct / total_items * 100
        print(f"Overall: {overall_accuracy:.2f}%")

        # Save files (silent)
        confusion_matrix = build_confusion_matrix(all_confusion_data)
        results = {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_items": total_items,
            "confusion_matrix": confusion_matrix,
        }
        with open("router_verification_results.json", "w") as f:
            json.dump(results, f, indent=2)
        save_results_to_csv(csv_results, "router_verification_results.csv")
        save_confusion_matrix_csv(confusion_matrix, "router_confusion_matrix.csv")


if __name__ == "__main__":
    main()
