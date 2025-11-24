#!/usr/bin/env python3
"""
Evaluate router on ARC-C, TriviaQA, and IFEval test data.

Usage: python router_training/test_router.py
"""
import argparse
import json
from router import Router

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router_path", default="checkpoints/router/final")
    parser.add_argument("--eval_data", default="router_training/data/eval.jsonl")
    parser.add_argument("--out_dir", default="router_training/results")
    args = parser.parse_args()

    print(f"Loading router from {args.router_path}...")
    router = Router(args.router_path)

    print(f"Loading evaluation data from {args.eval_data}...")
    eval_data = load_jsonl(args.eval_data)
    print(f"Total: {len(eval_data)} examples\n")

    # Evaluate
    print("Evaluating...")
    correct = 0
    results_by_task = {}
    all_results = []

    for item in eval_data:
        prompt_text = item['prompt']
        # Extract question from "Classify task type:\n{question}\nTask:"
        question = prompt_text.split("Classify task type:\n")[1].split("\nTask:")[0]
        expected = item['target']

        predicted = router.classify(question)

        if expected not in results_by_task:
            results_by_task[expected] = {"correct": 0, "total": 0}

        results_by_task[expected]["total"] += 1
        is_correct = predicted == expected
        if is_correct:
            correct += 1
            results_by_task[expected]["correct"] += 1

        # Store result
        all_results.append({
            "question": question[:200],  # Truncate for readability
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct
        })

    # Calculate metrics
    metrics = {}
    for task in sorted(results_by_task.keys()):
        stats = results_by_task[task]
        metrics[task] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"]
        }

    metrics["overall"] = {
        "correct": correct,
        "total": len(eval_data),
        "accuracy": correct / len(eval_data)
    }

    # Save results
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    results_path = os.path.join(args.out_dir, "predictions.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print results
    print("\n" + "="*60)
    print("Router Evaluation Results (Generation-based)")
    print("="*60)
    for task in sorted(results_by_task.keys()):
        stats = results_by_task[task]
        acc = stats["correct"] / stats["total"] * 100
        print(f"{task:25} {stats['correct']:5}/{stats['total']:5} ({acc:6.2f}%)")

    print("="*60)
    overall_acc = correct / len(eval_data) * 100
    print(f"{'Overall':25} {correct:5}/{len(eval_data):5} ({overall_acc:6.2f}%)")
    print("="*60)

    print(f"\nResults saved to:")
    print(f"  {results_path}")
    print(f"  {metrics_path}")

    # Print sample predictions
    print("\n" + "="*60)
    print("Sample Predictions (5 per task):")
    print("="*60)

    # Group by task
    from collections import defaultdict
    by_task = defaultdict(list)
    for pred in all_results:
        by_task[pred['expected']].append(pred)

    for task in sorted(by_task.keys()):
        task_preds = by_task[task]
        print(f"\n--- {task} ---")
        for i, pred in enumerate(task_preds[:5]):
            status = "✓" if pred['correct'] else "✗"
            print(f"{status} Q: {pred['question'][:100]}...")
            print(f"  Expected: {pred['expected']}, Predicted: {pred['predicted']}")

    # Print errors if any
    errors = [p for p in all_results if not p['correct']]
    if errors:
        print(f"\n\nTotal errors: {len(errors)}")
        print("First 10 errors:")
        for i, err in enumerate(errors[:10]):
            print(f"\n{i+1}. Q: {err['question'][:100]}...")
            print(f"   Expected: {err['expected']}, Predicted: {err['predicted']}")

if __name__ == "__main__":
    main()
