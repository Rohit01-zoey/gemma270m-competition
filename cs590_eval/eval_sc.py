import argparse
import json
import os
from typing import Any, Dict, List

from pipelines import (
    RouterPipeline,
    SelfConsistencyPipeline,
)
from generate import generate_rows, _write_jsonl, _read_jsonl
from score import score_from_rows


def __default_data_file_for_task(task: str) -> str:
    if task == "triviaqa":
        return "data/triviaqa_test.jsonl"
    if task == "arc-c":
        return "data/arc_c_test.jsonl"
    if task == "ifeval":
        return "data/ifeval_test.jsonl"
    raise ValueError(f"Unknown task: {task}")




def main():
    ap = argparse.ArgumentParser(description="Given a task, generate predictions and then score them from a single unified JSONL (with optional self-consistency)")
    ap.add_argument("--task", choices=["triviaqa", "arc-c", "ifeval"], default="arc-c")
    ap.add_argument("--data_file", default=None, help="Optional explicit data file; otherwise chosen by task")
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Directory to save predictions and metrics")
    ap.add_argument("--data-size", type=int, default=1000, help="Number of data items to process; -1 means all")

    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--skip-existing", action="store_true", help="Skip generation if predictions file already exists")

    # Self-consistency argument
    ap.add_argument("--num-samples", type=int, default=1, help="Number of samples for self-consistency (1=disabled, 5+ recommended)")

    args = ap.parse_args()
    data_file = args.data_file or __default_data_file_for_task(args.task)
    items = _read_jsonl(data_file)
    items = items[: args.data_size if args.data_size > 0 else None]

    # Create detailed filename with model, data size, batch size, and max new tokens
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.basename(data_file).replace(".jsonl", "")
    model_name = args.model.replace("/", "-").replace("\\", "-")

    # Build detailed filename
    filename_parts = [
        base,
        model_name,
        f"n{args.data_size if args.data_size > 0 else 'all'}",
        f"bs{args.batch_size}",
        f"maxtok{args.max_new_tokens}"
    ]

    # Add self-consistency marker to filename if enabled
    if args.num_samples > 1:
        filename_parts.append(f"sc{args.num_samples}")

    filename_base = "_".join(filename_parts)

    preds_path = os.path.join(args.out_dir, f"{filename_base}_preds.jsonl")
    metrics_path = os.path.join(args.out_dir, f"{filename_base}_metrics.json")

    # Check if predictions already exist
    if args.skip_existing and os.path.exists(preds_path):
        print(f"Predictions already exist at {preds_path}, loading from file...")
        rows = _read_jsonl(preds_path)
    else:
        # Generate predictions using appropriate pipeline
        # Self-consistency is ONLY enabled for ARC-C
        if args.num_samples > 1 and args.task == "arc-c":
            print(f"Using SelfConsistencyPipeline with {args.num_samples} samples per input (ARC-C only)")
            pipeline = SelfConsistencyPipeline(args.model, num_samples=args.num_samples)
        else:
            if args.num_samples > 1:
                print(f"Note: Self-consistency is only implemented for ARC-C. Using standard RouterPipeline for {args.task}")
            else:
                print("Using standard RouterPipeline")
            pipeline = RouterPipeline(args.model)

        # Route by task_type values produced by downloader: triviaqa, arc-c, ifeval
        rows = generate_rows(
            items,
            pipeline,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            )

        _write_jsonl(preds_path, rows)
        print(f"Predictions saved to: {preds_path}")

    # Score using reusable scorer on the same unified rows
    result = score_from_rows(args.task, rows)

    # Save metrics to file
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print(json.dumps({
        "task": args.task,
        "n": len(rows),
        "preds": os.path.abspath(preds_path),
        "metrics": os.path.abspath(metrics_path),
        "num_samples": args.num_samples,
        **result,
    }, indent=2))


if __name__ == "__main__":
    main()
