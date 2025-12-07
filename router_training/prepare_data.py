#!/usr/bin/env python3
"""
Prepare router training and evaluation data.

Training: smol-smoltalk (train + test splits)
Evaluation: ARC-C, TriviaQA, IFEval test splits

Usage: python router_training/prepare_data.py
"""
import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset

# Map task types to routed model types
# TODO: Expand when adding more specialized models
TASK_MAP = {
    "ARC": "reasoning",
    "TRIVIA": "factual_qa",
    "IFEVAL": "instruction_following",
}

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def convert_smoltalk(example):
    """Convert smol-smoltalk to router format."""
    messages = example.get('messages', [])
    question = None
    for msg in messages:
        if msg.get('role') == 'user':
            question = msg.get('content', '').strip()
            break
    if not question:
        return None

    # Map category to routed model
    # TODO: Add mapping for smol-smoltalk categories
    routed_model = "instruction_following"  # Default
    # category = example.get('category', '')
    # if 'math' in category or 'stem' in category:
    #     routed_model = "reasoning"
    # else:
    #     routed_model = "factual_qa"  # Default

    prompt = f"Classify task type:\n{question}\nTask:"
    return {"prompt": prompt, "target": routed_model}

def convert_eval_data(item, task_type):
    """Convert eval data (ARC/TRIVIA/IFEVAL) to router format."""
    routed_model = TASK_MAP[task_type]
    question = item.get('question') or item.get('prompt', '')

    prompt = f"Classify task type:\n{question.strip()}\nTask:"
    return {"prompt": prompt, "target": routed_model}

def main():
    import random
    random.seed(42)

    # === Load training data from all sources ===
    print("Loading training data...")

    # 1. ARC-C from sft_train.jsonl
    sft_train = load_jsonl("data/sft_train.jsonl")
    arc_data = [convert_eval_data(x, "ARC") for x in sft_train if x.get('task') == 'ARC']
    print(f"ARC-C training: {len(arc_data)} examples")

    # 2. TriviaQA from sft_train.jsonl
    trivia_data = [convert_eval_data(x, "TRIVIA") for x in sft_train if x.get('task') == 'TRIVIA']
    print(f"TriviaQA training: {len(trivia_data)} examples")

    # 3. smol-smoltalk for instruction_following
    ds = load_dataset("HuggingFaceTB/smol-smoltalk")
    smol_data = []
    if 'train' in ds:
        smol_data.extend([convert_smoltalk(x) for x in ds['train']])
    if 'test' in ds:
        smol_data.extend([convert_smoltalk(x) for x in ds['test']])
    smol_data = [x for x in smol_data if x]
    print(f"smol-smoltalk: {len(smol_data)} examples")

    # === Balance datasets ===
    min_count = min(len(arc_data), len(trivia_data), len(smol_data))
    print(f"\nBalancing to {min_count} examples per task type...")

    # Sample equally from each
    random.shuffle(arc_data)
    random.shuffle(trivia_data)
    random.shuffle(smol_data)

    balanced_data = (
        arc_data[:min_count] +
        trivia_data[:min_count] +
        smol_data[:min_count]
    )
    random.shuffle(balanced_data)

    # Split 90/10 for train/dev
    split = int(len(balanced_data) * 0.9)
    train = balanced_data[:split]
    dev = balanced_data[split:]

    save_jsonl("router_training/data/train.jsonl", train)
    save_jsonl("router_training/data/dev.jsonl", dev)

    print(f"\nBalanced training data: {len(train)}")
    for task, count in Counter(x['target'] for x in train).items():
        print(f"  {task}: {count}")

    print(f"\nBalanced dev data: {len(dev)}")
    for task, count in Counter(x['target'] for x in dev).items():
        print(f"  {task}: {count}")

    # === Evaluation data from cs590_eval ===
    print("\n" + "="*60)
    print("Preparing evaluation data (sampled for faster evaluation)...")

    # Use smaller eval size for faster testing
    eval_per_task = 200  # 200 examples per task = 600 total

    # ARC-C
    arc_eval = load_jsonl("cs590_eval/data/arc_c_test.jsonl")
    arc_eval_converted = [convert_eval_data(x, "ARC") for x in arc_eval]
    random.shuffle(arc_eval_converted)
    arc_eval_sampled = arc_eval_converted[:eval_per_task]
    print(f"ARC-C: {len(arc_eval_sampled)} examples (sampled from {len(arc_eval)})")

    # TriviaQA
    trivia_eval = load_jsonl("cs590_eval/data/triviaqa_test.jsonl")
    trivia_eval_converted = [convert_eval_data(x, "TRIVIA") for x in trivia_eval]
    random.shuffle(trivia_eval_converted)
    trivia_eval_sampled = trivia_eval_converted[:eval_per_task]
    print(f"TriviaQA: {len(trivia_eval_sampled)} examples (sampled from {len(trivia_eval)})")

    # IFEval
    ifeval_eval = load_jsonl("cs590_eval/data/ifeval_test.jsonl")
    ifeval_eval_converted = [convert_eval_data(x, "IFEVAL") for x in ifeval_eval]
    random.shuffle(ifeval_eval_converted)
    ifeval_eval_sampled = ifeval_eval_converted[:eval_per_task]
    print(f"IFEval: {len(ifeval_eval_sampled)} examples (sampled from {len(ifeval_eval)})")

    eval_data = arc_eval_sampled + trivia_eval_sampled + ifeval_eval_sampled
    random.shuffle(eval_data)

    save_jsonl("router_training/data/eval.jsonl", eval_data)

    print(f"\nTotal eval: {len(eval_data)}")
    for task, count in Counter(x['target'] for x in eval_data).items():
        print(f"  {task}: {count}")

    print("\nDone! Saved to router_training/data/")

if __name__ == "__main__":
    main()
