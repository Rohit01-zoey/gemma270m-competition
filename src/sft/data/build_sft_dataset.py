# -*- coding: utf-8 -*-
"""
Build SFT dataset for Gemma-3-270M with ARC, TriviaQA and FLAN (supervised).
- ARC: ai2_arc / ARC-Challenge (train split)
- TriviaQA: trivia_qa / rc (train split)
- FLAN: openflan/openflan-submix-originals (train split), fields: inputs/targets

Output:
  data/sft_train.jsonl
  data/sft_dev.jsonl
Each line: {"task": "...", "prompt": "...", "target": "..."}

Notes:
- We DO NOT use google/IFEval here because it has no gold responses. It should be used for RL/reward.
- Keep templates identical to evaluation to avoid train/eval drift.
"""
from __future__ import annotations
import argparse
from typing import List, Sequence
from datasets import load_dataset, VerificationMode
from src.sft.prompts.templates import ARC, TRIVIA, IFEVAL
from .schema import Sample
from .io import write_jsonl, stratified_split


# ----------------------- ARC ----------------------- #
def _choices_map_arc(ex: dict) -> dict:
    ch = ex["choices"]
    labels = [str(x).strip().upper() for x in ch["label"]]
    texts = ch["text"]
    return dict(zip(labels, texts))

def build_arc(limit: int | None = None) -> List[Sample]:
    ds = load_dataset("ai2_arc", "ARC-Challenge")["train"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out: List[Sample] = []
    for ex in ds:
        cm = _choices_map_arc(ex)
        prompt = ARC.format(
            q=ex["question"],
            A=cm.get("A", ""), B=cm.get("B", ""),
            C=cm.get("C", ""), D=cm.get("D", "")
        )
        target = ex["answerKey"].strip().upper()[:1]  # "A"/"B"/"C"/"D"
        if target in {"A", "B", "C", "D"}:
            out.append(Sample(task="ARC", prompt=prompt, target=target))
    return out


# ----------------------- TriviaQA ----------------------- #
def build_trivia(limit: int | None = None) -> List[Sample]:
    ds = load_dataset("trivia_qa", "rc")["train"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out: List[Sample] = []
    for ex in ds:
        q = (ex.get("question") or "").strip()
        ans = ex.get("answer", {}).get("value", "")
        if not q or not ans:
            continue
        prompt = TRIVIA.format(q=q)
        # Keep target concise; downstream postprocess will normalize.
        out.append(Sample(task="TRIVIA", prompt=prompt, target=str(ans).strip()))
    return out


# ----------------------- FLAN (supervised) ----------------------- #
def build_flan_openflan(
    limit: int | None = None,
    allowed_subsets: Sequence[str] | None = None,
) -> List[Sample]:
    """
    Use OpenFLAN supervised instruction data.
    Dataset: openflan
    Important fields: 'inputs' (instruction), 'targets' (answer), 'subset'
    """
    ds = load_dataset("Muennighoff/flan", verification_mode=VerificationMode.NO_CHECKS)["train"]
    # Optional: filter by subset to control domain/style
    if allowed_subsets:
        ds = ds.filter(lambda ex: ex.get("subset") in set(allowed_subsets))

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    out: List[Sample] = []
    for ex in ds:
        instr = (ex.get("inputs") or "").strip()
        resp = (ex.get("targets") or "").strip()
        if not instr or not resp:
            continue
        # Reuse IFEVAL-style template for instruction-following format
        prompt = IFEVAL.format(instr=instr)
        out.append(Sample(task="IFEVAL", prompt=prompt, target=resp))
    return out


# ----------------------- CLI ----------------------- #
def main() -> None:
    ap = argparse.ArgumentParser("Build SFT dataset (ARC/Trivia/FLAN)")
    ap.add_argument("--out-train", default="data/sft_train.jsonl")
    ap.add_argument("--out-dev", default="data/sft_dev.jsonl")

    # sizes
    ap.add_argument("--arc", type=int, default=10000, help="ARC train samples")
    ap.add_argument("--trivia", type=int, default=10000, help="Trivia train samples")
    ap.add_argument("--flan", type=int, default=20000, help="FLAN supervised samples")

    # OpenFLAN subset filter (optional)
    ap.add_argument(
        "--flan_subsets",
        nargs="*",
        default=[],  # empty => use all subsets
        help="Limit to specific OpenFLAN subsets, e.g., --flan_subsets chain_of_thought flan_cot"
    )

    ap.add_argument("--dev_ratio", type=float, default=0.1)
    args = ap.parse_args()

    # Build per-task lists
    arc = build_arc(limit=args.arc)
    trivia = build_trivia(limit=args.trivia)
    flan = build_flan_openflan(limit=args.flan,
                               allowed_subsets=(args.flan_subsets or None))

    # Merge and stratified split (preserve task proportions)
    all_items = arc + trivia + flan
    train, dev = stratified_split(all_items, dev_ratio=args.dev_ratio, seed=42)

    write_jsonl(args.out_train, train)
    write_jsonl(args.out_dev, dev)

    print(
        f"[OK] train={len(train)} dev={len(dev)} | "
        f"ARC={len(arc)} TRIVIA={len(trivia)} FLAN={len(flan)}"
    )

if __name__ == "__main__":
    main()
