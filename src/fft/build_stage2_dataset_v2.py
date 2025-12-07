# src/fft/data/stage2_data.py

# trivia style 
#   - triviaqa
# mcq
#   - ARC-C
#   - Arc easy
#   - MMLU

from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

from datasets import load_dataset


Record = Dict[str, Any]
import random

TEMPLATES = [
    "Question: {q}",
    "Q: {q}",
    "{q}",
]

def format_question(raw_q: str, rng: random.Random) -> str:
    tpl = rng.choice(TEMPLATES)
    return tpl.format(q=raw_q.strip())


# ---------------- Basic IO helpers ----------------
def format_choices(choice_list):
    letters = [chr(ord("A") + i) for i in range(len(choice_list))]
    return "\n".join(f"{letters[i]}. {choice_list[i]}" for i in range(len(choice_list)))


def save_jsonl(path: str | Path, rows: List[Record]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_eval(
    rows: List[Record],
    eval_frac: float,
    seed: int = 42
) -> Tuple[List[Record], List[Record]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_eval = int(len(rows) * eval_frac)
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]
    return train_rows, eval_rows


# ---------------- TRIVIA / QA loaders ----------------

def sample_triviaqa_unfiltered(n: int, seed: int) -> List[Record]:
    ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="train")
    ds = ds.shuffle(seed=seed)
    if 0 < n < len(ds):
        ds = ds.select(range(n))
    rng = random.Random(seed)
    print(f"[TRIVIAQA] loading...chosen {len(ds)}")

    rows: List[Record] = []
    for ex in ds:
        q = format_question((ex["question"] or "").strip(), rng)
        ans = ex.get("answer", {}) or {}
        aliases = ans.get("normalized_aliases") or []
        main_val = ans.get("normalized_value") or ans.get("value")
        golds: List[str] = [str(a) for a in aliases] if aliases else ([str(main_val)] if main_val else [])

        rows.append({
            "source": "trivia_qa",
            # "instruction": "Answer the following question with a short phrase or name.",
            # "question": q,
            # "target": golds[0],
            "messages": [{"content": str(q), "role":"user"},
                        {"content": str(golds[0]), "role":"assistant"}],
            # "ds_metadata": {}
        })
    print(f"[TRIVIAQA] {len(rows)} examples.")
    return rows


def sample_mmlu(n: int, seed: int) -> List[Record]:
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    ds = ds.shuffle(seed=seed)
    if 0 < n < len(ds):
        ds = ds.select(range(n))
    rng = random.Random(seed)
    print(f"[MMLU] loading...chosen {len(ds)}")

    rows: List[Record] = []
    for ex in ds:
        q = format_question(ex.get("question"), rng)
        choice_list = format_choices(ex.get("choices"))
        q_full = q+"\n"+choice_list
        target = chr(ord("A") + int(ex.get("answer")))

        rows.append({
            "source": "mmlu",
            # "instruction": "Answer the following question with the correct option choice.",
            # "question": q_full,
            # "target": target,
            "messages": [{"content": str(q_full), "role":"user"},
                        {"content": str(target), "role":"assistant"}],
            # "ds_metadata": {"subject": str(ex.get("subject"))}
        })

    return rows

def sample_gsm8k(n: int, seed: int) -> List[Record]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed)
    if 0 < n < len(ds):
        ds = ds.select(range(n))
    rng = random.Random(seed)
    print(f"[GSM8K] loading...chosen {len(ds)}")
    
    rows: List[Record] = []
    for ex in ds:
        q = format_question(ex.get("question"), rng)
        ans = ex.get("answer")

        rows.append({
            "source": "gsm8k",
            # "instruction": "Reason briefly and then answer the question",
            # "question": q,
            # "target": ans,
            "messages": [{"content": str(q), "role":"user"},
                        {"content": str(ans), "role":"assistant"}],
            # "ds_metadata": {}
        })

    return rows

def sample_arc(n: int, seed: int, which: str = "ARC-Challenge", over_sample=1) -> List[Record]:
    ds = load_dataset("ai2_arc", which, split="train")
    ds = ds.shuffle(seed=seed)
    if 0 < n < len(ds):
        ds = ds.select(range(n))
    rng = random.Random(seed)
    print(f"[ARC] loading...chosen {len(ds)}")
    
    
    rows: List[Record] = []
    for ex in ds:
        q = format_question(ex.get("question"), rng)
        choice_list = format_choices(ex.get("choices")["text"])
        q_full = q+"\n"+choice_list
        ans = ex.get("answerKey")

        rows.append({
            "source": str(which).lower(),
            # "instruction": "Answer the following question with the correct option choice.",
            # "question": q_full,
            # "target": ans,
            "messages": [{"content": str(q_full), "role":"user"},
                        {"content": str(ans), "role":"assistant"}],
            # "ds_metadata": {}
        })
    if over_sample>1:
        return rows*over_sample
    else:
        return rows

def sample_smol(n: int, seed: int) -> List[Record]:
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    ds = ds.shuffle(seed=seed)
    if 0 < n < len(ds):
        ds = ds.select(range(n))
    print(f"[SMOL] loading...chosen {len(ds)}")
    rows: List[Record] = []
    for ex in ds:
        rows.append({
            "source": "smol",
            "messages": ex.get("messages"),
            # "ds_metadata": {}
        })
    return rows

@dataclass
class Stage2DataConfig:
    output_dir: str = "data/stage2_v2"
    eval_frac: float = 0.02
    seed: int = 42

    # ~100k QA / Trivia / Reasoning
    n_trivia_qa: int = 0 # 20_000
    n_gsm8k: int = 0 # 7_000      # NEW
    n_mmlu: int = 0 # 20_000      # NEW

    # ~50k ARC-style MCQ
    n_arc_challenge: int = 16_000 # actual only ~1k
    n_arc_easy: int = 0 # 12_000 # actual only ~2k

    # ~50k IFEval / constraints
    n_smol: int = 0



def build_stage2_records(cfg: Stage2DataConfig) -> List[Record]:
    random.seed(cfg.seed)
    all_rows: List[Record] = []

    # QA / Trivia
    if cfg.n_trivia_qa>0:
        all_rows += sample_triviaqa_unfiltered(cfg.n_trivia_qa, cfg.seed)
    if cfg.n_mmlu>0:
        all_rows += sample_mmlu(cfg.n_mmlu, cfg.seed)
    if cfg.n_gsm8k>0:
        all_rows += sample_gsm8k(cfg.n_gsm8k, cfg.seed)
    if cfg.n_arc_challenge>0:
        all_rows += sample_arc(cfg.n_arc_challenge, cfg.seed, which="ARC-Challenge", over_sample=10)
    if cfg.n_arc_easy>0:
        all_rows += sample_arc(cfg.n_arc_easy, cfg.seed, which="ARC-Easy", over_sample=3)
    if cfg.n_smol>0:
        all_rows += sample_smol(cfg.n_smol, cfg.seed)

    random.shuffle(all_rows)
    print(f"[STAGE2] total records: {len(all_rows)}")
    return all_rows


def prepare_stage2_data(cfg: Stage2DataConfig) -> Dict[str, str]:
    """
    Hyper-style function: builds all Stage 2 training data,
    saves JSONLs, and returns a dict with paths.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = build_stage2_records(cfg)
    train_rows, eval_rows = split_train_eval(all_rows, cfg.eval_frac, cfg.seed)

    train_path = out_dir / "stage2_train.jsonl"
    eval_path = out_dir / "stage2_eval.jsonl"

    print(f"[STAGE2] Saving train ({len(train_rows)}) -> {train_path}")
    save_jsonl(train_path, train_rows)

    print(f"[STAGE2] Saving eval  ({len(eval_rows)}) -> {eval_path}")
    save_jsonl(eval_path, eval_rows)

    return {
        "train": str(train_path),
        "eval": str(eval_path),
    }


# ---------------- CLI entrypoint ----------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build Stage 2 dataset (downloads + mixes HF datasets).")
    ap.add_argument("--output_dir", type=str, default="data/stage2_v2")
    ap.add_argument("--eval_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_trivia_qa", type=int)
    ap.add_argument("--n_gsm8k", type=int)
    ap.add_argument("--n_mmlu", type=int)
    ap.add_argument("--n_arc_challenge", type=int)
    ap.add_argument("--n_arc_easy", type=int)
    ap.add_argument("--n_smol", type=int)

    args = ap.parse_args()

    cfg = Stage2DataConfig()


    paths = prepare_stage2_data(cfg)
    print("\n[STAGE2] Done.")
    print("Train path:", paths["train"])
    print("Eval  path:", paths["eval"])
