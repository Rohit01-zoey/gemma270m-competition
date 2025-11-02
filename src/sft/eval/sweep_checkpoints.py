# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, logging, re, time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------- Logging ----------
LOG = logging.getLogger("sft_sweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------- Small helpers ----------
def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def is_lora_adapter_dir(d: Path) -> bool:
    return (d / "adapter_config.json").exists()

def list_candidate_checkpoints(root: Path, pattern: str | None) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"ckpt_root not found: {root}")
    cands = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if pattern and not re.search(pattern, name):
            continue
        if name.startswith("checkpoint-") or name.startswith("merged-") or name == "sft_final":
            cands.append(p)
    more = list(root.rglob("checkpoint-*")) + list(root.rglob("merged-*"))
    for p in more:
        if p.is_dir() and p not in cands:
            if not pattern or re.search(pattern, p.name):
                cands.append(p)
    def sort_key(p: Path):
        m = re.search(r"checkpoint-(\d+)", p.name)
        return (int(m.group(1)) if m else 10**12, p.name)
    cands.sort(key=sort_key)
    return cands

def load_model_and_tokenizer(ckpt: Path, base_model: str | None, dtype: str):
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }.get(dtype, "auto")

    if is_lora_adapter_dir(ckpt):
        if not base_model:
            raise ValueError(f"--base_model is required for LoRA adapter: {ckpt}")
        LOG.info("Loading base + LoRA adapter: %s", ckpt)
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map={"": 0})
        model = PeftModel.from_pretrained(base, str(ckpt))
        model.eval()
        return tok, model
    else:
        LOG.info("Loading merged/full model: %s", ckpt)
        tok = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=torch_dtype, device_map={"": 0})
        model.eval()
        return tok, model

# ---------- Metrics (reuse of your simple ones) ----------
import re as _re, string as _string
def _normalize_text(s: str) -> str:
    ARTICLES = _re.compile(r"\b(a|an|the)\b", _re.IGNORECASE)
    PUNCT = _re.compile("[" + _re.escape(_string.punctuation) + "]")
    s = s.strip().lower()
    s = PUNCT.sub(" ", s)
    s = ARTICLES.sub(" ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s
def em(a: str, b: str) -> int:
    return int(_normalize_text(a) == _normalize_text(b))
def f1(a: str, b: str) -> float:
    a, b = _normalize_text(a), _normalize_text(b)
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return float(a == b)
    common = {}
    for w in ta:
        common[w] = min(ta.count(w), tb.count(w))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)
def extract_letter(s: str) -> str | None:
    m = _re.search(r"\b([ABCD])\b", s.strip(), _re.I)
    return m.group(1).upper() if m else None

# ---------- Lightweight evaluation ----------
@torch.inference_mode()
def eval_light(dev_rows: List[dict], tok, model, device, max_eval=128, bs=2, max_new_tokens=8) -> Dict[str, float]:
    buckets = {"ARC": [], "TRIVIA": [], "IFEVAL": []}
    for r in dev_rows:
        t = r.get("task")
        if t in buckets:
            buckets[t].append(r)
    k = max_eval // 3 if max_eval else None
    eval_rows = []
    if k:
        for t in ["ARC", "TRIVIA", "IFEVAL"]:
            eval_rows += buckets[t][:k]
    else:
        eval_rows = dev_rows

    old_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True

    arc_gold, arc_pred = [], []
    tri_gold, tri_pred = [], []
    ife_gold, ife_pred = [], []

    for i in range(0, len(eval_rows), bs):
        chunk = eval_rows[i:i+bs]
        prompts = [c["prompt"] for c in chunk]
        enc = tok(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        input_lens = enc["attention_mask"].sum(dim=1)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        outs = [out[j, input_lens[j]:].detach().cpu() for j in range(len(chunk))]
        texts = tok.batch_decode(outs, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        for text, ex in zip(texts, chunk):
            task = ex["task"]
            if task == "ARC":
                arc_gold.append(ex["target"].strip().upper()[:1])
                arc_pred.append(extract_letter(text) or "")
            elif task == "TRIVIA":
                tri_gold.append(ex["target"])
                tri_pred.append(text.splitlines()[0].strip() if text else "")
            elif task == "IFEVAL":
                ife_gold.append(ex["target"])
                ife_pred.append(text.splitlines()[0].strip() if text else "")

    if old_cache is not None:
        model.config.use_cache = old_cache

    arc_acc = sum(int(p == g) for p, g in zip(arc_pred, arc_gold)) / max(len(arc_gold), 1)
    tri_em  = sum(em(p, g) for p, g in zip(tri_pred, tri_gold)) / max(len(tri_gold), 1)
    tri_f1  = sum(f1(p, g) for p, g in zip(tri_pred, tri_gold)) / max(len(tri_gold), 1)
    ife_em  = sum(em(p, g) for p, g in zip(ife_pred, ife_gold)) / max(len(ife_gold), 1) if ife_gold else 0.0
    ife_f1  = sum(f1(p, g) for p, g in zip(ife_pred, ife_gold)) / max(len(ife_gold), 1) if ife_gold else 0.0
    aggregate = 0.45 * arc_acc + 0.30 * tri_f1 + 0.25 * ife_f1

    return {
        "arc_acc": arc_acc,
        "trivia_em": tri_em, "trivia_f1": tri_f1,
        "flan_em": ife_em,   "flan_f1": ife_f1,
        "aggregate": aggregate,
    }

# ---------- Sweep runner ----------
def main():
    ap = argparse.ArgumentParser("Sweep checkpoints, evaluate, and pick the best")
    ap.add_argument("--ckpt_root", default="checkpoints/sft", help="Folder that contains checkpoint-* or merged-*")
    ap.add_argument("--base_model", default="google/gemma-3-270m", help="Required for LoRA adapters")
    ap.add_argument("--dev_file", default="data/sft_dev.jsonl")
    ap.add_argument("--pattern", default=None, help="Regex to filter checkpoint names (e.g. '^checkpoint-')")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32","auto"])
    ap.add_argument("--max_eval", type=int, default=128)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    dev_rows = load_jsonl(Path(args.dev_file))
    cands = list_candidate_checkpoints(root, args.pattern)
    if not cands:
        raise SystemExit(f"No checkpoints found under {root}")

    results: List[Tuple[str, Dict[str, float], float]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ckpt in cands:
        LOG.info("===== Evaluating: %s =====", ckpt)
        t0 = time.time()
        try:
            tok, model = load_model_and_tokenizer(ckpt, args.base_model, args.dtype)
            model.to(device)
            metrics = eval_light(dev_rows, tok, model, device,
                                 max_eval=args.max_eval, bs=args.bs, max_new_tokens=args.max_new_tokens)
            elapsed = time.time() - t0
            LOG.info("Done in %.1fs | aggregate=%.4f | arc=%.4f | trivia_f1=%.4f | ifeval_f1=%.4f",
                     elapsed, metrics["aggregate"], metrics["arc_acc"], metrics["trivia_f1"], metrics["flan_f1"])
            results.append((str(ckpt), metrics, elapsed))
        except Exception as e:
            LOG.exception("Failed on %s: %s", ckpt, e)
            results.append((str(ckpt), {"aggregate": -1.0}, float("inf")))
        finally:
            try:
                del model
            except:
                pass
            torch.cuda.empty_cache()

    results.sort(key=lambda x: x[1].get("aggregate", -1.0), reverse=True)

    jsonl_path = root / "sweep_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for ck, m, sec in results:
            rec = {"checkpoint": ck, "seconds": sec, **m}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    csv_path = root / "sweep_results.csv"
    cols = ["checkpoint", "aggregate", "arc_acc", "trivia_em", "trivia_f1", "flan_em", "flan_f1", "seconds"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ck, m, sec in results:
            row = {"checkpoint": ck, "seconds": f"{sec:.1f}"}
            row.update({k: f"{m.get(k, float('nan')):.4f}" if isinstance(m.get(k), (int,float)) else m.get(k, "") for k in cols if k not in ["checkpoint","seconds"]})
            w.writerow(row)

    best_ckpt = results[0][0]
    (root / "best_checkpoint.txt").write_text(best_ckpt, encoding="utf-8")
    LOG.info("🏁 Best checkpoint: %s", best_ckpt)
    LOG.info("Saved: %s | %s", jsonl_path, csv_path)

if __name__ == "__main__":
    main()
