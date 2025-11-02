# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.sft.train.metrics import em, f1, extract_letter  

LOG = logging.getLogger("sft_eval_light")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def is_lora_adapter_dir(ckpt_dir: Path) -> bool:
    return (ckpt_dir / "adapter_config.json").exists()

def load_model_and_tokenizer(ckpt: str, base_model: str | None, dtype: str):
    """merged（from_pretrained(ckpt)）
       or LoRA adapter（base_model + peft）"""
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }.get(dtype, "auto")

    ckpt_path = Path(ckpt)
    if ckpt_path.exists() and ckpt_path.is_dir() and is_lora_adapter_dir(ckpt_path):
        if not base_model:
            raise ValueError("--base_model is required when --ckpt is a LoRA adapter directory")
        LOG.info("Loading base model + LoRA adapter ...")
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map={"": 0})
        model = PeftModel.from_pretrained(base, ckpt)
        model.eval()
        return tok, model
    else:
        LOG.info("Loading merged/full model ...")
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch_dtype, device_map={"": 0})
        model.eval()
        return tok, model

@torch.inference_mode()
def eval_light(dev_rows: List[dict], tok, model, device, max_eval=128, bs=2, max_new_tokens=8) -> Dict[str, float]:
    per_task = {"ARC": [], "TRIVIA": [], "IFEVAL": []}
    for r in dev_rows:
        if r.get("task") in per_task:
            per_task[r["task"]].append(r)

    k = max_eval // 3 if max_eval else None
    eval_rows = []
    if k:
        for t in ["ARC", "TRIVIA", "IFEVAL"]:
            eval_rows += per_task[t][:k]
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
            if ex["task"] == "ARC":
                arc_gold.append(ex["target"].strip().upper()[:1])
                arc_pred.append(extract_letter(text) or "")
            elif ex["task"] == "TRIVIA":
                tri_gold.append(ex["target"])
                tri_pred.append(text.splitlines()[0].strip() if text else "")
            elif ex["task"] == "IFEVAL":
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

    LOG.info("ARC Acc=%.4f | TRIVIA EM=%.4f F1=%.4f | FLAN(IFEVAL) EM=%.4f F1=%.4f | aggregate=%.4f",
             arc_acc, tri_em, tri_f1, ife_em, ife_f1, aggregate)

    LOG.info("=== Sample generations ===")
    for idx, (p, g) in enumerate(zip(tri_pred[:3], tri_gold[:3])):
        LOG.info("[TRIVIA] pred=%s | gold=%s", p, g)
    for idx, (p, g) in enumerate(zip(arc_pred[:3], arc_gold[:3])):
        LOG.info("[ARC]    pred=%s | gold=%s", p, g)

    return {
        "arc_acc": arc_acc,
        "trivia_em": tri_em, "trivia_f1": tri_f1,
        "flan_em": ife_em,   "flan_f1": ife_f1,
        "aggregate": aggregate,
    }

def main():
    ap = argparse.ArgumentParser("Lightweight dev eval (supports LoRA adapter or merged weights)")
    ap.add_argument("--ckpt", required=True, help="Path to merged model or LoRA adapter dir")
    ap.add_argument("--base_model", default=None, help="Base model id when --ckpt is a LoRA adapter")
    ap.add_argument("--dev_file", default="data/sft_dev.jsonl")
    ap.add_argument("--max_eval", type=int, default=128)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32","auto"])
    args = ap.parse_args()

    dev_rows = load_jsonl(args.dev_file)
    tok, model = load_model_and_tokenizer(args.ckpt, args.base_model, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _ = eval_light(dev_rows, tok, model, device,
                   max_eval=args.max_eval, bs=args.bs, max_new_tokens=args.max_new_tokens)

if __name__ == "__main__":
    main()
