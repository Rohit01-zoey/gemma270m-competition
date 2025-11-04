# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import List
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from .collator import SFTDataCollator

LOG = logging.getLogger("sft")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------
# Data loading utils
# ---------------------------------------------------------
def load_jsonl(path: str | Path) -> List[dict]:
    buf: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                buf.append(json.loads(line))
    return buf

def build_hf_dataset(path: str | Path) -> Dataset:
    rows = load_jsonl(path)
    return Dataset.from_list(rows)

# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser("LoRA SFT Trainer (Gemma-3-270M, train-only)")
    ap.add_argument("--model_name", default="google/gemma-3-270m")
    ap.add_argument("--train_file", default="data/sft_train.jsonl")
    ap.add_argument("--out_dir", default="checkpoints/sft")
    ap.add_argument("--max_len", type=int, default=512)

    # Training hyperparams
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--num_train_epochs", type=int, default=2)

    # Saving & logging
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=5)
    ap.add_argument("--logging_steps", type=int, default=50)

    # Precision
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    # LoRA config
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--resume_adapter", default=None,
                help="Path to a previous LoRA adapter checkpoint to continue training from (e.g., checkpoints/sft/checkpoint-2600)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Tokenizer & Base Model
    # ---------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map={"": 0},   # force model on single GPU
    )

    # ---------------------------------------------------------
    # LoRA adapter
    # ---------------------------------------------------------
    
    if args.resume_adapter:
        # continue training previous LoRA adapter
        model = PeftModel.from_pretrained(base, args.resume_adapter)
    else:
        # new LoRA adapter
        lconf = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        )
        model = get_peft_model(base, lconf)
    model.print_trainable_parameters()

    # Optimize for stability & memory
    model.config.use_cache = False

    # ---------------------------------------------------------
    # Dataset & Data Collator
    # ---------------------------------------------------------
    train_ds = build_hf_dataset(args.train_file)
    collator = SFTDataCollator(tokenizer=tok, max_len=args.max_len)

    # ---------------------------------------------------------
    # TrainingArguments (no eval)
    # ---------------------------------------------------------
    targs = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        logging_steps=args.logging_steps,

        # ✅ train-only, no evaluation
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,

        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
    )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    LOG.info("Starting SFT training (no eval)...")
    trainer.train()
    LOG.info("Training completed.")

    # ---------------------------------------------------------
    # Final save
    # ---------------------------------------------------------
    final_dir = out_dir / "sft_final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    LOG.info("Final model saved to: %s", final_dir)


if __name__ == "__main__":
    main()
