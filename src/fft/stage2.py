from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizerBase,
)

def apply_instruct_template(prompt, tokenizer: PreTrainedTokenizerBase) -> str:
    has_chat = tokenizer and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template
    if has_chat:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(prompt, list):
        lines: List[str] = []
        for m in prompt:
            role = m.get("role", "user")
            prefix = "User" if role == "user" else ("Assistant" if role == "assistant" else role.capitalize())
            lines.append(f"{prefix}: {m.get('content', '')}")
        return "\n".join(lines)
    return str(prompt)

@dataclass
class Stage2Config:
    model_name: str
    train_path: str
    eval_path: Optional[str]
    max_length: int
    output_dir: str

    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    warmup_ratio: float
    weight_decay: float

    seed: int
    bf16: bool
    fp16: bool


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows

def build_user_text(rec: Dict[str, Any]) -> str:
    """
    Build the *user-facing* string that eval also uses, BEFORE
    going through apply_instruct_template.

    Fields expected in rec:
      task: "TRIVIA" | "ARC" | "IFEVAL"
      instruction: str
      input: optional str
    """
    task = rec.get("task", "TRIVIA")
    instr = (rec.get("instruction") or "").strip()
    inp = (rec.get("input") or "").strip()

    if task == "ARC":
        if inp:
            return instr + "\n\n" + inp
        return instr

    if task == "IFEVAL":
        if inp:
            return instr + "\n\nInput:\n" + inp
        return instr
    
    if inp:
        return instr + "\n\nQuestion:\n" + inp
    return instr


def build_full_prompt(rec: Dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> str:
    """
    Turn a record into the *actual* model input string by:
      user_text -> apply_instruct_template(user_text, tokenizer)
    This MUST match what the standard eval does.
    """
    user_text = build_user_text(rec)
    prompt = apply_instruct_template(user_text, tokenizer)
    return prompt

class Stage2Dataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.rows[idx]
        target = (rec.get("target") or "").strip()
        if not target:
            target = ""

        prompt = build_full_prompt(rec, self.tokenizer)

        enc_prompt = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_ids = enc_prompt["input_ids"]
        prompt_len = len(prompt_ids)

        full_text = prompt + target + self.tokenizer.eos_token
        enc_full = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc_full["input_ids"]

        labels = [-100] * len(input_ids)
        for t in range(prompt_len, len(input_ids)):
            labels[t] = input_ids[t]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ids_list = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        labs_list = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]

        ids = torch.nn.utils.rnn.pad_sequence(
            ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labs = torch.nn.utils.rnn.pad_sequence(
            labs_list,
            batch_first=True,
            padding_value=-100,
        )

        ids = ids[:, : self.max_length]
        labs = labs[:, : self.max_length]
        attn = (ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labs,
        }

def build_config(args: argparse.Namespace) -> Stage2Config:
    return Stage2Config(
        model_name=args.model_name,
        train_path=args.train_path,
        eval_path=args.eval_path,
        max_length=args.max_length,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
    )

def train_stage2(cfg: Stage2Config):
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg.max_length

    train_rows = load_jsonl(cfg.train_path)
    train_ds = Stage2Dataset(train_rows, tokenizer, cfg.max_length)

    eval_ds = None
    if cfg.eval_path:
        eval_rows = load_jsonl(cfg.eval_path)
        eval_ds = Stage2Dataset(eval_rows, tokenizer, cfg.max_length)

    collator = DataCollator(tokenizer, cfg.max_length)

    if cfg.bf16:
        dtype = torch.bfloat16
    elif cfg.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
    )

    for p in model.parameters():
        p.requires_grad = True

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

def parse_args():
    ap = argparse.ArgumentParser(description="Stage 2 SFT (benchmark-focused, Gemma3 270M)")

    ap.add_argument("--model_name", type=str, required=True,
                    help="Path or HF id of Stage-1 checkpoint.")
    ap.add_argument("--train_path", type=str, required=True,
                    help="stage2_train.jsonl")
    ap.add_argument("--eval_path", type=str, default=None,
                    help="stage2_eval.jsonl (optional)")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--output_dir", type=str, default="ckpts/stage2_sft")

    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)
    train_stage2(cfg)


if __name__ == "__main__":
    main()
