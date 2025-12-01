from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    # dataset = load_dataset("HuggingFaceTB/smol-smoltalk")
    dataset ={}
    train_rows = load_jsonl("/usr/xtmp/rkv6/projects/gemma270m-competition/src/fft/data/stage2_v2/stage2_train.jsonl")
    test_rows = load_jsonl("/usr/xtmp/rkv6/projects/gemma270m-competition/src/fft/data/stage2_v2/stage2_eval.jsonl")

    dataset["train"] = Dataset.from_list(train_rows)
    dataset["test"] = Dataset.from_list(test_rows)
    # Configure model and tokenizer
    model_name = "google/gemma-3-270m"
    ckpt = "/usr/xtmp/rkv6/projects/gemma270m-competition/src/fft/sft_output/checkpoint-20000"
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=ckpt).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=ckpt)
    
    def formatting_func(example):
        return [
            tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,  # assistant message already included
            )
        ]

    # Configure trainer
    training_args = SFTConfig(
        output_dir="./sft_output_stage2",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=100
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    
    # ---- Sanity check: are labels masked correctly? ----
    batch = next(iter(trainer.get_train_dataloader()))
    ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    print("First ~80 token/label pairs:")
    for tid, lab in zip(ids[:80], labels[:80]):
        print(repr(tokenizer.decode([tid])), lab.item())
        
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()