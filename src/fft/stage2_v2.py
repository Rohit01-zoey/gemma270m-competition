from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

@dataclass
class DataCollatorForCompletionOnlyLM:
    tokenizer: Any
    response_template: str

    def __post_init__(self):
        self.response_template_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()

        for i, input_ids in enumerate(batch["input_ids"]):
            response_start = self._find_response_start(input_ids.tolist())
            if response_start is not None:
                labels[i, :response_start] = -100
            else:
                labels[i, :] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    def _find_response_start(self, input_ids: List[int]) -> int | None:
        template_len = len(self.response_template_ids)
        for i in range(len(input_ids) - template_len + 1):
            if input_ids[i:i + template_len] == self.response_template_ids:
                return i + template_len
        return None


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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = {}
    train_rows = load_jsonl("/usr/xtmp/rkv6/projects/gemma270m-competition/src/fft/data/stage2_v2/stage2_train.jsonl")
    test_rows = load_jsonl("/usr/xtmp/rkv6/projects/gemma270m-competition/src/fft/data/stage2_v2/stage2_eval.jsonl")
    dataset["train"] = Dataset.from_list(train_rows)
    dataset["test"] = Dataset.from_list(test_rows)

    ckpt = "/usr/xtmp/rkv6/projects/gemma270m-competition/src/fft/sft_output/checkpoint-20000"
    model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    sample_formatted = formatting_func(train_rows[0])
    print("Sample formatted text:")
    print(repr(sample_formatted))
    
    response_template = "<|im_start|>assistant\n"
    
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
    )

    training_args = SFTConfig(
        output_dir="./sft_output_stage2",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        formatting_func=formatting_func,
        data_collator=collator,
    )

    batch = next(iter(trainer.get_train_dataloader()))
    ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    print("\nFirst ~80 token/label pairs:")
    for tid, lab in zip(ids[:80], labels[:80]):
        print(repr(tokenizer.decode([tid])), lab.item())

    trainer.train()

if __name__ == "__main__":
    main()