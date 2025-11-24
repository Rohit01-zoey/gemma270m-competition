#!/usr/bin/env python3
"""
Train router model.

Usage: python router_training/train.py [--use_lora]
"""
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset

def load_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def tokenize(examples, tokenizer):
    texts = [f"{p} {t}" for p, t in zip(examples['prompt'], examples['target'])]
    tokens = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    # Copy input_ids to labels, replacing pad tokens with -100
    labels = []
    for input_ids in tokens['input_ids']:
        label = [id if id != tokenizer.pad_token_id else -100 for id in input_ids]
        labels.append(label)
    tokens['labels'] = labels
    return tokens

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/gemma-3-270m-it")
    parser.add_argument("--out_dir", default="checkpoints/router")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    # Load data
    train_data = Dataset.from_list(load_data("router_training/data/train.jsonl"))
    dev_data = Dataset.from_list(load_data("router_training/data/dev.jsonl"))
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # LoRA if requested
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        # Enable gradient checkpointing and prepare for training
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    model.config.use_cache = False

    # Tokenize
    train_data = train_data.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=train_data.column_names
    )
    dev_data = dev_data.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=dev_data.column_names
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        bf16=True,
        save_total_limit=2
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer
    )

    trainer.train()

    # Save
    final_path = Path(args.out_dir) / "final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved to {final_path}")

if __name__ == "__main__":
    main()
