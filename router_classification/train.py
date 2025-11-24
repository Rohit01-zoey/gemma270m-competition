#!/usr/bin/env python3
"""
Train router with classification head.

Usage: python router_classification/train.py --epochs 5
"""
import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments, Trainer, TrainerCallback
)
from datasets import Dataset

# Task label mapping
TASK_TO_ID = {
    "reasoning": 0,
    "factual_qa": 1,
    "instruction_following": 2
}
ID_TO_TASK = {v: k for k, v in TASK_TO_ID.items()}

def load_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

class MetricsCallback(TrainerCallback):
    """Callback to track training and validation metrics history."""
    def __init__(self):
        self.history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "steps": []
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens."""
        if logs:
            step = state.global_step

            # Track training loss
            if "loss" in logs:
                self.history["train_loss"].append({
                    "step": step,
                    "epoch": logs.get("epoch", 0),
                    "loss": logs["loss"]
                })

            # Track eval metrics
            if "eval_loss" in logs:
                self.history["eval_loss"].append({
                    "step": step,
                    "epoch": logs.get("epoch", 0),
                    "loss": logs["eval_loss"]
                })

            if "eval_accuracy" in logs:
                self.history["eval_accuracy"].append({
                    "step": step,
                    "epoch": logs.get("epoch", 0),
                    "accuracy": logs["eval_accuracy"]
                })

    def save_history(self, path):
        """Save metrics history to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

class RouterClassificationModel(nn.Module):
    """Model with classification head."""
    def __init__(self, base_model_name, num_labels=3, hidden_size=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16
        )

        if hidden_size is None:
            hidden_size = self.encoder.config.hidden_size

        # Classification head with 2 layers + activation + dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use last hidden state, take mean over sequence
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

def tokenize(examples, tokenizer):
    # Tokenize questions only (no prompt prefix)
    tokens = tokenizer(examples['prompt'], truncation=True, max_length=512, padding="max_length")

    # Convert target labels to IDs
    tokens['labels'] = [TASK_TO_ID[t] for t in examples['target']]

    return tokens

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/gemma-3-270m")
    parser.add_argument("--out_dir", default="checkpoints/router_cls")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    # Load data
    train_data = Dataset.from_list(load_data("router_classification/data/train.jsonl"))
    dev_data = Dataset.from_list(load_data("router_classification/data/dev.jsonl"))
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with classification head
    print("Loading model with classification head...")
    model = RouterClassificationModel(args.base_model, num_labels=3)

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
        metric_for_best_model="accuracy",
        bf16=True,
        save_total_limit=2
    )

    # Create metrics callback to track training history
    metrics_callback = MetricsCallback()

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )

    print(f"\nTraining for {args.epochs} epochs...")
    trainer.train()

    # Save
    final_path = Path(args.out_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), final_path / "model.pt")
    tokenizer.save_pretrained(final_path)

    # Save label mapping and base model name
    with open(final_path / "label_map.json", 'w') as f:
        json.dump({
            "task_to_id": TASK_TO_ID,
            "id_to_task": ID_TO_TASK,
            "base_model": args.base_model
        }, f)

    # Save training metrics history
    metrics_path = final_path / "training_history.json"
    metrics_callback.save_history(metrics_path)
    print(f"\nTraining history saved to {metrics_path}")

    # Print summary of training
    if metrics_callback.history["train_loss"]:
        final_train_loss = metrics_callback.history["train_loss"][-1]["loss"]
        print(f"Final training loss: {final_train_loss:.4f}")

    if metrics_callback.history["eval_accuracy"]:
        final_eval_acc = metrics_callback.history["eval_accuracy"][-1]["accuracy"]
        print(f"Final validation accuracy: {final_eval_acc:.4f}")

    print(f"\nModel saved to {final_path}")

if __name__ == "__main__":
    main()
