"""
Router inference for classification-based router.

Usage:
    from router_classification.router import Router

    router = Router("checkpoints/router_cls/final")
    task = router.classify("What is 2+2?")
    print(task)  # "reasoning" or "factual_qa" or "instruction_following"
"""
import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

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

class Router:
    """Classification-based router for task type prediction."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)

        # Load label mapping
        with open(self.model_path / "label_map.json") as f:
            label_map = json.load(f)
            self.id_to_task = {int(k): v for k, v in label_map["id_to_task"].items()}
            # Get base model if saved, otherwise use default
            base_model = label_map.get("base_model", "google/gemma-3-270m")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with classification head
        self.model = RouterClassificationModel(base_model, num_labels=3)

        # Load trained weights
        state_dict = torch.load(self.model_path / "model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move to device and ensure bfloat16 dtype
        self.model.to(self.device)
        self.model = self.model.to(torch.bfloat16)

    @torch.inference_mode()
    def classify(self, question):
        """Classify a single question."""
        # Tokenize (no prompt prefix, just the question)
        inputs = self.tokenizer(
            question.strip(),
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        # Get predictions
        outputs = self.model(**inputs)
        logits = outputs["logits"]

        # Get predicted class
        pred_id = logits.argmax(dim=-1).item()
        return self.id_to_task[pred_id]

    @torch.inference_mode()
    def classify_batch(self, questions, batch_size=16):
        """Classify a batch of questions."""
        results = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                [q.strip() for q in batch],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            # Get predictions
            outputs = self.model(**inputs)
            logits = outputs["logits"]

            # Get predicted classes
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            results.extend([self.id_to_task[pred_id] for pred_id in pred_ids])

        return results
