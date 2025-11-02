# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import torch

IGNORE_INDEX = -100

@dataclass
class SFTDataCollator:
    """
    Build input_ids/labels by concatenating:
      [prompt] + ' ' + [target] + eos
    Only supervise the target span (prompt tokens are masked to -100).
    """
    tokenizer: Any
    max_len: int = 512

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]

        inputs = [p + " " + t + self.tokenizer.eos_token for p, t in zip(prompts, targets)]
        enc = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        labels = enc["input_ids"].clone()

        # mask prompt tokens
        for i, p in enumerate(prompts):
            # Note: add_special_tokens=False to get raw tokenization
            p_ids = self.tokenizer(p + " ", add_special_tokens=False).input_ids
            cutoff = min(len(p_ids), labels.shape[1])
            labels[i, :cutoff] = IGNORE_INDEX

        enc["labels"] = labels
        return enc
