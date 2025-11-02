# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Sample:
    """
    A single SFT training example.
    - task: "ARC" | "TRIVIA" | "IFEVAL"
    - prompt: input text (already templated)
    - target: expected output text (short)
    """
    task: str
    prompt: str
    target: str

    def to_dict(self) -> dict:
        return asdict(self)
