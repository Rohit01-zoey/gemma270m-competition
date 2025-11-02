# -*- coding: utf-8 -*-
"""
Prompt templates used in SFT.
These MUST stay identical to the evaluation templates to avoid train/eval drift.
"""
from __future__ import annotations

ARC: str = (
    "Choose ONLY one option: A, B, C, or D.\n\n"
    "Question: {q}\n\n"
    "Options:\n"
    "A) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
    "Answer (one letter A-D):"
)

TRIVIA: str = (
    "You are a concise QA assistant.\n"
    "Question: {q}\n"
    "Answer with a single short phrase.\n"
    "Answer:"
)

IFEVAL: str = "Instruction: {instr}\nResponse:"
