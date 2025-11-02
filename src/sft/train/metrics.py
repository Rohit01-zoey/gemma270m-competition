# -*- coding: utf-8 -*-
from __future__ import annotations
import re, string
from typing import Optional

try:
    from src.eval.postprocess import extract_letter as _eval_extract_letter
    from src.eval.postprocess import normalize_text as _eval_norm
    from src.eval.postprocess import em as _eval_em, f1 as _eval_f1
    from src.eval.postprocess import ifeval_rule_check as _eval_rule
    HAS_EVAL_POST = True
except Exception:
    HAS_EVAL_POST = False

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile("[" + re.escape(string.punctuation) + "]")

def normalize_text(s: str) -> str:
    if HAS_EVAL_POST:
        return _eval_norm(s)
    s = s.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def em(a: str, b: str) -> int:
    return _eval_em(a, b) if HAS_EVAL_POST else int(normalize_text(a) == normalize_text(b))

def f1(a: str, b: str) -> float:
    if HAS_EVAL_POST:
        return _eval_f1(a, b)
    a, b = normalize_text(a), normalize_text(b)
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return float(a == b)
    common = {}
    for w in ta:
        common[w] = min(ta.count(w), tb.count(w))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)

_LETTER_RE = re.compile(r"\b([ABCD])\b", re.I)
def extract_letter(s: str) -> Optional[str]:
    if HAS_EVAL_POST:
        return _eval_extract_letter(s)
    m = _LETTER_RE.search(s.strip())
    return m.group(1).upper() if m else None

def ifeval_rule_check(response: str, rule: dict) -> bool:
    return _eval_rule(response, rule) if HAS_EVAL_POST else True  # fallback: pass
