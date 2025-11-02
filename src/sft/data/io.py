# -*- coding: utf-8 -*-
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple, Dict
from .schema import Sample

def write_jsonl(path: str | Path, items: Iterable[Sample]) -> None:
    """Write a list/iterable of Sample to JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")

def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream JSONL dicts from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def stratified_split(
    items: Sequence[Sample],
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """
    Split by 'task' field to preserve task proportions.
    """
    by_task: Dict[str, List[Sample]] = {}
    for it in items:
        by_task.setdefault(it.task, []).append(it)

    rng = random.Random(seed)
    train, dev = [], []
    for t, rows in by_task.items():
        rows = rows[:]  # copy
        rng.shuffle(rows)
        k = max(1, int(len(rows) * (1.0 - dev_ratio)))
        train.extend(rows[:k])
        dev.extend(rows[k:])
    rng.shuffle(train); rng.shuffle(dev)
    return train, dev
