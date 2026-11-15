import numpy as np

from app.retrieval.retrievers import tokenize


def is_relevant(chunk_text: str, evidence: str, thresh: float = 0.6) -> bool:
    ev, ch = set(tokenize(evidence)), set(tokenize(chunk_text))
    return len(ev & ch) / max(len(ev), 1) >= thresh


def hit_at_k(rel: list[bool], k: int) -> float:
    return float(any(rel[:k]))


def reciprocal_rank(rel: list[bool]) -> float:
    for i, r in enumerate(rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(rel: list[bool], k: int) -> float:
    return sum(rel[:k]) / k


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))