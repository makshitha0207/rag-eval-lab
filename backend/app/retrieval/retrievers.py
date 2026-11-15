import re
import time
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings
from app.retrieval.embeddings import embed_query


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


@dataclass
class Retrieved:
    id: str
    text: str


class Retriever:
    MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

    def __init__(self, collection, chunks, mode: str):
        assert mode in self.MODES, f"unknown mode {mode}"
        self.col = collection
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.mode = mode
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if mode != "dense" else None
        self.reranker = _reranker() if mode == "hybrid_rerank" else None

    def _dense(self, query: str, n: int) -> list[str]:
        n = min(n, self.col.count())
        return self.col.query(query_embeddings=[embed_query(query)], n_results=n)["ids"][0]

    def _bm25(self, query: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [self.chunks[i].id for i in top]

    def retrieve(self, query: str, k: int):
        t0 = time.perf_counter()
        if self.mode == "dense":
            ids = self._dense(query, k)
        elif self.mode == "bm25":
            ids = self._bm25(query, k)
        else:
            fused = rrf([self._dense(query, 20), self._bm25(query, 20)])[:20]
            if self.mode == "hybrid_rerank":
                scores = self.reranker.predict([(query, self.by_id[i].text) for i in fused])
                fused = [i for _, i in sorted(zip(scores, fused), key=lambda x: -x[0])]
            ids = fused[:k]
        ms = (time.perf_counter() - t0) * 1000
        return [Retrieved(i, self.by_id[i].text) for i in ids], ms