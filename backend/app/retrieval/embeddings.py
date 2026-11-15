from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    return _model().encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    ).tolist()


def embed_query(query: str) -> list[float]:
    return _model().encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()