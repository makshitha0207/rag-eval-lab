import chromadb

from app.config import settings
from app.ingestion.chunking import CHUNKERS
from app.retrieval.embeddings import embed

_client = chromadb.PersistentClient(path=str(settings.chroma_dir))


def ensure_index(corpus: str, docs, chunker: str, size: int = 512, overlap: int = 50):
    """Chunk docs and index them. Skips embedding if the collection already exists."""
    chunks = []
    for d in docs:
        chunks.extend(CHUNKERS[chunker](d, size, overlap))

    name = f"{corpus}_{chunker}_{size}_{overlap}"
    col = _client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        print(f"Indexing {len(chunks)} chunks into '{name}' ...")
        for i in range(0, len(chunks), 256):
            batch = chunks[i : i + 256]
            col.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embed([c.text for c in batch]),
                metadatas=[{"doc_id": c.doc_id} for c in batch],
            )
    return col, chunks