import sys

from app.config import settings
from app.ingestion.loaders import load_documents
from app.retrieval.retrievers import Retriever
from app.retrieval.store import ensure_index

question = " ".join(sys.argv[1:]) or input("Question: ")

docs = load_documents(settings.data_dir / "raw")
col, chunks = ensure_index("corpus", docs, "recursive")

for mode in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
    retriever = Retriever(col, chunks, mode)
    got, ms = retriever.retrieve(question, 3)
    print(f"\n=== {mode}  ({ms:.0f} ms) ===")
    for i, c in enumerate(got, 1):
        print(f"[{i}] {c.id}: {c.text[:220]!r}")