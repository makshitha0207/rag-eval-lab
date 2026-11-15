from app.ingestion.chunking import fixed_chunker, ntok, recursive_chunker
from app.ingestion.loaders import Document

DOC = Document("d", "d", ("Hello world. This is a sentence.\n\n" * 200))


def test_fixed_chunks_not_too_big():
    for c in fixed_chunker(DOC, size=100, overlap=10):
        assert ntok(c.text) <= 100


def test_recursive_chunks_not_too_big():
    chunks = recursive_chunker(DOC, size=100, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert ntok(c.text) <= 100