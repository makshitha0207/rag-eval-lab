from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    id: str
    source: str
    text: str


def load_documents(folder) -> list[Document]:
    docs = []
    for p in sorted(Path(folder).rglob("*")):
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            text = "\n\n".join((page.extract_text() or "") for page in PdfReader(p).pages)
        elif suffix in {".md", ".txt"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        if text.strip():
            docs.append(Document(id=p.stem, source=str(p), text=text))
    return docs