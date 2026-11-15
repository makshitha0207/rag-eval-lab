from dataclasses import dataclass, field

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


def ntok(text: str) -> int:
    return len(enc.encode(text))


def fixed_chunker(doc, size: int = 512, overlap: int = 50) -> list[Chunk]:
    toks = enc.encode(doc.text)
    step = size - overlap
    out, i, n = [], 0, 0
    while i < len(toks):
        piece = enc.decode(toks[i : i + size])
        out.append(Chunk(f"{doc.id}-fixed-{n}", doc.id, piece, {"strategy": "fixed"}))
        n += 1
        if i + size >= len(toks):
            break
        i += step
    return out


SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if ntok(text) <= size:
        return [text]
    if not seps:
        toks = enc.encode(text)
        return [enc.decode(toks[i : i + size]) for i in range(0, len(toks), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out = []
    for j, part in enumerate(parts):
        if not part.strip():
            continue
        piece = part + (sep if j < len(parts) - 1 else "")
        out.extend(_split(piece, size, rest))
    return out


def recursive_chunker(doc, size: int = 512, overlap: int = 50) -> list[Chunk]:
    pieces = _split(doc.text, size, SEPARATORS)
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_tok = 0
    fresh = False

    def flush():
        text = "".join(cur).strip()
        if text:
            n = len(chunks)
            chunks.append(Chunk(f"{doc.id}-recursive-{n}", doc.id, text, {"strategy": "recursive"}))

    for p in pieces:
        t = ntok(p)
        if cur and cur_tok + t > size:
            flush()
            carry, ct = [], 0
            for q in reversed(cur):
                qt = ntok(q)
                if ct + qt > overlap:
                    break
                carry.insert(0, q)
                ct += qt
            if ct + t > size:
                carry, ct = [], 0
            cur, cur_tok, fresh = carry, ct, False
        cur.append(p)
        cur_tok += t
        fresh = True
    if cur and fresh:
        flush()
    return chunks


CHUNKERS = {"fixed": fixed_chunker, "recursive": recursive_chunker}