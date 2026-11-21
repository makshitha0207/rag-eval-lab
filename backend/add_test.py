import json
import unicodedata

from app.config import settings
from app.ingestion.loaders import load_documents


def norm_with_map(text):
    """Lowercase letters/digits only, with a map back to original positions.
    This ignores PDF quirks like ligatures and '/f_i' markers."""
    out, idx = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if c.isalnum():
                out.append(c.lower())
                idx.append(i)
    return "".join(out), idx


def find_evidence(docs, phrase):
    pn, _ = norm_with_map(phrase)
    if len(pn) < 12:
        return None, None
    for d in docs:
        tn, idx = norm_with_map(d.text)
        pos = tn.find(pn)
        if pos != -1:
            s, e = idx[pos], idx[pos + len(pn) - 1]
            return d.id, d.text[max(0, s - 200): e + 200].strip()
    return None, None


docs = load_documents(settings.data_dir / "raw")
path = settings.data_dir / "eval" / "custom.jsonl"
path.parent.mkdir(parents=True, exist_ok=True)

print("Add tests. Leave the question empty to stop.\n")
while True:
    q = input("Question: ").strip()
    if not q:
        break
    a = input("Short correct answer: ").strip()
    phrase = input("A phrase (6-12 words) copied from the paper that contains the answer: ").strip()
    doc_id, evidence = find_evidence(docs, phrase)
    if not evidence:
        print("  Could not find that phrase in any PDF. Try different words. Not saved.\n")
        continue
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"question": q, "reference_answer": a,
                            "evidence": evidence, "doc_id": doc_id}, ensure_ascii=False) + "\n")
    print(f"  Saved (found in {doc_id}).\n")
print("Tests are in", path)