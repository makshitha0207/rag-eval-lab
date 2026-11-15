
import json
import random
import re
import time
from collections import Counter

from app.config import settings
from app.generation.llm import LLM
from app.ingestion.loaders import load_documents

random.seed(7)

SYSTEM = (
    "You create evaluation questions for a RAG benchmark. "
    'Return ONLY JSON: {"question": str, "reference_answer": str}. '
    "Ask about a concrete fact, method, or result found in the text. "
    "The question must be self-contained and answerable using only the text. "
    "Do not copy sentences directly. "
    "If the text is a reference list, table of contents, or lacks useful "
    'information, return {"question": "", "reference_answer": ""}.'
)


def good_passage(p: str) -> bool:
    if not (400 <= len(p) <= 1200):
        return False

    low = p.lower()

    if low.count("arxiv") + low.count("proceedings") + low.count("et al") > 2:
        return False

    if len(re.findall(r"\[\d+\]", p)) > 3:
        return False

    if len(re.findall(r"\.\s\.\s\.", p)) > 1:
        return False

    letters = sum(c.isalpha() for c in p)

    return letters / len(p) > 0.60


def make_chunks(text: str, chunk_words: int = 150, overlap: int = 35):
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    step = chunk_words - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words]).strip()

        if len(chunk) >= 400:
            yield chunk


def main(n: int = 50, per_doc: int = 5):
    docs = load_documents(settings.data_dir / "raw")
    path = settings.data_dir / "eval" / "eval.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    pool = []

    for d in docs:
        chunks = 0

        for p in make_chunks(d.text):
            if good_passage(p):
                pool.append((d.id, p))
                chunks += 1

        print(f"{d.id}: {chunks} candidate chunks")

    random.shuffle(pool)

    print(f"\nTotal candidate passages: {len(pool)}")

    llm = LLM()
    count = 0
    attempts = 0
    used = set()
    per = Counter()

    with open(path, "w", encoding="utf-8") as f:
        for doc_id, passage in pool:

            if count >= n or attempts >= n * 4:
                break

            if passage in used or per[doc_id] >= per_doc:
                continue

            attempts += 1

            try:
                response, _ = llm.complete(
                    SYSTEM,
                    f"Text:\n{passage}"
                )

                time.sleep(3)

                response = response.strip()
                response = response.replace("```json", "").replace("```", "")

                start = response.find("{")
                end = response.rfind("}")

                if start == -1 or end == -1:
                    continue

                obj = json.loads(response[start:end + 1])

                q = obj.get("question", "").strip()
                a = obj.get("reference_answer", "").strip()

                if not q or not a or a.lower() == q.lower():
                    continue

                used.add(passage)
                per[doc_id] += 1

                row = {
                    "question": q,
                    "reference_answer": a,
                    "evidence": passage,
                    "doc_id": doc_id
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                count += 1
                print(f"{count}/{n} questions generated")

            except Exception as e:
                print("Skipped:", str(e)[:120])

                if "429" in str(e):
                    print("Rate limit reached. Stopping generation.")
                    break

    print(f"\nWrote {count} questions to {path}")


if __name__ == "__main__":
    main()
