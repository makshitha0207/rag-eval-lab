import json
import random
import time

from app.config import settings
from app.generation.llm import LLM
from app.ingestion.loaders import load_documents

random.seed(42)

SYSTEM = (
    "You write evaluation questions for a RAG benchmark. Return ONLY JSON: "
    '{"question": str, "reference_answer": str}. The question must be answerable '
    "solely from the passage, be self-contained, and must NOT copy phrases verbatim "
    "from the passage."
)


def main(n: int = 50):
    docs = load_documents(settings.data_dir / "raw")
    print(f"Loaded {len(docs)} documents")
    if not docs:
        print("No documents found in data/raw. Download the PDFs first.")
        return

    path = settings.data_dir / "eval" / "eval.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    llm, count, attempts = LLM(), 0, 0
    with open(path, "w", encoding="utf-8") as f:
        while count < n and attempts < n * 4:
            attempts += 1
            d = random.choice(docs)
            paras = [p.strip() for p in d.text.split("\n\n") if 300 <= len(p.strip()) <= 1200]
            if not paras:
                continue
            passage = random.choice(paras)
            try:
                text, _ = llm.complete(SYSTEM, f"Passage:\n{passage}")
            except Exception as e:
                print(f"  API error: {str(e)[:150]} (waiting 30s)")
                time.sleep(30)
                continue
            time.sleep(5)
            try:
                obj = json.loads(text[text.index("{") : text.rindex("}") + 1])
                row = {
                    "question": obj["question"],
                    "reference_answer": obj["reference_answer"],
                    "evidence": passage,
                    "doc_id": d.id,
                }
            except Exception:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            count += 1
            print(f"  {count}/{n} questions done")

    print(f"Wrote {count} questions to {path}")


if __name__ == "__main__":
    main()