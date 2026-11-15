# RAG Eval Lab

![CI](https://github.com/makshitha0207/rag-eval-lab/actions/workflows/ci.yml/badge.svg)

**Which chunking strategy and retriever is best for your documents? Measure it instead of guessing.**

A small framework that compares 2 chunkers × 4 retrievers × 3 values of k on a RAG corpus and reports Hit@k, MRR, Precision@k and latency.

![accuracy vs latency](docs/accuracy_vs_latency.png)

## Results (k=5, 50 questions, 12 arXiv RAG papers)

| Chunker | Retriever | Hit@5 | MRR | p50 latency |
|---|---|---|---|---|
| recursive | dense | 0.700 | 0.575 | 30 ms |
| recursive | hybrid | 0.660 | 0.498 | 34 ms |
| recursive | bm25 | 0.560 | 0.468 | 3 ms |
| recursive | hybrid + rerank | 0.520 | 0.447 | 2226 ms |
| fixed | dense | 0.640 | 0.414 | 31 ms |
| fixed | bm25 | 0.540 | 0.470 | 3 ms |
| fixed | hybrid | 0.580 | 0.461 | 36 ms |
| fixed | hybrid + rerank | 0.560 | 0.436 | 2329 ms |

Best Hit@10 overall: recursive + hybrid (0.780). Full grid in `data/results.json`.

## Findings

1. **Recursive chunking beat fixed-size chunking for dense retrieval** (MRR 0.575 vs 0.414 at k=5).
2. **BM25 is ~10x faster than dense retrieval** (3 ms vs 30 ms) with competitive MRR.
3. **The cross-encoder reranker did not help on this corpus and was ~70x slower** (about 2.3 s per query on CPU).
4. **Hybrid (RRF) helped most at larger k** (Hit@10 0.780 with recursive chunks).

## Limitations

- Only 50 questions, so differences under ~0.05 are noise.
- Questions were LLM-generated from the corpus and are not yet fully hand-verified.
- Relevance is judged by token overlap with the source passage, which is approximate.
- One domain (RAG papers), one embedding model (bge-small), CPU timings.

## Run it

```bash
git clone https://github.com/makshitha0207/rag-eval-lab.git
cd rag-eval-lab/backend
pip install -r requirements.txt
# put PDFs in data/raw, add a key to backend/.env (see below)
python make_dataset.py        # generate eval questions
python run_retrieval_eval.py  # run the grid
python make_chart.py          # build the chart
```

`backend/.env` (any OpenAI-compatible API):

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
```

## Stack

Python, ChromaDB, sentence-transformers (bge-small), rank-bm25, cross-encoder reranker, tiktoken, pytest, GitHub Actions.

## Roadmap

LLM-as-judge for faithfulness, FastAPI service, Next.js dashboard, Qdrant backend.