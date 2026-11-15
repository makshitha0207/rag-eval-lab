import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "data" / "results.json").read_text(encoding="utf-8"))
rows = [r for r in rows if r["k"] == 5]

colors = {"dense": "#2563eb", "bm25": "#16a34a", "hybrid": "#f59e0b", "hybrid_rerank": "#dc2626"}
fig, ax = plt.subplots(figsize=(9, 5.5))
for r in rows:
    marker = "o" if r["chunker"] == "recursive" else "s"
    ax.scatter(r["p50_ms"], r["mrr"], s=130, c=colors[r["retriever"]], marker=marker, edgecolors="black")
    ax.annotate(f'{r["chunker"][:3]}+{r["retriever"]}', (r["p50_ms"], r["mrr"]),
                textcoords="offset points", xytext=(8, 6), fontsize=8)

ax.set_xscale("log")
ax.set_xlabel("Median retrieval latency (ms, log scale)  <- faster")
ax.set_ylabel("MRR (higher is better)")
ax.set_title("RAG Eval Lab: accuracy vs latency (k=5, 50 questions)")
ax.grid(alpha=0.3)
fig.tight_layout()
out = ROOT / "docs" / "accuracy_vs_latency.png"
fig.savefig(out, dpi=160)
print("Saved", out)