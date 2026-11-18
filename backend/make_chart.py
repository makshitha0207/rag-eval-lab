import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "data" / "results.json").read_text(encoding="utf-8"))
rows = [r for r in rows if r["k"] == 5]

colors = {"dense": "#2563eb", "bm25": "#16a34a", "hybrid": "#f59e0b", "hybrid_rerank": "#dc2626"}
names = {"dense": "dense", "bm25": "BM25", "hybrid": "hybrid", "hybrid_rerank": "hybrid + rerank"}

fig, ax = plt.subplots(figsize=(9, 5.5))
for r in rows:
    marker = "o" if r["chunker"] == "recursive" else "s"
    ax.scatter(r["p50_ms"], r["mrr"], s=130, c=colors[r["retriever"]],
               marker=marker, edgecolors="black", zorder=3)

# one label per retriever, placed at the average position of its two points
for ret, color in colors.items():
    pts = [r for r in rows if r["retriever"] == ret]
    x = sum(p["p50_ms"] for p in pts) / len(pts)
    y = sum(p["mrr"] for p in pts) / len(pts)
    ax.annotate(names[ret], (x, y), textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=10, fontweight="bold", color=color)

ax.set_xscale("log")
ax.set_xlabel("Median retrieval latency (ms, log scale)  -  faster is to the left")
ax.set_ylabel("MRR (higher is better)")
ax.set_title("RAG Eval Lab: accuracy vs latency (k=5, 50 questions)")
ax.grid(alpha=0.3)
ax.legend(handles=[
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
           markersize=10, label="recursive chunks"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black",
           markersize=10, label="fixed chunks"),
], loc="lower right")
fig.tight_layout()
out = ROOT / "docs" / "accuracy_vs_latency.png"
fig.savefig(out, dpi=160)
print("Saved", out)