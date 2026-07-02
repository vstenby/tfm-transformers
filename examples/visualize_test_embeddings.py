"""Visualize embeddings of a held-out test split, one panel per backend.

The clean workflow: fit on the train split (context), encode only the
held-out test rows, and project them with UMAP. Test rows are never part
of the context, so their embeddings carry no information about their own
labels — the separation you see is what the model genuinely infers.

Requires: pip install umap-learn matplotlib
"""

import time

import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tfm_embeddings import TabularEmbedder

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42
)

BACKENDS = [
    ("TabICLv2", TabularEmbedder("tabicl")),
    ("TabPFN", TabularEmbedder("tabpfn")),
    ("Google TabFM", TabularEmbedder("tabfm", n_estimators=4)),
]

panels = []
for name, model in BACKENDS:
    try:
        t0 = time.time()
        model.fit(X_train, y_train)
        emb = model.encode(X_test)
        print(f"{name}: {emb.shape} in {time.time() - t0:.0f}s", flush=True)
        panels.append((f"{name} ({emb.shape[1]}-d)", emb))
    except Exception as e:  # backend not installed / weights unavailable
        print(f"{name}: skipped ({type(e).__name__}: {e})", flush=True)

fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), squeeze=False)
for ax, (title, emb) in zip(axes[0], panels):
    proj = umap.UMAP(n_components=2, metric="cosine", random_state=42).fit_transform(emb)
    for label, color, cls in [(0, "#d62728", "malignant"), (1, "#1f77b4", "benign")]:
        mask = y_test == label
        ax.scatter(proj[mask, 0], proj[mask, 1], s=14, c=color, alpha=0.65, label=cls, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)

fig.suptitle(
    f"UMAP of held-out test embeddings — breast cancer "
    f"({len(X_train)} train rows as context, {len(X_test)} test rows shown)",
    y=0.98,
)
fig.tight_layout()
fig.savefig("docs/test_embeddings_umap.png", dpi=150, bbox_inches="tight")
print("saved: docs/test_embeddings_umap.png")
