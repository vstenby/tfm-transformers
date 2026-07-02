"""2D UMAP of TabICLv2 embeddings on the breast cancer dataset.

Requires: pip install umap-learn matplotlib
"""

import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_breast_cancer

from tfm_embeddings import TabularEmbedder

data = load_breast_cancer()
X, y = data.data, data.target

model = TabularEmbedder("tabicl").fit(X, y)
embeddings = model.encode(X)  # rows embedded as test rows: no label leakage
print(f"embeddings: {embeddings.shape}")

proj = umap.UMAP(n_components=2, metric="cosine", random_state=42).fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(7, 6))
for label, color, name in [(0, "#d62728", "malignant"), (1, "#1f77b4", "benign")]:
    mask = y == label
    ax.scatter(proj[mask, 0], proj[mask, 1], s=12, c=color, alpha=0.6, label=name, linewidths=0)
ax.set_title("UMAP of TabICLv2 row embeddings (breast cancer)")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("umap_breast_cancer.png", dpi=150)
print("saved: umap_breast_cancer.png")
