"""Gallery: held-out test-split embeddings across datasets and backends.

3x3 grid — rows are datasets (binary classification, 10-class
classification, regression), columns are backends. Each panel fits the
model on the train split, encodes only the held-out test rows, and
projects them with UMAP, so the visible structure is genuinely inferred.

Requires: pip install umap-learn matplotlib
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

from tfm_embeddings import TabularEmbedder

RANDOM_STATE = 42


def subsample(X, y, n):
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X), n, replace=False)
    return X[idx], y[idx]


def breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return X, y, "classification"


def digits():
    X, y = load_digits(return_X_y=True)
    X, y = subsample(X, y, 1000)
    return X, y, "classification"


def california_housing():
    X, y = fetch_california_housing(return_X_y=True)
    X, y = subsample(X, y, 1000)
    return X, y, "regression"


DATASETS = [
    ("Breast cancer\n2 classes", breast_cancer),
    ("Digits\n10 classes", digits),
    ("California housing\nregression", california_housing),
]

BACKENDS = [
    ("TabICLv2", lambda: TabularEmbedder("tabicl")),
    ("TabPFN", lambda: TabularEmbedder("tabpfn")),
    ("Google TabFM", lambda: TabularEmbedder("tabfm", n_estimators=4)),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), layout="constrained")

for i, (ds_name, loader) in enumerate(DATASETS):
    X, y, task = loader()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y if task == "classification" else None,
        random_state=RANDOM_STATE,
    )
    last_scatter = None
    for j, (backend_name, make_model) in enumerate(BACKENDS):
        ax = axes[i, j]
        model = make_model()
        t0 = time.time()
        model.fit(X_train, y_train)
        emb = model.encode(X_test)
        elapsed = time.time() - t0
        print(f"{ds_name.splitlines()[0]} x {backend_name}: {emb.shape} in {elapsed:.0f}s", flush=True)

        proj = umap.UMAP(n_components=2, metric="cosine", random_state=RANDOM_STATE).fit_transform(emb)
        cmap = "viridis" if task == "regression" else ("coolwarm" if len(np.unique(y)) == 2 else "tab10")
        last_scatter = ax.scatter(
            proj[:, 0], proj[:, 1], c=y_test, cmap=cmap, s=10, alpha=0.75, linewidths=0
        )
        ax.text(
            0.02, 0.02, f"{emb.shape[1]}-d · {elapsed:.0f}s",
            transform=ax.transAxes, fontsize=9, color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title(backend_name, fontsize=13)
        if j == 0:
            ax.set_ylabel(ds_name, fontsize=12)
    if task == "regression":
        fig.colorbar(last_scatter, ax=axes[i, :], shrink=0.8, pad=0.01, label="median house value")

fig.suptitle(
    "Held-out test rows, embedded and UMAP-projected — colored by true class / target\n"
    "(models fitted on the train split only; test rows are never part of the context)",
    fontsize=13,
)
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/embedding_gallery.png", dpi=150, bbox_inches="tight")
print("saved: docs/embedding_gallery.png")
