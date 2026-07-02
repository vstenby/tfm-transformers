"""TabPFN backend tests.

Skipped when tabpfn is not installed or when no TabPFN license token is
available (downloading TabPFN weights requires authentication, see
https://ux.priorlabs.ai).
"""

import os

import numpy as np
import pytest
from sklearn.datasets import make_classification

from tfm_embeddings import TabularEmbedder

pytest.importorskip("tabpfn")

if not os.environ.get("TABPFN_TOKEN"):
    pytest.skip(
        "TabPFN weight download requires license authentication (set TABPFN_TOKEN)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    return X[:40], y[:40], X[40:]


def test_encode_and_search(data):
    X_corpus, y_corpus, X_query = data
    model = TabularEmbedder("tabpfn", n_estimators=2).fit(X_corpus, y_corpus)

    corpus_emb = model.encode(X_corpus)
    query_emb = model.encode(X_query)
    assert corpus_emb.ndim == 2
    assert corpus_emb.shape[0] == 40
    assert query_emb.shape == (10, corpus_emb.shape[1])
    assert np.all(np.isfinite(corpus_emb))

    indices, scores = model.search(X_query, top_k=3)
    assert indices.shape == (10, 3)
    assert np.all(scores <= 1.0 + 1e-6)


def test_unlabeled_context(data):
    X_corpus, _, X_query = data
    model = TabularEmbedder("tabpfn", n_estimators=2).fit(X_corpus)  # y=None
    emb = model.encode(X_query)
    assert emb.ndim == 2
    assert emb.shape[0] == 10
    assert np.all(np.isfinite(emb))
