"""TabFM backend tests. Skipped when tabfm (with PyTorch) is not installed."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from tfm_embeddings import TabularEmbedder

tabfm = pytest.importorskip("tabfm")
if not hasattr(tabfm, "tabfm_v1_0_0_pytorch"):
    pytest.skip("TabFM PyTorch backend not available", allow_module_level=True)


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    return X[:40], y[:40], X[40:]


@pytest.fixture(scope="module")
def fitted_model(data):
    X_corpus, y_corpus, _ = data
    return TabularEmbedder("tabfm", n_estimators=2).fit(X_corpus, y_corpus)


def test_encode_shapes(fitted_model, data):
    X_corpus, _, X_query = data
    corpus_emb = fitted_model.encode(X_corpus)
    query_emb = fitted_model.encode(X_query)

    dim = fitted_model.embedding_dim
    assert dim is not None
    assert corpus_emb.shape == (40, dim)
    assert query_emb.shape == (10, dim)
    assert np.all(np.isfinite(corpus_emb))
    assert np.all(np.isfinite(query_emb))


def test_search(fitted_model, data):
    X_corpus, _, X_query = data
    indices, scores = fitted_model.search(X_query, top_k=3)
    assert indices.shape == (10, 3)
    assert indices.max() < 40
    assert np.all(scores <= 1.0 + 1e-6)

    # Querying with corpus rows should retrieve those very rows first
    indices, scores = fitted_model.search(X_corpus[:5], top_k=1)
    np.testing.assert_array_equal(indices[:, 0], np.arange(5))
    np.testing.assert_allclose(scores[:, 0], 1.0, atol=1e-3)


def test_unlabeled_context(data):
    X_corpus, _, X_query = data
    model = TabularEmbedder("tabfm", n_estimators=2).fit(X_corpus)  # y=None
    emb = model.encode(X_query)
    assert emb.shape == (10, model.embedding_dim)
    assert np.all(np.isfinite(emb))


def test_regression_context():
    X, y = make_regression(n_samples=50, n_features=5, random_state=42)
    model = TabularEmbedder("tabfm", n_estimators=2).fit(X[:40], y[:40])
    emb = model.encode(X[40:])
    assert emb.shape == (10, model.embedding_dim)
