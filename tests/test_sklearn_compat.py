"""scikit-learn estimator API compatibility: pipelines, clone, get/set_params."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from tfm_embeddings import TabularEmbedder

pytest.importorskip("tabicl")


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=60, n_features=5, random_state=42)
    return X[:40], y[:40], X[40:], y[40:]


def test_get_params_includes_backend_kwargs():
    model = TabularEmbedder("tabicl", n_estimators=2, random_state=0)
    params = model.get_params()
    assert params["model"] == "tabicl"
    assert params["aggregate"] == "mean"
    assert params["n_estimators"] == 2
    assert params["random_state"] == 0


def test_set_params_resets_fitted_state(data):
    X_train, y_train, X_test, _ = data
    model = TabularEmbedder("tabicl", n_estimators=2).fit(X_train, y_train)
    model.set_params(n_estimators=4)
    assert model.get_params()["n_estimators"] == 4
    with pytest.raises(RuntimeError, match="not fitted"):
        model.encode(X_test)


def test_clone_preserves_backend_kwargs():
    model = TabularEmbedder("tabicl", aggregate="concat", n_estimators=2)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_transform_matches_encode(data):
    X_train, y_train, X_test, _ = data
    model = TabularEmbedder("tabicl", n_estimators=2).fit(X_train, y_train)
    np.testing.assert_array_equal(model.transform(X_test), model.encode(X_test))


def test_fit_transform(data):
    X_train, y_train, _, _ = data
    model = TabularEmbedder("tabicl", n_estimators=2)
    emb = model.fit_transform(X_train, y_train)
    assert emb.shape == (len(X_train), model.embedding_dim)


def test_pipeline(data):
    X_train, y_train, X_test, y_test = data
    pipe = make_pipeline(
        TabularEmbedder("tabicl", n_estimators=2),
        LogisticRegression(max_iter=1000),
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert pipe.score(X_test, y_test) > 0.5
