"""Out-of-fold embedding tests, using the TabICL backend."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from tfm_embeddings import TabularEmbedder

pytest.importorskip("tabicl")


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    return X[:40], y[:40], X[40:]


def test_oof_shapes_and_final_model(data):
    X_train, y_train, X_test = data
    model = TabularEmbedder("tabicl", n_estimators=2)
    oof = model.fit_transform_oof(X_train, y_train, n_fold=2)

    dim = model.embedding_dim
    assert oof.shape == (40, dim)
    assert np.all(np.isfinite(oof))

    # The final full-data model serves encode() for unseen rows
    assert model.encode(X_test).shape == (10, dim)


def test_oof_differs_from_self_encoding(data):
    X_train, y_train, _ = data
    model = TabularEmbedder("tabicl", n_estimators=2)
    oof = model.fit_transform_oof(X_train, y_train, n_fold=2)

    # encode() on the training data uses the full-data model (twin in
    # context), which must give different vectors than the OOF ones.
    self_encoded = model.encode(X_train)
    assert not np.allclose(oof, self_encoded, atol=1e-4)


def test_oof_row_order(data):
    """OOF embeddings are aligned to the original row order.

    Row i's OOF embedding must equal what a fold model (fitted without
    fold containing i) produces for row i — verified by recomputing one
    fold manually with the same deterministic split.
    """
    from sklearn.model_selection import StratifiedKFold

    X_train, y_train, _ = data
    model = TabularEmbedder("tabicl", n_estimators=2)
    oof = model.fit_transform_oof(X_train, y_train, n_fold=2)

    train_idx, val_idx = next(iter(StratifiedKFold(n_splits=2).split(X_train, y_train)))
    manual = TabularEmbedder("tabicl", n_estimators=2)
    manual.fit(X_train[train_idx], y_train[train_idx])
    expected = manual.encode(X_train[val_idx])

    np.testing.assert_allclose(oof[val_idx], expected, rtol=1e-4, atol=1e-4)


def test_oof_with_groups(data):
    X_train, y_train, _ = data
    groups = np.arange(40) % 4  # 4 artificial entities
    model = TabularEmbedder("tabicl", n_estimators=2)
    oof = model.fit_transform_oof(X_train, y_train, n_fold=2, groups=groups)
    assert oof.shape == (40, model.embedding_dim)


def test_oof_regression_and_unlabeled():
    X, y = make_regression(n_samples=40, n_features=5, random_state=42)
    model = TabularEmbedder("tabicl", n_estimators=2)
    assert model.fit_transform_oof(X, y, n_fold=2).shape == (40, model.embedding_dim)

    model = TabularEmbedder("tabicl", n_estimators=2)
    assert model.fit_transform_oof(X, n_fold=2).shape == (40, model.embedding_dim)  # y=None


def test_oof_validation(data):
    X_train, y_train, _ = data
    with pytest.raises(ValueError, match="n_fold"):
        TabularEmbedder("tabicl").fit_transform_oof(X_train, y_train, n_fold=1)
