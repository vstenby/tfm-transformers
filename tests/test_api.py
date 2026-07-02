"""Backend-independent tests: registry, aggregation, similarity math."""

import numpy as np
import pytest

from tfm_embeddings import TabularEmbedder, available_backends, cosine_similarity


def test_available_backends():
    assert available_backends() == ["tabfm", "tabicl", "tabpfn"]


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend 'llm'"):
        TabularEmbedder("llm")


def test_invalid_aggregate():
    with pytest.raises(ValueError, match="aggregate"):
        TabularEmbedder("tabicl", aggregate="max")


def test_encode_requires_fit():
    pytest.importorskip("tabicl")
    model = TabularEmbedder("tabicl")
    with pytest.raises(RuntimeError, match="not fitted"):
        model.encode(np.zeros((3, 2)))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.search(np.zeros((3, 2)))


def test_cosine_similarity():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    sims = cosine_similarity(a, b)

    assert sims.shape == (2, 3)
    np.testing.assert_allclose(sims[0, 0], 1.0)
    np.testing.assert_allclose(sims[0, 1], 1 / np.sqrt(2))
    np.testing.assert_allclose(sims[:, 2], 0.0)  # zero vector -> zero similarity


def test_aggregate_shapes():
    members = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)

    model = TabularEmbedder.__new__(TabularEmbedder)
    model.aggregate = "mean"
    assert model._aggregate(members).shape == (4, 3)

    model.aggregate = "concat"
    concat = model._aggregate(members)
    assert concat.shape == (4, 6)
    np.testing.assert_array_equal(concat[0], np.concatenate([members[0, 0], members[1, 0]]))

    model.aggregate = "none"
    assert model._aggregate(members).shape == (2, 4, 3)
