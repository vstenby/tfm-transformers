from __future__ import annotations

import numpy as np

from .adapters import resolve_backend


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two sets of embeddings.

    Parameters
    ----------
    a : np.ndarray of shape (n_a, dim)
    b : np.ndarray of shape (n_b, dim)

    Returns
    -------
    np.ndarray of shape (n_a, n_b)
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


class TabularEmbedder:
    """Sentence-transformers-style embeddings for tabular foundation models.

    Tabular foundation models (TabICL, TabPFN, ...) compute rich per-row
    representations internally. This class exposes them behind a familiar
    encode/similarity API::

        from tfm_embeddings import TabularEmbedder

        model = TabularEmbedder("tabicl")

        model.fit(X_corpus, y_corpus)          # 1. set the context table
        embeddings = model.encode(X_corpus)    # 2. embed rows -> (n, dim)
        query_emb = model.encode(X_query)      # same space, comparable

        similarities = model.similarity(query_emb, embeddings)
        indices, scores = model.search(X_query, top_k=5)

    Unlike sentence embeddings, tabular embeddings are **context-dependent**:
    a row's vector is conditioned on the entire context table (column
    distributions and, for target-aware models, the labels). ``fit`` sets
    that context; every ``encode`` call embeds rows against it as unseen test
    rows. Embeddings are therefore only comparable when produced by the same
    fitted model — there is no shared space across tables or across models.

    Parameters
    ----------
    model : str, default="tabicl"
        Backend to use, optionally with a variant after a slash:

        - ``"tabicl"`` or ``"tabicl/<checkpoint_version>"``
        - ``"tabpfn"`` or ``"tabpfn/<model_path>"``

    aggregate : {"mean", "concat", "none"}, default="mean"
        How to combine embeddings across the backend's ensemble members
        (each member embeds a differently shuffled view of the features):

        - ``"mean"``: Average across members -> ``(n_rows, dim)``.
        - ``"concat"``: Concatenate members -> ``(n_rows, n_members * dim)``.
        - ``"none"``: Raw members -> ``(n_members, n_rows, dim)``.

    **backend_kwargs
        Forwarded to the backend adapter (e.g. ``n_estimators``, ``device``,
        ``random_state``, or any underlying estimator parameter).

    Attributes
    ----------
    adapter : BackendAdapter
        The resolved backend adapter.

    corpus_embeddings_ : np.ndarray
        Aggregated embeddings of the context rows, computed lazily on the
        first ``search`` call.
    """

    def __init__(self, model: str = "tabicl", *, aggregate: str = "mean", **backend_kwargs):
        if aggregate not in ("mean", "concat", "none"):
            raise ValueError(f"aggregate must be 'mean', 'concat', or 'none', got '{aggregate}'.")

        backend_name, _, variant = model.partition("/")
        adapter_cls = resolve_backend(backend_name)

        self.model = model
        self.aggregate = aggregate
        self.adapter = adapter_cls(variant or None, **backend_kwargs)

    def _aggregate(self, embeddings: np.ndarray) -> np.ndarray:
        if self.aggregate == "mean":
            return embeddings.mean(axis=0)
        if self.aggregate == "concat":
            n_members, n_rows, dim = embeddings.shape
            return embeddings.transpose(1, 0, 2).reshape(n_rows, n_members * dim)
        return embeddings  # "none"

    def fit(self, X, y=None) -> "TabularEmbedder":
        """Set the context table that all embeddings are conditioned on.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Context rows (the "corpus" in retrieval terms).

        y : array-like of shape (n_samples,) or None
            Context targets. When provided, the embedding space is shaped by
            the prediction task (label-aware similarity). When None, a
            pseudo-target is synthesized for approximately unsupervised
            embeddings (experimental).

        Returns
        -------
        self : TabularEmbedder
        """
        self.adapter.fit(X, y)
        self._X_context = X
        self.corpus_embeddings_ = None
        return self

    def encode(self, X) -> np.ndarray:
        """Embed rows against the fitted context.

        Rows are embedded as unseen test rows: their targets (if any) are
        never visible to the model, so embedding context rows through
        ``encode`` does not leak their labels.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Rows to embed. Must have the same columns as the context table.

        Returns
        -------
        np.ndarray
            Embeddings aggregated according to ``aggregate``
            (``(n_rows, dim)`` for the default ``"mean"``).
        """
        if not hasattr(self, "_X_context"):
            raise RuntimeError("TabularEmbedder is not fitted. Call fit(X, y) first.")
        return self._aggregate(self.adapter.encode(X))

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity between two sets of embeddings.

        Both inputs must come from this fitted model — embeddings from
        different contexts or backends are not comparable.

        Returns
        -------
        np.ndarray of shape (n_a, n_b)
        """
        return cosine_similarity(a, b)

    def search(self, X_query, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Find the context rows most similar to each query row.

        Context-row embeddings are computed on the first call and cached on
        ``corpus_embeddings_``.

        Parameters
        ----------
        X_query : array-like of shape (n_queries, n_features)
            Rows to search with.

        top_k : int, default=5
            Number of nearest context rows to return per query.

        Returns
        -------
        indices : np.ndarray of shape (n_queries, top_k)
            Row indices into the context table, most similar first.

        scores : np.ndarray of shape (n_queries, top_k)
            Corresponding cosine similarities.
        """
        if not hasattr(self, "_X_context"):
            raise RuntimeError("TabularEmbedder is not fitted. Call fit(X, y) first.")
        if self.aggregate == "none":
            raise ValueError("search requires 2D embeddings; use aggregate='mean' or 'concat'.")

        if self.corpus_embeddings_ is None:
            self.corpus_embeddings_ = self.encode(self._X_context)

        top_k = min(top_k, len(self.corpus_embeddings_))
        sims = self.similarity(self.encode(X_query), self.corpus_embeddings_)
        indices = np.argsort(-sims, axis=1)[:, :top_k]
        scores = np.take_along_axis(sims, indices, axis=1)
        return indices, scores

    @property
    def embedding_dim(self) -> int | None:
        """Per-member embedding dimension, or None if not yet known."""
        return self.adapter.embedding_dim
