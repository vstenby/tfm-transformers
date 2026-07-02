from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils import _safe_indexing
from sklearn.utils.multiclass import type_of_target

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
        self._adapter_cls = resolve_backend(backend_name)
        self._variant = variant or None
        self._backend_kwargs = backend_kwargs

        self.model = model
        self.aggregate = aggregate
        self.adapter = self._adapter_cls(self._variant, **backend_kwargs)

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

    def fit_transform_oof(
        self,
        X,
        y=None,
        *,
        n_fold: int = 5,
        groups=None,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Out-of-fold embeddings of the training data, for downstream training.

        ``fit(X, y)`` followed by ``encode(X)`` embeds every row with an
        identical, labeled copy of itself in the context, so each embedding
        partially encodes its own label. That is harmless for retrieval and
        visualization, but when the embeddings become *features for training
        another model*, the downstream model learns to rely on a signal that
        is absent for genuinely unseen rows.

        This method removes that self-influence: the data is split into
        ``n_fold`` folds and each fold is embedded by a model fitted on the
        *other* folds, so no row (or its label) is ever part of the context
        that embeds it. Finally, one model is fitted on the full data — that
        model serves all subsequent ``encode`` calls for unseen rows.
        Follows "A Closer Look at TabPFN v2" (https://arxiv.org/abs/2502.17361).

        Note the asymmetry: ``encode`` always uses the final full-data model
        and never returns out-of-fold embeddings — calling ``encode(X)`` on
        the training data afterwards gives different (self-influenced)
        vectors. Keep this method's return value.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows.

        y : array-like of shape (n_samples,) or None
            Training targets. Discrete targets enable stratified splits.
            With None, the pseudo-target mechanism applies per fold.

        n_fold : int, default=5
            Number of folds. Total cost is ``n_fold + 1`` fits and encodes.

        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for rows that are not independent (duplicated
            entities, repeated measurements). Rows sharing a group are kept
            in the same fold, so an entity never appears in the context that
            embeds it. Plain K-fold cannot detect this — it is your domain
            knowledge.

        shuffle : bool, default=False
            Whether to shuffle rows before splitting. Ignored when ``groups``
            is given for a non-classification target (``GroupKFold``).

        random_state : int or None, default=None
            Seed for the split when ``shuffle=True``.

        Returns
        -------
        np.ndarray
            Out-of-fold embeddings aligned to the original row order,
            aggregated according to ``aggregate``.
        """
        if n_fold < 2:
            raise ValueError(f"n_fold must be >= 2, got {n_fold}.")

        y_arr = None if y is None else np.asarray(y)
        is_classification = y_arr is not None and type_of_target(y_arr) in ("binary", "multiclass")

        rs = random_state if shuffle else None
        if groups is not None:
            if is_classification:
                cv = StratifiedGroupKFold(n_splits=n_fold, shuffle=shuffle, random_state=rs)
            else:
                cv = GroupKFold(n_splits=n_fold)
            splits = cv.split(X, y_arr, groups)
        elif is_classification:
            cv = StratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=rs)
            splits = cv.split(X, y_arr)
        else:
            cv = KFold(n_splits=n_fold, shuffle=shuffle, random_state=rs)
            splits = cv.split(X)

        chunks = []
        val_indices = []
        for train_idx, val_idx in splits:
            fold_adapter = self._adapter_cls(self._variant, **self._backend_kwargs)
            fold_adapter.fit(
                _safe_indexing(X, train_idx),
                None if y_arr is None else y_arr[train_idx],
            )
            chunks.append(fold_adapter.encode(_safe_indexing(X, val_idx)))
            val_indices.append(val_idx)

        oof = np.concatenate(chunks, axis=1)
        order = np.argsort(np.concatenate(val_indices))
        oof = oof[:, order]

        # Final full-data model for embedding unseen rows via encode()
        self.fit(X, y)

        return self._aggregate(oof)

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
