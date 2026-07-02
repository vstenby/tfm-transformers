from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BackendAdapter(ABC):
    """Interface for a tabular foundation model backend.

    An adapter wraps one model family (TabICL, TabPFN, ...) and exposes two
    operations: ``fit`` establishes the context table that all embeddings are
    conditioned on, and ``encode`` embeds rows against that context.

    Adapters return raw per-ensemble-member embeddings of shape
    ``(n_members, n_rows, embed_dim)``; aggregation across members is handled
    by :class:`tfm_embeddings.TabularEmbedder`.
    """

    #: Set after fit; None if the backend cannot know the dimension upfront.
    embedding_dim: int | None = None

    @abstractmethod
    def fit(self, X, y=None) -> "BackendAdapter":
        """Set the context table.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Context rows. All later ``encode`` calls embed rows against this
            table.

        y : array-like of shape (n_samples,) or None
            Context targets. Backends with target-aware embeddings use these
            to shape the embedding space. When None, the adapter synthesizes
            a pseudo-target (see the adapter's documentation).
        """

    @abstractmethod
    def encode(self, X) -> np.ndarray:
        """Embed rows as unseen test rows against the fitted context.

        Returns
        -------
        np.ndarray of shape (n_members, n_rows, embed_dim)
        """
