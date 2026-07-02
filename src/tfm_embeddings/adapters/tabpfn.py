from __future__ import annotations

import numpy as np
from sklearn.utils.multiclass import type_of_target

from .base import BackendAdapter


class TabPFNAdapter(BackendAdapter):
    """TabPFN backend (https://github.com/PriorLabs/TabPFN).

    Embeddings are the per-row transformer outputs exposed by TabPFN's public
    ``get_embeddings(X, data_source="test")`` API, so rows passed to
    ``encode`` are always embedded as test rows against the fitted context
    and their own labels are never visible to the model.

    Requires the full local ``tabpfn`` package; the TabPFN client (API) does
    not support embedding extraction.

    Parameters
    ----------
    variant : str or None
        Optional model path or identifier, passed to the estimator as
        ``model_path``. None uses the default checkpoint.

    n_estimators : int or None, default=None
        Number of TabPFN ensemble members. None uses the estimator default.

    device : str or None, default=None
        Inference device. None auto-selects.

    random_state : int, default=42
        Seed for the estimator and for the pseudo-target used when ``y=None``.

    **estimator_kwargs
        Forwarded to the underlying ``TabPFNClassifier`` / ``TabPFNRegressor``.
    """

    def __init__(
        self,
        variant: str | None = None,
        *,
        n_estimators: int | None = None,
        device: str | None = None,
        random_state: int = 42,
        **estimator_kwargs,
    ):
        try:
            import tabpfn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The TabPFN backend requires the tabpfn package. "
                "Install it with: pip install tfm-embeddings[tabpfn]"
            ) from e

        self.variant = variant
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

    def fit(self, X, y=None) -> "TabPFNAdapter":
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        if y is None:
            # Pseudo-target: uninformative continuous target so the
            # embeddings reflect the features. Experimental.
            rng = np.random.default_rng(self.random_state)
            y = rng.standard_normal(len(X))
            estimator_cls = TabPFNRegressor
        else:
            y = np.asarray(y)
            target_type = type_of_target(y)
            if target_type in ("binary", "multiclass"):
                estimator_cls = TabPFNClassifier
            elif target_type == "continuous":
                estimator_cls = TabPFNRegressor
            else:
                raise ValueError(
                    f"Could not infer task from target type '{target_type}'. "
                    "Pass a discrete y for classification or a continuous y for regression."
                )

        kwargs = dict(random_state=self.random_state, **self.estimator_kwargs)
        if self.n_estimators is not None:
            kwargs["n_estimators"] = self.n_estimators
        if self.device is not None:
            kwargs["device"] = self.device
        if self.variant is not None:
            kwargs["model_path"] = self.variant

        self.estimator_ = estimator_cls(**kwargs)
        self.estimator_.fit(X, y)
        return self

    def encode(self, X) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Adapter is not fitted. Call fit(X, y) first.")

        embeddings = np.asarray(self.estimator_.get_embeddings(X, data_source="test"))
        if embeddings.ndim == 2:  # single ensemble member
            embeddings = embeddings[None]
        self.embedding_dim = embeddings.shape[-1]
        return embeddings
