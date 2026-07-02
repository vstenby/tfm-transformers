from __future__ import annotations

import numpy as np
from sklearn.utils.multiclass import type_of_target

from .base import BackendAdapter


class TabICLAdapter(BackendAdapter):
    """TabICL backend (https://github.com/soda-inria/tabicl).

    Embeddings are the row representations produced by TabICL's column-wise
    embedding and row-wise interaction stages (the vectors fed to the
    in-context learning transformer). They are extracted with a forward hook
    on ``row_interactor``, which works with released ``tabicl>=2.1`` without
    requiring any upstream changes.

    Notes
    -----
    - TabICLv2 checkpoints are target-aware: the context labels shape the
      embedding space. With ``y=None``, a standard-normal pseudo-target is
      synthesized and the regressor checkpoint is used, which makes the
      embeddings approximately unsupervised. This is experimental.
    - Rows passed to ``encode`` are always embedded as *test* rows, so their
      own labels are never visible to the model.

    Parameters
    ----------
    variant : str or None
        Optional checkpoint version, e.g. ``"tabicl-classifier-v2-20260212.ckpt"``.
        Passed to the estimator as ``checkpoint_version``. None uses the
        default checkpoint of the resolved estimator.

    n_estimators : int, default=8
        Number of TabICL ensemble members. Each member embeds a differently
        shuffled view of the features and contributes one embedding space.

    device : str or None, default=None
        Inference device. None auto-selects CUDA or CPU.

    random_state : int, default=42
        Seed for the estimator and for the pseudo-target used when ``y=None``.

    **estimator_kwargs
        Forwarded to the underlying ``TabICLClassifier`` / ``TabICLRegressor``.
    """

    def __init__(
        self,
        variant: str | None = None,
        *,
        n_estimators: int = 8,
        device: str | None = None,
        random_state: int = 42,
        **estimator_kwargs,
    ):
        try:
            import tabicl  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The TabICL backend requires the tabicl package. "
                "Install it with: pip install tfm-embeddings[tabicl]"
            ) from e

        self.variant = variant
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

    def fit(self, X, y=None) -> "TabICLAdapter":
        from tabicl import TabICLClassifier, TabICLRegressor

        if y is None:
            # Pseudo-target: TabICLv2 requires labels in context (target-aware
            # column embedding). A random continuous target provides an
            # uninformative signal so the embeddings reflect the features.
            rng = np.random.default_rng(self.random_state)
            y = rng.standard_normal(len(X))
            estimator_cls = TabICLRegressor
        else:
            y = np.asarray(y)
            target_type = type_of_target(y)
            if target_type in ("binary", "multiclass"):
                estimator_cls = TabICLClassifier
            elif target_type == "continuous":
                estimator_cls = TabICLRegressor
            else:
                raise ValueError(
                    f"Could not infer task from target type '{target_type}'. "
                    "Pass a discrete y for classification or a continuous y for regression."
                )

        kwargs = dict(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
            **self.estimator_kwargs,
        )
        if self.variant is not None:
            kwargs["checkpoint_version"] = self.variant

        self.estimator_ = estimator_cls(**kwargs)
        self.estimator_.fit(X, y)
        self.embedding_dim = self.estimator_.model_.embed_dim * self.estimator_.model_.row_num_cls
        return self

    def encode(self, X) -> np.ndarray:
        import torch

        if not hasattr(self, "estimator_"):
            raise RuntimeError("Adapter is not fitted. Call fit(X, y) first.")

        n_rows = len(X)
        captured = []
        hook = self.estimator_.model_.row_interactor.register_forward_hook(
            lambda module, args, output: captured.append(output.detach().float().cpu())
        )
        try:
            # predict() forwards [context; X] through the model; the hook
            # captures the row representations of every forward pass (one or
            # more per ensemble batch and normalization method).
            self.estimator_.predict(X)
        finally:
            hook.remove()

        # Keep only the representations of the encoded rows. They occupy the
        # trailing positions regardless of whether the context rows were part
        # of the same forward pass (no KV cache) or not (KV cache).
        members = torch.cat([reprs[:, -n_rows:, :] for reprs in captured], dim=0)
        return members.numpy()
