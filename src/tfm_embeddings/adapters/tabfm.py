from __future__ import annotations

import numpy as np
from sklearn.utils.multiclass import type_of_target

from .base import BackendAdapter


class TabFMAdapter(BackendAdapter):
    """Google TabFM backend (https://github.com/google-research/tabfm).

    Uses the PyTorch implementation of TabFM v1.0.0. Embeddings are the
    per-row representations produced right before the in-context learning
    transformer (the output of the second row-interaction stage,
    ``row_interactor_2``), extracted with a forward hook. With the released
    v1.0.0 checkpoints this gives 2048-dimensional vectors
    (``embed_dim=256 x row_num_cls=8``).

    Notes
    -----
    - Requires the PyTorch backend: ``pip install tabfm[pytorch]`` (the JAX
      backend does not support forward hooks). TabFM requires Python >= 3.11.
    - Rows passed to ``encode`` are always embedded as *test* rows, so their
      own labels are never visible to the model.
    - With ``y=None``, a standard-normal pseudo-target is synthesized and the
      regression checkpoint is used. This is experimental.

    Parameters
    ----------
    variant : str or None
        Optional local checkpoint path, passed to the TabFM loader as
        ``checkpoint_path``. None downloads the default weights from
        Hugging Face (``google/tabfm-1.0.0-pytorch``).

    n_estimators : int, default=8
        Number of TabFM ensemble members. The upstream default is 32, which
        is accurate but slow for embedding purposes.

    device : str or None, default=None
        Device for the torch model, e.g. ``"cpu"`` or ``"cuda"``. None keeps
        the loader default (CPU).

    random_state : int, default=42
        Seed for the estimator and for the pseudo-target used when ``y=None``.

    **estimator_kwargs
        Forwarded to the underlying ``TabFMClassifier`` / ``TabFMRegressor``.
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
            import tabfm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The TabFM backend requires the tabfm package with its PyTorch "
                "extra (and Python >= 3.11). "
                "Install it with: pip install tfm-embeddings[tabfm]"
            ) from e
        try:
            from tabfm import tabfm_v1_0_0_pytorch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The TabFM backend requires the PyTorch implementation of TabFM. "
                "Install it with: pip install tabfm[pytorch]"
            ) from e

        self.variant = variant
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

    def fit(self, X, y=None) -> "TabFMAdapter":
        from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch

        if y is None:
            # Pseudo-target: uninformative continuous target so the
            # embeddings reflect the features. Experimental.
            rng = np.random.default_rng(self.random_state)
            y = rng.standard_normal(len(X))
            model_type, estimator_cls = "regression", TabFMRegressor
        else:
            y = np.asarray(y)
            target_type = type_of_target(y)
            if target_type in ("binary", "multiclass"):
                model_type, estimator_cls = "classification", TabFMClassifier
            elif target_type == "continuous":
                model_type, estimator_cls = "regression", TabFMRegressor
            else:
                raise ValueError(
                    f"Could not infer task from target type '{target_type}'. "
                    "Pass a discrete y for classification or a continuous y for regression."
                )

        torch_model = tabfm_v1_0_0_pytorch.load(
            model_type, checkpoint_path=self.variant, device=self.device
        )

        self.estimator_ = estimator_cls(
            torch_model,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.estimator_kwargs,
        )
        self.estimator_.fit(X, y)
        # Row representations are the concatenated CLS tokens: (num_cls, embed_dim)
        self.embedding_dim = int(torch_model.cls_tokens.numel())
        return self

    def encode(self, X) -> np.ndarray:
        import torch

        if not hasattr(self, "estimator_"):
            raise RuntimeError("Adapter is not fitted. Call fit(X, y) first.")

        n_rows = len(X)
        captured = []
        # row_interactor_2 produces the per-row representations consumed by
        # the in-context learning transformer.
        hook = self.estimator_.model.row_interactor_2.register_forward_hook(
            lambda module, args, output: captured.append(output.detach().float().cpu())
        )
        try:
            # predict() forwards [context; X] through the model per ensemble
            # batch; in the PyTorch path the encoded rows occupy the trailing
            # sequence positions of every forward pass.
            self.estimator_.predict(X)
        finally:
            hook.remove()

        members = torch.cat([reprs[:, -n_rows:, :] for reprs in captured], dim=0)
        return members.numpy()
