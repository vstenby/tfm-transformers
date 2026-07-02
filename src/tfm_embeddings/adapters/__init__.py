from __future__ import annotations

import importlib

from .base import BackendAdapter

#: Registry mapping backend names to (module, class). Modules are imported
#: lazily so that installing one backend does not require the others.
_REGISTRY: dict[str, tuple[str, str]] = {
    "tabicl": ("tfm_embeddings.adapters.tabicl", "TabICLAdapter"),
    "tabpfn": ("tfm_embeddings.adapters.tabpfn", "TabPFNAdapter"),
    "tabfm": ("tfm_embeddings.adapters.tabfm", "TabFMAdapter"),
}


def available_backends() -> list[str]:
    """Names of all registered backends (installed or not)."""
    return sorted(_REGISTRY)


def resolve_backend(name: str) -> type[BackendAdapter]:
    """Return the adapter class registered under ``name``."""
    try:
        module_name, class_name = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {', '.join(available_backends())}."
        ) from None
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


__all__ = ["BackendAdapter", "available_backends", "resolve_backend"]
