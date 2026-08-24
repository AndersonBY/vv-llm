"""Runtime model defaults sourced from the vendored contract catalog."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any, cast

from ...contract import assert_contract_integrity, load_catalog
from .base import ModelSettingDict


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    """Load and validate the immutable-on-disk catalog once per process."""

    assert_contract_integrity()
    return load_catalog()


def load_backend_defaults(backend: str) -> tuple[str, dict[str, ModelSettingDict]]:
    """Return a backend's default model and an isolated model mapping.

    The mapping is copied for each consumer so legacy callers that mutate a
    ``*_MODELS`` dictionary cannot mutate the cached contract catalog.
    """

    catalog = _catalog()
    try:
        default_model = catalog["default_models"][backend]
        models = catalog["backends"][backend]["models"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"contract catalog is missing backend {backend!r}") from exc
    if not isinstance(default_model, str) or not isinstance(models, dict):
        raise RuntimeError(f"contract catalog backend {backend!r} has invalid defaults")
    return default_model, cast(dict[str, ModelSettingDict], deepcopy(models))


__all__ = ["load_backend_defaults"]
