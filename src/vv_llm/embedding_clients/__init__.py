from __future__ import annotations

import httpx
from typing import TYPE_CHECKING

from .base_client import EmbeddingClient, AsyncEmbeddingClient
from ..types.enums import EmbeddingBackendType

if TYPE_CHECKING:
    from ..settings import Settings
    from ..types.settings import SettingsDict


def create_embedding_client(
    backend: EmbeddingBackendType | str,
    model: str | None = None,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: Settings | SettingsDict | None = None,
) -> EmbeddingClient:
    return EmbeddingClient(
        backend=backend,
        model=model,
        random_endpoint=random_endpoint,
        endpoint_id=endpoint_id,
        http_client=http_client,
        settings=settings,
    )


def create_async_embedding_client(
    backend: EmbeddingBackendType | str,
    model: str | None = None,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | SettingsDict | None = None,
) -> AsyncEmbeddingClient:
    return AsyncEmbeddingClient(
        backend=backend,
        model=model,
        random_endpoint=random_endpoint,
        endpoint_id=endpoint_id,
        http_client=http_client,
        settings=settings,
    )


__all__ = [
    "EmbeddingBackendType",
    "EmbeddingClient",
    "AsyncEmbeddingClient",
    "create_embedding_client",
    "create_async_embedding_client",
]
