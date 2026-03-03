from __future__ import annotations

import httpx
from typing import TYPE_CHECKING

from .base_client import RerankClient, AsyncRerankClient
from ..types.enums import RerankBackendType

if TYPE_CHECKING:
    from ..settings import Settings
    from ..types.settings import SettingsDict


def create_rerank_client(
    backend: RerankBackendType | str,
    model: str | None = None,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: Settings | SettingsDict | None = None,
) -> RerankClient:
    return RerankClient(
        backend=backend,
        model=model,
        random_endpoint=random_endpoint,
        endpoint_id=endpoint_id,
        http_client=http_client,
        settings=settings,
    )


def create_async_rerank_client(
    backend: RerankBackendType | str,
    model: str | None = None,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: Settings | SettingsDict | None = None,
) -> AsyncRerankClient:
    return AsyncRerankClient(
        backend=backend,
        model=model,
        random_endpoint=random_endpoint,
        endpoint_id=endpoint_id,
        http_client=http_client,
        settings=settings,
    )


__all__ = [
    "RerankBackendType",
    "RerankClient",
    "AsyncRerankClient",
    "create_rerank_client",
    "create_async_rerank_client",
]
