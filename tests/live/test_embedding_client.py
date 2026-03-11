from __future__ import annotations

import asyncio
import os

from vv_llm.embedding_clients import create_async_embedding_client, create_embedding_client
from vv_llm.settings import settings

from live_common import load_live_settings, run_with_timer

DEFAULT_BACKEND = os.getenv("VV_LLM_EMBEDDING_BACKEND", "siliconflow")
DEFAULT_MODEL = os.getenv("VV_LLM_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
DEFAULT_TEXT = os.getenv(
    "VV_LLM_EMBEDDING_TEXT",
    "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!",
)


def run_sync(backend: str, model: str, text: str) -> None:
    client = create_embedding_client(backend=backend, model=model, settings=settings)
    response = client.create_embeddings(input=text, timeout=60)

    print(f"[embedding][sync] model={response.model} items={len(response.data)}")
    if response.data:
        print(f"[embedding][sync] dim={len(response.data[0].embedding)}")
    if response.usage:
        print(f"[embedding][sync] usage={response.usage.model_dump()}")


async def run_async(backend: str, model: str, text: str) -> None:
    client = create_async_embedding_client(backend=backend, model=model, settings=settings)
    response = await client.create_embeddings(input=text, timeout=60)

    print(f"[embedding][async] model={response.model} items={len(response.data)}")
    if response.data:
        print(f"[embedding][async] dim={len(response.data[0].embedding)}")
    if response.usage:
        print(f"[embedding][async] usage={response.usage.model_dump()}")


def main() -> None:
    load_live_settings(settings)
    print(f"[live] embedding backend={DEFAULT_BACKEND} model={DEFAULT_MODEL}")

    try:
        run_with_timer("embedding_sync", lambda: run_sync(DEFAULT_BACKEND, DEFAULT_MODEL, DEFAULT_TEXT))
        run_with_timer("embedding_async", lambda: asyncio.run(run_async(DEFAULT_BACKEND, DEFAULT_MODEL, DEFAULT_TEXT)))
    except ValueError as exc:
        print(f"[live] skip embedding test: {exc}")


if __name__ == "__main__":
    main()
