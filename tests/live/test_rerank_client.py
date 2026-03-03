from __future__ import annotations

import asyncio
import os

from vv_llm.rerank_clients import create_async_rerank_client, create_rerank_client
from vv_llm.settings import settings

from live_common import load_live_settings, run_with_timer

DEFAULT_BACKEND = os.getenv("VV_LLM_RERANK_BACKEND", "siliconflow")
DEFAULT_MODEL = os.getenv("VV_LLM_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEFAULT_QUERY = os.getenv("VV_LLM_RERANK_QUERY", "Apple")
DEFAULT_DOCUMENTS = [
    item.strip()
    for item in os.getenv("VV_LLM_RERANK_DOCUMENTS", "apple,banana,fruit,vegetable").split(",")
    if item.strip()
]


def run_sync(backend: str, model: str, query: str, documents: list[str]) -> None:
    client = create_rerank_client(backend=backend, model=model, settings=settings)
    response = client.rerank(query=query, documents=documents, timeout=60)

    print(f"[rerank][sync] model={response.model} results={len(response.results)}")
    for idx, result in enumerate(response.results[:5]):
        print(
            f"[rerank][sync] top{idx} index={result.index} score={result.relevance_score:.6f} doc={result.document}"
        )
    if response.usage:
        print(f"[rerank][sync] usage={response.usage.model_dump()}")


async def run_async(backend: str, model: str, query: str, documents: list[str]) -> None:
    client = create_async_rerank_client(backend=backend, model=model, settings=settings)
    response = await client.rerank(query=query, documents=documents, timeout=60)

    print(f"[rerank][async] model={response.model} results={len(response.results)}")
    for idx, result in enumerate(response.results[:5]):
        print(
            f"[rerank][async] top{idx} index={result.index} score={result.relevance_score:.6f} doc={result.document}"
        )
    if response.usage:
        print(f"[rerank][async] usage={response.usage.model_dump()}")


def main() -> None:
    load_live_settings(settings)
    print(f"[live] rerank backend={DEFAULT_BACKEND} model={DEFAULT_MODEL}")

    try:
        run_with_timer("rerank_sync", lambda: run_sync(DEFAULT_BACKEND, DEFAULT_MODEL, DEFAULT_QUERY, DEFAULT_DOCUMENTS))
        run_with_timer(
            "rerank_async",
            lambda: asyncio.run(run_async(DEFAULT_BACKEND, DEFAULT_MODEL, DEFAULT_QUERY, DEFAULT_DOCUMENTS)),
        )
    except ValueError as exc:
        print(f"[live] skip rerank test: {exc}")


if __name__ == "__main__":
    main()
