from __future__ import annotations

from typing import Any, TYPE_CHECKING

import httpx

from ..settings import normalize_settings
from ..types.enums import RerankBackendType
from ..types.llm_parameters import EndpointSetting, ResponseMapping
from ..types.retrieval_parameters import RerankResponse, RerankResult, RerankUsage
from ..types.settings import SettingsDict
from ..utilities.gcp_token import get_token_with_cache
from ..retrieval_clients.common import BaseAsyncRetrievalClient, BaseRetrievalClient
from ..retrieval_clients.common import build_url, extract_json_path, render_template

if TYPE_CHECKING:
    from ..settings import Settings

DEFAULT_RERANK_PROTOCOLS: dict[str, str] = {
    RerankBackendType.Cohere.value: "cohere_rerank_v2",
    RerankBackendType.Jina.value: "jina_rerank_v1",
    RerankBackendType.Voyage.value: "voyage_rerank_v1",
    RerankBackendType.OpenAI.value: "custom_json_http",
    RerankBackendType.Local.value: "custom_json_http",
    RerankBackendType.Custom.value: "custom_json_http",
}


def _normalize_backend_name(backend: RerankBackendType | str) -> str:
    if isinstance(backend, RerankBackendType):
        return backend.value
    return str(backend).lower()


def _normalize_document(value: Any) -> str | dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if set(value.keys()) == {"text"}:
            return value.get("text")
        return value
    if isinstance(value, str):
        return value
    return str(value)


def _usage_from_mapping(raw: dict[str, Any], response_mapping: ResponseMapping | None) -> RerankUsage | None:
    if response_mapping is None:
        return None
    search_units_path = response_mapping.usage_map.get("search_units")
    total_tokens_path = response_mapping.usage_map.get("total_tokens")

    search_units = None
    if search_units_path:
        search_units = extract_json_path(raw, search_units_path)
        if search_units is None:
            raise ValueError(
                f"Rerank response_mapping.usage_map.search_units path '{search_units_path}' not found in response."
            )

    total_tokens = None
    if total_tokens_path:
        total_tokens = extract_json_path(raw, total_tokens_path)
        if total_tokens is None:
            raise ValueError(
                f"Rerank response_mapping.usage_map.total_tokens path '{total_tokens_path}' not found in response."
            )

    if search_units is None and total_tokens is None:
        return None

    try:
        parsed_search_units = int(search_units) if search_units is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Rerank response_mapping.usage_map.search_units must resolve to an integer-compatible value."
        ) from exc

    try:
        parsed_total_tokens = int(total_tokens) if total_tokens is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Rerank response_mapping.usage_map.total_tokens must resolve to an integer-compatible value."
        ) from exc

    return RerankUsage(
        search_units=parsed_search_units,
        total_tokens=parsed_total_tokens,
    )


def _extract_item_field(item: Any, path: str | None):
    if path is None:
        return None
    if path.startswith("$"):
        return extract_json_path(item, path)
    return extract_json_path(item, f"$.{path}")


def _parse_result_list(
    *,
    items: list[Any],
    default_documents: list[str | dict],
    field_map: dict[str, str] | None = None,
) -> list[RerankResult]:
    field_map = field_map or {}
    results: list[RerankResult] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        mapped_index = _extract_item_field(item, field_map.get("index"))
        mapped_score = _extract_item_field(item, field_map.get("relevance_score"))
        mapped_document = _extract_item_field(item, field_map.get("document"))
        mapped_id = _extract_item_field(item, field_map.get("id"))
        mapped_metadata = _extract_item_field(item, field_map.get("metadata"))

        index_field_path = field_map.get("index")
        if index_field_path and mapped_index is None:
            raise ValueError(
                f"Rerank response_mapping.field_map.index path '{index_field_path}' not found for item[{idx}]."
            )

        try:
            result_index = int(mapped_index) if mapped_index is not None else int(item.get("index", idx))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Rerank item[{idx}] index must be integer-compatible, got: {mapped_index!r}."
            ) from exc

        score = mapped_score
        score_field_path = field_map.get("relevance_score")
        if score_field_path and score is None:
            raise ValueError(
                f"Rerank response_mapping.field_map.relevance_score path '{score_field_path}' not found for item[{idx}]."
            )
        if score is None:
            score = item.get("relevance_score", item.get("score", 0.0))
        try:
            score_value = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Rerank item[{idx}] relevance_score must be numeric-compatible, got: {score!r}."
            ) from exc

        document = mapped_document
        if document is None:
            document = item.get("document")
        if document is None and 0 <= result_index < len(default_documents):
            document = default_documents[result_index]

        result_id = mapped_id if mapped_id is not None else item.get("id")
        metadata = mapped_metadata if mapped_metadata is not None else item.get("metadata")

        results.append(
            RerankResult(
                index=result_index,
                relevance_score=score_value,
                document=_normalize_document(document),
                id=str(result_id) if result_id is not None else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )

    return results


def _parse_cohere_rerank(raw: dict[str, Any], model_id: str, documents: list[str | dict]) -> RerankResponse:
    items = raw.get("results", [])
    if not isinstance(items, list):
        items = []

    usage = None
    billed_units = raw.get("meta", {}).get("billed_units", {}) if isinstance(raw.get("meta"), dict) else {}
    if isinstance(billed_units, dict):
        search_units = billed_units.get("search_units")
        if search_units is not None:
            usage = RerankUsage(search_units=int(search_units))

    return RerankResponse(
        model=str(raw.get("model", model_id)),
        results=_parse_result_list(items=items, default_documents=documents),
        usage=usage,
        raw_response=raw,
    )


def _parse_jina_rerank(raw: dict[str, Any], model_id: str, documents: list[str | dict]) -> RerankResponse:
    items = raw.get("results", raw.get("data", []))
    if not isinstance(items, list):
        items = []

    usage_raw = raw.get("usage") if isinstance(raw.get("usage"), dict) else None
    usage = None
    if usage_raw is not None:
        total_tokens = usage_raw.get("total_tokens")
        usage = RerankUsage(total_tokens=int(total_tokens) if total_tokens is not None else None)

    return RerankResponse(
        model=str(raw.get("model", model_id)),
        results=_parse_result_list(items=items, default_documents=documents),
        usage=usage,
        raw_response=raw,
    )


def _parse_voyage_rerank(raw: dict[str, Any], model_id: str, documents: list[str | dict]) -> RerankResponse:
    items = raw.get("data", raw.get("results", []))
    if not isinstance(items, list):
        items = []

    usage = None
    usage_raw = raw.get("usage") if isinstance(raw.get("usage"), dict) else None
    if usage_raw is not None:
        total_tokens = usage_raw.get("total_tokens")
        usage = RerankUsage(total_tokens=int(total_tokens) if total_tokens is not None else None)

    return RerankResponse(
        model=str(raw.get("model", model_id)),
        results=_parse_result_list(items=items, default_documents=documents),
        usage=usage,
        raw_response=raw,
    )


def _parse_custom_rerank(
    raw: dict[str, Any],
    model_id: str,
    documents: list[str | dict],
    response_mapping: ResponseMapping | None,
) -> RerankResponse:
    results_path = response_mapping.results_path if response_mapping and response_mapping.results_path else "$.results[*]"
    items = extract_json_path(raw, results_path)

    if items is None and response_mapping and response_mapping.results_path:
        raise ValueError(f"Rerank response_mapping.results_path '{results_path}' not found in response.")

    if items is None:
        items = []
    if not isinstance(items, list):
        items = [items]

    model = model_id
    if response_mapping and response_mapping.model_path:
        mapped_model = extract_json_path(raw, response_mapping.model_path)
        if mapped_model is None:
            raise ValueError(
                f"Rerank response_mapping.model_path '{response_mapping.model_path}' not found in response."
            )
        model = str(mapped_model)

    field_map = response_mapping.field_map if response_mapping is not None else {}
    usage = _usage_from_mapping(raw, response_mapping)

    results = _parse_result_list(items=items, default_documents=documents, field_map=field_map)
    if not results:
        raise ValueError(
            f"Rerank response produced no results after applying mapping. results_path='{results_path}'."
        )

    return RerankResponse(
        model=model,
        results=results,
        usage=usage,
        raw_response=raw,
    )


def _default_auth_headers(endpoint: EndpointSetting) -> dict[str, str]:
    if endpoint.is_azure or endpoint.endpoint_type == "openai_azure":
        return {"api-key": endpoint.api_key or ""}
    if endpoint.endpoint_type == "openai_vertex":
        if endpoint.credentials is None:
            raise ValueError("OpenAI Vertex endpoint requires credentials")
        access_token, expires_at = get_token_with_cache(
            credentials=endpoint.credentials,
            proxy=endpoint.proxy,
            cached_token=endpoint.access_token,
            cached_expires_at=endpoint.access_token_expires_at,
        )
        endpoint.access_token = access_token
        endpoint.access_token_expires_at = expires_at
        return {"Authorization": f"Bearer {access_token}"}
    if endpoint.api_key:
        return {"Authorization": f"Bearer {endpoint.api_key}"}
    return {}


class RerankClient(BaseRetrievalClient):
    def __init__(
        self,
        backend: RerankBackendType | str,
        model: str | None = None,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.Client | None = None,
        settings: Settings | SettingsDict | None = None,
    ):
        normalized_settings = normalize_settings(settings)
        backend_name = _normalize_backend_name(backend)
        backend_settings = normalized_settings.get_rerank_backend(backend_name)

        super().__init__(
            model=model or "",
            backend_name=backend_name,
            backend_settings=backend_settings,
            random_endpoint=random_endpoint,
            endpoint_id=endpoint_id,
            http_client=http_client,
            settings=normalized_settings,
        )

    def _switch_model(self, model: str):
        if model == self.model:
            return
        self.model = model
        self.model_setting = self.backend_settings.get_model_setting(model)
        self.model_id = self.model_setting.id
        self.endpoint = None

    def _request_json(
        self,
        *,
        endpoint: EndpointSetting,
        method: str,
        path: str,
        headers: dict[str, str] | None,
        body: dict | list | str | None,
        query: dict | None,
        timeout: float | httpx.Timeout | None,
    ) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json", **_default_auth_headers(endpoint)}
        if headers:
            request_headers.update(headers)

        client = self.http_client or httpx.Client(proxy=endpoint.proxy)
        should_close = self.http_client is None

        try:
            response = client.request(
                method=method.upper(),
                url=build_url(endpoint.api_base, path),
                headers=request_headers,
                params=query,
                json=body,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        finally:
            if should_close:
                client.close()

    def _build_custom_request(
        self,
        *,
        model_id: str,
        query_text: str,
        documents: list[str | dict],
        top_n: int | None,
        return_documents: bool,
        extra_body: dict[str, Any] | None,
    ) -> tuple[str, str, dict[str, str] | None, dict | list | str | None, dict | None]:
        mapping = self.model_setting.request_mapping
        method = mapping.method if mapping else "POST"
        path = mapping.path if mapping else "/rerank"

        context = {
            "model": self.model,
            "model_id": model_id,
            "query": query_text,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }

        headers = render_template(mapping.headers, context) if mapping and mapping.headers else None
        body = render_template(mapping.body_template, context) if mapping and mapping.body_template is not None else None
        query = render_template(mapping.query_template, context) if mapping and mapping.query_template is not None else None

        if body is None:
            body = {
                "model": model_id,
                "query": query_text,
                "documents": documents,
                "top_n": top_n,
                "return_documents": return_documents,
            }
        if extra_body and isinstance(body, dict):
            body.update(extra_body)

        return method, path, headers, body, query

    def rerank(
        self,
        *,
        query: str,
        documents: list[str | dict],
        top_n: int | None = None,
        return_documents: bool = True,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> RerankResponse:
        if model is not None:
            self._switch_model(model)

        endpoint, model_id = self._set_endpoint()
        protocol = self.model_setting.protocol or DEFAULT_RERANK_PROTOCOLS.get(self.backend_name, "custom_json_http")

        effective_top_n = top_n if top_n is not None else self.model_setting.default_top_n
        request_body_for_rate = {
            "query": query,
            "documents": documents,
            "top_n": effective_top_n,
        }
        self._acquire_rate_limit(endpoint, request_body_for_rate)

        if protocol == "cohere_rerank_v2":
            body = {
                "model": model_id,
                "query": query,
                "documents": documents,
                "return_documents": return_documents,
            }
            if effective_top_n is not None:
                body["top_n"] = effective_top_n
            if extra_body:
                body.update(extra_body)
            raw = self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/v2/rerank",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_cohere_rerank(raw, model_id, documents)

        if protocol == "jina_rerank_v1":
            body = {
                "model": model_id,
                "query": query,
                "documents": documents,
                "return_documents": return_documents,
            }
            if effective_top_n is not None:
                body["top_n"] = effective_top_n
            if extra_body:
                body.update(extra_body)
            raw = self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/v1/rerank",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_jina_rerank(raw, model_id, documents)

        if protocol == "voyage_rerank_v1":
            body = {
                "model": model_id,
                "query": query,
                "documents": documents,
                "return_documents": return_documents,
            }
            if effective_top_n is not None:
                body["top_k"] = effective_top_n
            if extra_body:
                body.update(extra_body)
            raw = self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/rerank",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_voyage_rerank(raw, model_id, documents)

        if protocol == "custom_json_http":
            method, path, headers, body, req_query = self._build_custom_request(
                model_id=model_id,
                query_text=query,
                documents=documents,
                top_n=effective_top_n,
                return_documents=return_documents,
                extra_body=extra_body,
            )
            raw = self._request_json(
                endpoint=endpoint,
                method=method,
                path=path,
                headers=headers,
                body=body,
                query=req_query if isinstance(req_query, dict) else None,
                timeout=timeout,
            )
            return _parse_custom_rerank(raw, model_id, documents, self.model_setting.response_mapping)

        raise ValueError(f"Unsupported rerank protocol: {protocol}")

    def create_rerank(
        self,
        *,
        query: str,
        documents: list[str | dict],
        top_n: int | None = None,
        return_documents: bool = True,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> RerankResponse:
        return self.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            model=model,
            extra_body=extra_body,
            timeout=timeout,
        )


class AsyncRerankClient(BaseAsyncRetrievalClient):
    def __init__(
        self,
        backend: RerankBackendType | str,
        model: str | None = None,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.AsyncClient | None = None,
        settings: Settings | SettingsDict | None = None,
    ):
        normalized_settings = normalize_settings(settings)
        backend_name = _normalize_backend_name(backend)
        backend_settings = normalized_settings.get_rerank_backend(backend_name)

        super().__init__(
            model=model or "",
            backend_name=backend_name,
            backend_settings=backend_settings,
            random_endpoint=random_endpoint,
            endpoint_id=endpoint_id,
            http_client=http_client,
            settings=normalized_settings,
        )

    def _switch_model(self, model: str):
        if model == self.model:
            return
        self.model = model
        self.model_setting = self.backend_settings.get_model_setting(model)
        self.model_id = self.model_setting.id
        self.endpoint = None

    async def _request_json(
        self,
        *,
        endpoint: EndpointSetting,
        method: str,
        path: str,
        headers: dict[str, str] | None,
        body: dict | list | str | None,
        query: dict | None,
        timeout: float | httpx.Timeout | None,
    ) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json", **_default_auth_headers(endpoint)}
        if headers:
            request_headers.update(headers)

        client = self.http_client or httpx.AsyncClient(proxy=endpoint.proxy)
        should_close = self.http_client is None

        try:
            response = await client.request(
                method=method.upper(),
                url=build_url(endpoint.api_base, path),
                headers=request_headers,
                params=query,
                json=body,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        finally:
            if should_close:
                await client.aclose()

    def _build_custom_request(
        self,
        *,
        model_id: str,
        query_text: str,
        documents: list[str | dict],
        top_n: int | None,
        return_documents: bool,
        extra_body: dict[str, Any] | None,
    ) -> tuple[str, str, dict[str, str] | None, dict | list | str | None, dict | None]:
        mapping = self.model_setting.request_mapping
        method = mapping.method if mapping else "POST"
        path = mapping.path if mapping else "/rerank"

        context = {
            "model": self.model,
            "model_id": model_id,
            "query": query_text,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }

        headers = render_template(mapping.headers, context) if mapping and mapping.headers else None
        body = render_template(mapping.body_template, context) if mapping and mapping.body_template is not None else None
        query = render_template(mapping.query_template, context) if mapping and mapping.query_template is not None else None

        if body is None:
            body = {
                "model": model_id,
                "query": query_text,
                "documents": documents,
                "top_n": top_n,
                "return_documents": return_documents,
            }
        if extra_body and isinstance(body, dict):
            body.update(extra_body)

        return method, path, headers, body, query

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str | dict],
        top_n: int | None = None,
        return_documents: bool = True,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> RerankResponse:
        if model is not None:
            self._switch_model(model)

        endpoint, model_id = self._set_endpoint()
        protocol = self.model_setting.protocol or DEFAULT_RERANK_PROTOCOLS.get(self.backend_name, "custom_json_http")

        effective_top_n = top_n if top_n is not None else self.model_setting.default_top_n
        request_body_for_rate = {
            "query": query,
            "documents": documents,
            "top_n": effective_top_n,
        }
        await self._acquire_rate_limit(endpoint, request_body_for_rate)

        if protocol == "cohere_rerank_v2":
            body = {
                "model": model_id,
                "query": query,
                "documents": documents,
                "return_documents": return_documents,
            }
            if effective_top_n is not None:
                body["top_n"] = effective_top_n
            if extra_body:
                body.update(extra_body)
            raw = await self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/v2/rerank",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_cohere_rerank(raw, model_id, documents)

        if protocol == "jina_rerank_v1":
            body = {
                "model": model_id,
                "query": query,
                "documents": documents,
                "return_documents": return_documents,
            }
            if effective_top_n is not None:
                body["top_n"] = effective_top_n
            if extra_body:
                body.update(extra_body)
            raw = await self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/v1/rerank",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_jina_rerank(raw, model_id, documents)

        if protocol == "voyage_rerank_v1":
            body = {
                "model": model_id,
                "query": query,
                "documents": documents,
                "return_documents": return_documents,
            }
            if effective_top_n is not None:
                body["top_k"] = effective_top_n
            if extra_body:
                body.update(extra_body)
            raw = await self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/rerank",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_voyage_rerank(raw, model_id, documents)

        if protocol == "custom_json_http":
            method, path, headers, body, req_query = self._build_custom_request(
                model_id=model_id,
                query_text=query,
                documents=documents,
                top_n=effective_top_n,
                return_documents=return_documents,
                extra_body=extra_body,
            )
            raw = await self._request_json(
                endpoint=endpoint,
                method=method,
                path=path,
                headers=headers,
                body=body,
                query=req_query if isinstance(req_query, dict) else None,
                timeout=timeout,
            )
            return _parse_custom_rerank(raw, model_id, documents, self.model_setting.response_mapping)

        raise ValueError(f"Unsupported rerank protocol: {protocol}")

    async def create_rerank(
        self,
        *,
        query: str,
        documents: list[str | dict],
        top_n: int | None = None,
        return_documents: bool = True,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> RerankResponse:
        return await self.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            model=model,
            extra_body=extra_body,
            timeout=timeout,
        )


__all__ = [
    "RerankClient",
    "AsyncRerankClient",
]
