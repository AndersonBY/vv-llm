from __future__ import annotations

from typing import Any, TYPE_CHECKING

import httpx

from ..settings import normalize_settings
from ..types.enums import EmbeddingBackendType
from ..types.llm_parameters import EndpointSetting, ResponseMapping
from ..types.retrieval_parameters import EmbeddingData, EmbeddingResponse, EmbeddingUsage
from ..types.settings import SettingsDict
from ..utilities.gcp_token import get_token_with_cache
from ..retrieval_clients.common import BaseAsyncRetrievalClient, BaseRetrievalClient
from ..retrieval_clients.common import async_request_json_with_retry, build_url, extract_json_path, render_template
from ..retrieval_clients.common import request_json_with_retry

if TYPE_CHECKING:
    from ..settings import Settings

DEFAULT_EMBEDDING_PROTOCOLS: dict[str, str] = {
    EmbeddingBackendType.OpenAI.value: "openai_embeddings",
    EmbeddingBackendType.Cohere.value: "cohere_embed_v2",
    EmbeddingBackendType.Jina.value: "openai_embeddings",
    EmbeddingBackendType.Voyage.value: "voyage_embeddings_v1",
    EmbeddingBackendType.Siliconflow.value: "siliconflow",
    EmbeddingBackendType.Local.value: "openai_embeddings",
    EmbeddingBackendType.Custom.value: "custom_json_http",
}


def _normalize_backend_name(backend: EmbeddingBackendType | str) -> str:
    if isinstance(backend, EmbeddingBackendType):
        return backend.value
    return str(backend).lower()


def _ensure_list_input(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return value


def _to_float_vector(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, tuple):
        return [float(v) for v in value]
    raise ValueError(f"Invalid embedding vector type: {type(value)}")


def _parse_openai_style_embeddings(raw: dict[str, Any], model_id: str, inputs: list[str]) -> EmbeddingResponse:
    data = []
    for idx, item in enumerate(raw.get("data", [])):
        if not isinstance(item, dict):
            continue
        data.append(
            EmbeddingData(
                index=int(item.get("index", idx)),
                embedding=_to_float_vector(item.get("embedding", [])),
                text=inputs[idx] if idx < len(inputs) else None,
            )
        )

    usage_raw = raw.get("usage") if isinstance(raw.get("usage"), dict) else None
    usage = None
    if usage_raw is not None:
        usage = EmbeddingUsage(
            prompt_tokens=usage_raw.get("prompt_tokens"),
            total_tokens=usage_raw.get("total_tokens"),
        )

    return EmbeddingResponse(
        model=str(raw.get("model", model_id)),
        data=data,
        usage=usage,
        raw_response=raw,
    )


def _parse_cohere_embeddings(raw: dict[str, Any], model_id: str, inputs: list[str]) -> EmbeddingResponse:
    vectors: list[list[float]] = []

    embeddings = raw.get("embeddings")
    if isinstance(embeddings, dict):
        float_vectors = embeddings.get("float")
        if isinstance(float_vectors, list):
            vectors = [_to_float_vector(v) for v in float_vectors]
    elif isinstance(embeddings, list):
        vectors = [_to_float_vector(v) for v in embeddings]

    data = [EmbeddingData(index=idx, embedding=vector, text=inputs[idx] if idx < len(inputs) else None) for idx, vector in enumerate(vectors)]

    billed_units = raw.get("meta", {}).get("billed_units", {}) if isinstance(raw.get("meta"), dict) else {}
    prompt_tokens = billed_units.get("input_tokens") if isinstance(billed_units, dict) else None
    usage = EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens) if prompt_tokens is not None else None

    return EmbeddingResponse(
        model=str(raw.get("model", model_id)),
        data=data,
        usage=usage,
        raw_response=raw,
    )


def _parse_siliconflow_embeddings(raw: dict[str, Any], model_id: str, inputs: list[str]) -> EmbeddingResponse:
    return _parse_openai_style_embeddings(raw, model_id, inputs)


def _usage_from_mapping(raw: dict[str, Any], response_mapping: ResponseMapping | None) -> EmbeddingUsage | None:
    if response_mapping is None:
        return None
    prompt_path = response_mapping.usage_map.get("prompt_tokens")
    total_path = response_mapping.usage_map.get("total_tokens")

    prompt_tokens = None
    if prompt_path:
        prompt_tokens = extract_json_path(raw, prompt_path)
        if prompt_tokens is None:
            raise ValueError(
                f"Embedding response_mapping.usage_map.prompt_tokens path '{prompt_path}' not found in response."
            )

    total_tokens = None
    if total_path:
        total_tokens = extract_json_path(raw, total_path)
        if total_tokens is None:
            raise ValueError(
                f"Embedding response_mapping.usage_map.total_tokens path '{total_path}' not found in response."
            )

    if prompt_tokens is None and total_tokens is None:
        return None

    try:
        parsed_prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Embedding response_mapping.usage_map.prompt_tokens must resolve to an integer-compatible value."
        ) from exc

    try:
        parsed_total_tokens = int(total_tokens) if total_tokens is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Embedding response_mapping.usage_map.total_tokens must resolve to an integer-compatible value."
        ) from exc

    return EmbeddingUsage(
        prompt_tokens=parsed_prompt_tokens,
        total_tokens=parsed_total_tokens,
    )


def _extract_item_field(item: Any, path: str | None):
    if path is None:
        return None
    if path.startswith("$"):
        return extract_json_path(item, path)
    return extract_json_path(item, f"$.{path}")


def _parse_custom_embeddings(raw: dict[str, Any], model_id: str, inputs: list[str], response_mapping: ResponseMapping | None) -> EmbeddingResponse:
    data_path = response_mapping.data_path if response_mapping and response_mapping.data_path else "$.data[*]"
    item_list = extract_json_path(raw, data_path)

    if item_list is None and response_mapping and response_mapping.data_path:
        raise ValueError(f"Embedding response_mapping.data_path '{data_path}' not found in response.")

    if item_list is None:
        item_list = []
    if not isinstance(item_list, list):
        item_list = [item_list]

    field_map = response_mapping.field_map if response_mapping is not None else {}
    response_data = []
    for idx, item in enumerate(item_list):
        mapped_index = _extract_item_field(item, field_map.get("index"))
        mapped_embedding = _extract_item_field(item, field_map.get("embedding"))
        mapped_text = _extract_item_field(item, field_map.get("text"))

        embedding_field_path = field_map.get("embedding")
        if embedding_field_path and mapped_embedding is None:
            raise ValueError(
                f"Embedding response_mapping.field_map.embedding path '{embedding_field_path}' not found for item[{idx}]."
            )

        if mapped_embedding is None:
            if isinstance(item, dict):
                mapped_embedding = item.get("embedding")
            elif isinstance(item, list):
                mapped_embedding = item

        if mapped_embedding is None:
            raise ValueError(
                f"Embedding item[{idx}] has no embedding vector. Provide field_map.embedding or ensure item contains 'embedding'."
            )

        index_field_path = field_map.get("index")
        if index_field_path and mapped_index is None:
            raise ValueError(
                f"Embedding response_mapping.field_map.index path '{index_field_path}' not found for item[{idx}]."
            )

        try:
            parsed_index = int(mapped_index) if mapped_index is not None else idx
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Embedding item[{idx}] index must be integer-compatible, got: {mapped_index!r}."
            ) from exc

        try:
            parsed_embedding = _to_float_vector(mapped_embedding)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Embedding item[{idx}] vector must be a list/tuple of numeric values."
            ) from exc

        response_data.append(
            EmbeddingData(
                index=parsed_index,
                embedding=parsed_embedding,
                text=str(mapped_text) if mapped_text is not None else (inputs[idx] if idx < len(inputs) else None),
            )
        )

    model = model_id
    if response_mapping and response_mapping.model_path:
        mapped_model = extract_json_path(raw, response_mapping.model_path)
        if mapped_model is None:
            raise ValueError(
                f"Embedding response_mapping.model_path '{response_mapping.model_path}' not found in response."
            )
        model = str(mapped_model)

    if not response_data:
        raise ValueError(
            f"Embedding response produced no vectors after applying mapping. data_path='{data_path}'."
        )

    return EmbeddingResponse(
        model=model,
        data=response_data,
        usage=_usage_from_mapping(raw, response_mapping),
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


class EmbeddingClient(BaseRetrievalClient):
    def __init__(
        self,
        backend: EmbeddingBackendType | str,
        model: str | None = None,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.Client | None = None,
        settings: Settings | SettingsDict | None = None,
    ):
        normalized_settings = normalize_settings(settings)
        backend_name = _normalize_backend_name(backend)
        backend_settings = normalized_settings.get_embedding_backend(backend_name)

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
            return request_json_with_retry(
                client=client,
                method=method,
                url=build_url(endpoint.api_base, path),
                headers=request_headers,
                params=query,
                json_body=body,
                timeout=timeout,
            )
        finally:
            if should_close:
                client.close()

    def _build_custom_request(
        self,
        *,
        model_id: str,
        input_value: str | list[str],
        dimensions: int | None,
        extra_body: dict[str, Any] | None,
    ) -> tuple[str, str, dict[str, str] | None, dict | list | str | None, dict | None]:
        mapping = self.model_setting.request_mapping
        method = mapping.method if mapping else "POST"
        path = mapping.path if mapping else "/embeddings"

        context = {
            "model": self.model,
            "model_id": model_id,
            "input": input_value,
            "inputs": _ensure_list_input(input_value),
            "dimensions": dimensions,
        }

        headers = render_template(mapping.headers, context) if mapping and mapping.headers else None
        body = render_template(mapping.body_template, context) if mapping and mapping.body_template is not None else None
        query = render_template(mapping.query_template, context) if mapping and mapping.query_template is not None else None

        if body is None:
            body = {
                "model": model_id,
                "input": input_value,
            }
        if dimensions is not None and isinstance(body, dict) and "dimensions" not in body:
            body["dimensions"] = dimensions
        if extra_body and isinstance(body, dict):
            body.update(extra_body)

        return method, path, headers, body, query

    def create_embeddings(
        self,
        *,
        input: str | list[str],
        model: str | None = None,
        dimensions: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> EmbeddingResponse:
        if model is not None:
            self._switch_model(model)

        endpoint, model_id = self._set_endpoint()
        protocol = self.model_setting.protocol or DEFAULT_EMBEDDING_PROTOCOLS.get(self.backend_name, "openai_embeddings")

        request_body_for_rate = {
            "input": input,
            "model": model_id,
            "dimensions": dimensions,
            "extra_body": extra_body,
        }
        self._acquire_rate_limit(endpoint, request_body_for_rate)

        if protocol == "siliconflow":
            body: dict[str, Any] = {"model": model_id, "input": input}
            if dimensions is not None:
                body["dimensions"] = dimensions
            if extra_body:
                body.update(extra_body)

            raw = self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/embeddings",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_siliconflow_embeddings(raw, model_id, _ensure_list_input(input))

        if protocol == "openai_embeddings":
            if endpoint.is_azure or endpoint.endpoint_type == "openai_azure":
                path = f"/openai/deployments/{model_id}/embeddings"
                query = {"api-version": "2025-04-01-preview"}
                body: dict[str, Any] = {"input": input}
            else:
                path = "/embeddings"
                query = None
                body = {"model": model_id, "input": input}
            if dimensions is not None:
                body["dimensions"] = dimensions
            if extra_body:
                body.update(extra_body)

            raw = self._request_json(
                endpoint=endpoint,
                method="POST",
                path=path,
                headers=None,
                body=body,
                query=query,
                timeout=timeout,
            )
            return _parse_openai_style_embeddings(raw, model_id, _ensure_list_input(input))

        if protocol == "cohere_embed_v2":
            body = {"model": model_id, "texts": _ensure_list_input(input), "input_type": "search_document"}
            if extra_body:
                body.update(extra_body)
            raw = self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/v2/embed",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_cohere_embeddings(raw, model_id, _ensure_list_input(input))

        if protocol == "voyage_embeddings_v1":
            body = {"model": model_id, "input": input}
            if dimensions is not None:
                body["output_dimension"] = dimensions
            if extra_body:
                body.update(extra_body)
            raw = self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/embeddings",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_openai_style_embeddings(raw, model_id, _ensure_list_input(input))

        if protocol == "custom_json_http":
            method, path, headers, body, query = self._build_custom_request(
                model_id=model_id,
                input_value=input,
                dimensions=dimensions,
                extra_body=extra_body,
            )
            raw = self._request_json(
                endpoint=endpoint,
                method=method,
                path=path,
                headers=headers,
                body=body,
                query=query if isinstance(query, dict) else None,
                timeout=timeout,
            )
            return _parse_custom_embeddings(raw, model_id, _ensure_list_input(input), self.model_setting.response_mapping)

        raise ValueError(f"Unsupported embedding protocol: {protocol}")

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> list[float]:
        response = self.create_embeddings(
            input=text,
            model=model,
            dimensions=dimensions,
            extra_body=extra_body,
            timeout=timeout,
        )
        if not response.data:
            return []
        return response.data[0].embedding

    def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> list[list[float]]:
        response = self.create_embeddings(
            input=texts,
            model=model,
            dimensions=dimensions,
            extra_body=extra_body,
            timeout=timeout,
        )
        return [item.embedding for item in response.data]


class AsyncEmbeddingClient(BaseAsyncRetrievalClient):
    def __init__(
        self,
        backend: EmbeddingBackendType | str,
        model: str | None = None,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.AsyncClient | None = None,
        settings: Settings | SettingsDict | None = None,
    ):
        normalized_settings = normalize_settings(settings)
        backend_name = _normalize_backend_name(backend)
        backend_settings = normalized_settings.get_embedding_backend(backend_name)

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
            return await async_request_json_with_retry(
                client=client,
                method=method,
                url=build_url(endpoint.api_base, path),
                headers=request_headers,
                params=query,
                json_body=body,
                timeout=timeout,
            )
        finally:
            if should_close:
                await client.aclose()

    def _build_custom_request(
        self,
        *,
        model_id: str,
        input_value: str | list[str],
        dimensions: int | None,
        extra_body: dict[str, Any] | None,
    ) -> tuple[str, str, dict[str, str] | None, dict | list | str | None, dict | None]:
        mapping = self.model_setting.request_mapping
        method = mapping.method if mapping else "POST"
        path = mapping.path if mapping else "/embeddings"

        context = {
            "model": self.model,
            "model_id": model_id,
            "input": input_value,
            "inputs": _ensure_list_input(input_value),
            "dimensions": dimensions,
        }

        headers = render_template(mapping.headers, context) if mapping and mapping.headers else None
        body = render_template(mapping.body_template, context) if mapping and mapping.body_template is not None else None
        query = render_template(mapping.query_template, context) if mapping and mapping.query_template is not None else None

        if body is None:
            body = {
                "model": model_id,
                "input": input_value,
            }
        if dimensions is not None and isinstance(body, dict) and "dimensions" not in body:
            body["dimensions"] = dimensions
        if extra_body and isinstance(body, dict):
            body.update(extra_body)

        return method, path, headers, body, query

    async def create_embeddings(
        self,
        *,
        input: str | list[str],
        model: str | None = None,
        dimensions: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> EmbeddingResponse:
        if model is not None:
            self._switch_model(model)

        endpoint, model_id = self._set_endpoint()
        protocol = self.model_setting.protocol or DEFAULT_EMBEDDING_PROTOCOLS.get(self.backend_name, "openai_embeddings")

        request_body_for_rate = {
            "input": input,
            "model": model_id,
            "dimensions": dimensions,
            "extra_body": extra_body,
        }
        await self._acquire_rate_limit(endpoint, request_body_for_rate)

        if protocol == "siliconflow":
            body: dict[str, Any] = {"model": model_id, "input": input}
            if dimensions is not None:
                body["dimensions"] = dimensions
            if extra_body:
                body.update(extra_body)

            raw = await self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/embeddings",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_siliconflow_embeddings(raw, model_id, _ensure_list_input(input))

        if protocol == "openai_embeddings":
            if endpoint.is_azure or endpoint.endpoint_type == "openai_azure":
                path = f"/openai/deployments/{model_id}/embeddings"
                query = {"api-version": "2025-04-01-preview"}
                body: dict[str, Any] = {"input": input}
            else:
                path = "/embeddings"
                query = None
                body = {"model": model_id, "input": input}
            if dimensions is not None:
                body["dimensions"] = dimensions
            if extra_body:
                body.update(extra_body)

            raw = await self._request_json(
                endpoint=endpoint,
                method="POST",
                path=path,
                headers=None,
                body=body,
                query=query,
                timeout=timeout,
            )
            return _parse_openai_style_embeddings(raw, model_id, _ensure_list_input(input))

        if protocol == "cohere_embed_v2":
            body = {"model": model_id, "texts": _ensure_list_input(input), "input_type": "search_document"}
            if extra_body:
                body.update(extra_body)
            raw = await self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/v2/embed",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_cohere_embeddings(raw, model_id, _ensure_list_input(input))

        if protocol == "voyage_embeddings_v1":
            body = {"model": model_id, "input": input}
            if dimensions is not None:
                body["output_dimension"] = dimensions
            if extra_body:
                body.update(extra_body)
            raw = await self._request_json(
                endpoint=endpoint,
                method="POST",
                path="/embeddings",
                headers=None,
                body=body,
                query=None,
                timeout=timeout,
            )
            return _parse_openai_style_embeddings(raw, model_id, _ensure_list_input(input))

        if protocol == "custom_json_http":
            method, path, headers, body, query = self._build_custom_request(
                model_id=model_id,
                input_value=input,
                dimensions=dimensions,
                extra_body=extra_body,
            )
            raw = await self._request_json(
                endpoint=endpoint,
                method=method,
                path=path,
                headers=headers,
                body=body,
                query=query if isinstance(query, dict) else None,
                timeout=timeout,
            )
            return _parse_custom_embeddings(raw, model_id, _ensure_list_input(input), self.model_setting.response_mapping)

        raise ValueError(f"Unsupported embedding protocol: {protocol}")

    async def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> list[float]:
        response = await self.create_embeddings(
            input=text,
            model=model,
            dimensions=dimensions,
            extra_body=extra_body,
            timeout=timeout,
        )
        if not response.data:
            return []
        return response.data[0].embedding

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> list[list[float]]:
        response = await self.create_embeddings(
            input=texts,
            model=model,
            dimensions=dimensions,
            extra_body=extra_body,
            timeout=timeout,
        )
        return [item.embedding for item in response.data]


__all__ = [
    "EmbeddingClient",
    "AsyncEmbeddingClient",
]
