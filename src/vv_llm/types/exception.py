from __future__ import annotations

from enum import Enum
from typing import Any

import httpx
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from openai import APIConnectionError, APIStatusError, APITimeoutError


class ErrorKind(str, Enum):
    AUTHENTICATION = "authentication"
    RATE_LIMITED = "rate_limited"
    NETWORK = "network"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LENGTH = "context_length"
    CONTENT_POLICY = "content_policy"
    MODEL_NOT_FOUND = "model_not_found"
    PROVIDER_INTERNAL = "provider_internal"
    SERIALIZATION = "serialization"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class VvLlmError(Exception):
    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind = ErrorKind.UNKNOWN,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        provider_code: str | None = None,
        retry_after: float | None = None,
        request_id: str | None = None,
        raw_body: str | None = None,
        source: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.provider_code = provider_code
        self.retry_after = retry_after
        self.request_id = request_id
        self.raw_body = raw_body
        self.source = source

    @property
    def retryable(self) -> bool:
        return self.kind in {
            ErrorKind.RATE_LIMITED,
            ErrorKind.NETWORK,
            ErrorKind.TIMEOUT,
            ErrorKind.PROVIDER_INTERNAL,
        }


def classify_exception(
    exception: BaseException,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> VvLlmError:
    if isinstance(exception, VvLlmError):
        return exception

    if isinstance(exception, (APITimeoutError, AnthropicAPITimeoutError, httpx.TimeoutException)):
        return VvLlmError(
            str(exception),
            kind=ErrorKind.TIMEOUT,
            provider=provider,
            model=model,
            source=exception,
        )
    if isinstance(exception, (APIConnectionError, AnthropicAPIConnectionError, httpx.NetworkError)):
        return VvLlmError(
            str(exception),
            kind=ErrorKind.NETWORK,
            provider=provider,
            model=model,
            source=exception,
        )
    if isinstance(exception, (APIStatusError, AnthropicAPIStatusError)):
        status_code = int(exception.status_code)
        body = getattr(exception, "body", None)
        provider_code = _provider_code(body)
        return VvLlmError(
            str(exception),
            kind=_status_error_kind(status_code, provider_code, str(exception)),
            provider=provider,
            model=model,
            status_code=status_code,
            provider_code=provider_code,
            retry_after=_retry_after(getattr(exception, "response", None)),
            request_id=getattr(exception, "request_id", None),
            source=exception,
        )
    if isinstance(exception, (ValueError, TypeError)):
        return VvLlmError(
            str(exception),
            kind=ErrorKind.INVALID_REQUEST,
            provider=provider,
            model=model,
            source=exception,
        )
    return VvLlmError(
        str(exception),
        kind=ErrorKind.UNKNOWN,
        provider=provider,
        model=model,
        source=exception,
    )


def _provider_code(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None
    code = error.get("code") or error.get("type")
    return str(code) if code is not None else None


def _status_error_kind(status_code: int, provider_code: str | None, message: str) -> ErrorKind:
    normalized = f"{provider_code or ''} {message}".lower()
    if status_code in {401, 403}:
        return ErrorKind.AUTHENTICATION
    if status_code == 404:
        return ErrorKind.MODEL_NOT_FOUND
    if status_code == 429:
        return ErrorKind.RATE_LIMITED
    if "context" in normalized and ("length" in normalized or "token" in normalized):
        return ErrorKind.CONTEXT_LENGTH
    if "content" in normalized and ("policy" in normalized or "filter" in normalized):
        return ErrorKind.CONTENT_POLICY
    if 400 <= status_code < 500:
        return ErrorKind.INVALID_REQUEST
    if status_code >= 500:
        return ErrorKind.PROVIDER_INTERNAL
    return ErrorKind.UNKNOWN


def _retry_after(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


__all__ = [
    "APIConnectionError",
    "APIStatusError",
    "ErrorKind",
    "VvLlmError",
    "classify_exception",
]
