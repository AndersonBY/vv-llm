from __future__ import annotations

import asyncio

import pytest

from vv_llm import ErrorKind as PublicErrorKind
from vv_llm import RetryPolicy as PublicRetryPolicy
from vv_llm import VvLlmError as PublicVvLlmError
from vv_llm.types.exception import ErrorKind, VvLlmError
from vv_llm.utilities.retry_executor import RetryPolicy, execute_with_retry, execute_with_retry_async


def test_error_and_retry_types_are_exported_from_package_root() -> None:
    assert PublicErrorKind is ErrorKind
    assert PublicRetryPolicy is RetryPolicy
    assert PublicVvLlmError is VvLlmError


def test_sync_retry_executor_retries_only_classified_errors() -> None:
    attempts = 0
    delays: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise VvLlmError(
                "rate limited",
                kind=ErrorKind.RATE_LIMITED,
                retry_after=2,
            )
        return "ok"

    result = execute_with_retry(
        operation,
        RetryPolicy(max_attempts=3, max_delay=1, jitter_ratio=0),
        sleep=delays.append,
    )

    assert result == "ok"
    assert attempts == 3
    assert delays == [1, 1]


def test_sync_retry_executor_does_not_retry_authentication_errors() -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise VvLlmError("unauthorized", kind=ErrorKind.AUTHENTICATION)

    with pytest.raises(VvLlmError) as caught:
        execute_with_retry(operation, RetryPolicy(max_attempts=3), sleep=lambda _: None)

    assert caught.value.kind is ErrorKind.AUTHENTICATION
    assert attempts == 1


def test_async_retry_executor_uses_the_same_policy() -> None:
    attempts = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise VvLlmError("temporary", kind=ErrorKind.PROVIDER_INTERNAL)
        return "ok"

    async def sleep(delay: float) -> None:
        delays.append(delay)

    result = asyncio.run(
        execute_with_retry_async(
            operation,
            RetryPolicy(max_attempts=2, base_delay=0.25, jitter_ratio=0),
            sleep=sleep,
        )
    )

    assert result == "ok"
    assert attempts == 2
    assert delays == [0.25]


def test_retry_executor_stops_before_total_timeout() -> None:
    attempts = 0
    clock_values = iter([0.0, 0.75])

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise VvLlmError("temporary", kind=ErrorKind.NETWORK)

    with pytest.raises(VvLlmError) as caught:
        execute_with_retry(
            operation,
            RetryPolicy(
                max_attempts=3,
                base_delay=0.5,
                jitter_ratio=0,
                total_timeout=1,
            ),
            sleep=lambda _: None,
            monotonic=lambda: next(clock_values),
        )

    assert caught.value.kind is ErrorKind.NETWORK
    assert attempts == 1
