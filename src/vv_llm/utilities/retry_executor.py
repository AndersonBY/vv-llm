from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ..types.exception import ErrorKind, VvLlmError, classify_exception


ResultType = TypeVar("ResultType")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter_ratio: float = 0.2
    total_timeout: float | None = None
    retryable_kinds: frozenset[ErrorKind] = field(
        default_factory=lambda: frozenset(
            {
                ErrorKind.RATE_LIMITED,
                ErrorKind.NETWORK,
                ErrorKind.TIMEOUT,
                ErrorKind.PROVIDER_INTERNAL,
            }
        )
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("retry delays cannot be negative")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")
        if self.total_timeout is not None and self.total_timeout <= 0:
            raise ValueError("total_timeout must be positive")

    def should_retry(self, error: VvLlmError, attempt: int) -> bool:
        return attempt < self.max_attempts and error.kind in self.retryable_kinds

    def delay_for(self, error: VvLlmError, attempt: int, *, random_value: float | None = None) -> float:
        if error.retry_after is not None:
            return min(self.max_delay, error.retry_after)
        delay = min(self.max_delay, self.base_delay * (2 ** max(0, attempt - 1)))
        sample = random.random() if random_value is None else random_value
        jitter = delay * self.jitter_ratio * ((sample * 2) - 1)
        return max(0.0, delay + jitter)


def execute_with_retry(
    operation: Callable[[], ResultType],
    policy: RetryPolicy,
    *,
    provider: str | None = None,
    model: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> ResultType:
    started = monotonic()
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as exception:
            error = classify_exception(exception, provider=provider, model=model)
            if not policy.should_retry(error, attempt):
                raise error from exception
            delay = policy.delay_for(error, attempt)
            if policy.total_timeout is not None and monotonic() - started + delay >= policy.total_timeout:
                raise error from exception
            sleep(delay)
            attempt += 1


async def execute_with_retry_async(
    operation: Callable[[], Awaitable[ResultType]],
    policy: RetryPolicy,
    *,
    provider: str | None = None,
    model: str | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> ResultType:
    started = monotonic()
    attempt = 1
    while True:
        try:
            return await operation()
        except Exception as exception:
            error = classify_exception(exception, provider=provider, model=model)
            if not policy.should_retry(error, attempt):
                raise error from exception
            delay = policy.delay_for(error, attempt)
            if policy.total_timeout is not None and monotonic() - started + delay >= policy.total_timeout:
                raise error from exception
            await sleep(delay)
            attempt += 1


__all__ = [
    "RetryPolicy",
    "execute_with_retry",
    "execute_with_retry_async",
]
