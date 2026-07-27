from __future__ import annotations

from vv_llm import (
    ChatRequest,
    ChatRequestOptions,
    MiddlewareChatClient,
    RetryPolicy,
    ThinkingPreference,
)
from vv_llm.chat_clients import BackendType, create_chat_client
from vv_llm.settings import settings

from tests.sample_settings import sample_settings


settings.load(sample_settings)


def test_deepseek_execution_pipeline_thinking_and_metadata() -> None:
    client = create_chat_client(BackendType.DeepSeek, model="deepseek-v4-flash")
    runtime = MiddlewareChatClient(
        client,
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.25),
    )

    for preference, expect_reasoning in [
        (ThinkingPreference.default(), True),
        (ThinkingPreference.enabled(), True),
        (ThinkingPreference.disabled(), False),
    ]:
        result = runtime.create_with_metadata(
            ChatRequest(
                messages=[
                    {
                        "role": "user",
                        "content": "Compute 37 * 19 and answer briefly.",
                    }
                ],
                options=ChatRequestOptions(
                    thinking=preference,
                    max_tokens=128,
                ),
            )
        )
        assert result.response.content
        assert bool(result.response.reasoning_content) is expect_reasoning
        assert result.metadata.provider == "deepseek"
        assert result.metadata.model == "deepseek-v4-flash"
        assert result.metadata.attempts >= 1
        assert result.metadata.latency_ms is not None

    legacy = client.create_completion(
        messages=[{"role": "user", "content": "Reply with exactly OK."}],
        thinking={"type": "disabled"},
        max_tokens=64,
    )
    assert legacy.content
    assert not legacy.reasoning_content
