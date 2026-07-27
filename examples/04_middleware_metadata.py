from vv_llm import (
    ChatMiddlewareV1,
    ChatRequest,
    MiddlewareChatClient,
    RetryPolicy,
)

from common import load_deepseek_client


class TraceMiddleware(ChatMiddlewareV1):
    def on_request(self, context, request):
        context.attributes["trace_id"] = "example-request"
        return request


runtime = MiddlewareChatClient(
    load_deepseek_client(),
    [TraceMiddleware()],
    retry_policy=RetryPolicy(max_attempts=3, total_timeout=30),
)
result = runtime.create_with_metadata(
    ChatRequest(messages=[{"role": "user", "content": "Reply with exactly OK."}])
)

print(result.response.content)
print(result.metadata.model_dump())
