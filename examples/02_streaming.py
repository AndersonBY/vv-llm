from vv_llm import ChatRequest, ChatRequestOptions, ThinkingPreference

from common import load_deepseek_client


client = load_deepseek_client()
stream = client.create(
    ChatRequest(
        messages=[{"role": "user", "content": "Explain retry backoff briefly."}],
        stream=True,
        options=ChatRequestOptions(
            thinking=ThinkingPreference.disabled(),
            max_tokens=256,
        ),
    )
)

for chunk in stream:
    if chunk.reasoning_content:
        print(chunk.reasoning_content, end="", flush=True)
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()
