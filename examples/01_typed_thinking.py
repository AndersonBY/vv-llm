from vv_llm import ChatRequest, ChatRequestOptions, ThinkingPreference

from common import load_deepseek_client


client = load_deepseek_client()
request = ChatRequest(
    messages=[{"role": "user", "content": "Compute 37 * 19 and answer briefly."}],
    options=ChatRequestOptions(
        thinking=ThinkingPreference.enabled(),
        max_tokens=256,
    ),
)
response = client.create(request)

print("reasoning:", response.reasoning_content)
print("answer:", response.content)
