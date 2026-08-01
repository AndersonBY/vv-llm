# vv-llm

[English README](./README.md)

面向多模型场景的统一 LLM 接口层。一套 API，16 种后端，同步 & 异步。

```
pip install vv-llm
```

## 支持的后端

OpenAI | Anthropic | DeepSeek | Gemini | Qwen | Groq | Mistral | Moonshot | MiniMax | Yi | ZhiPuAI | Baichuan | StepFun | xAI | Ernie | Local

同时支持 Azure OpenAI、Vertex AI 和 AWS Bedrock 部署。

## 快速开始

### 加载配置

```python
from vv_llm.settings import settings

settings.load({
    "VERSION": "2",
    "endpoints": [
        {
            "id": "openai-default",
            "api_base": "https://api.openai.com/v1",
            "api_key": "sk-...",
        }
    ],
    "backends": {
        "openai": {
            "models": {
                "gpt-4o": {
                    "id": "gpt-4o",
                    "endpoints": ["openai-default"],
                }
            }
        }
    }
})
```

### 同步调用

```python
from vv_llm.chat_clients import create_chat_client, BackendType

client = create_chat_client(BackendType.OpenAI, model="gpt-4o")
resp = client.create_completion([
    {"role": "user", "content": "用一句话解释 RAG"}
])
print(resp.content)
```

支持 Anthropic 风格 thinking 控制的 provider 可以显式传入 `thinking`；不传时继续使用 provider 默认值：

```python
resp = client.create_completion(
    messages=[{"role": "user", "content": "直接回答"}],
    thinking={"type": "disabled"},
)
```

旧的关键字参数 API 会继续兼容。新代码可以使用规范化请求和类型化 thinking 控制：

```python
from vv_llm import ChatRequest, ChatRequestOptions, ThinkingPreference

resp = client.create(
    ChatRequest(
        messages=[{"role": "user", "content": "直接回答"}],
        options=ChatRequestOptions(
            thinking=ThinkingPreference.disabled(),
            max_tokens=512,
        ),
    )
)

print(client.capabilities.thinking)
```

`ThinkingPreference.default()` 保留 provider 默认行为，`enabled()` 或
`enabled(budget_tokens=...)` 显式开启，`disabled()` 显式关闭。

### Middleware、重试与 Metadata

原有 client 行为不变。只有在应用需要版本化 middleware、分类重试或执行 metadata
时才进行包装：

```python
from vv_llm import ChatMiddlewareV1, ChatRequest, MiddlewareChatClient, RetryPolicy

class TraceMiddleware(ChatMiddlewareV1):
    def on_request(self, context, request):
        context.attributes["trace_id"] = "request-42"
        return request

runtime = MiddlewareChatClient(
    client,
    [TraceMiddleware()],
    retry_policy=RetryPolicy(max_attempts=3, total_timeout=20),
)
result = runtime.create_with_metadata(
    ChatRequest(messages=[{"role": "user", "content": "直接回答"}])
)

print(result.response.content)
print(result.metadata.provider, result.metadata.attempts, result.metadata.latency_ms)
```

`ErrorKind` 统一区分认证、限流、网络、超时、无效请求、上下文长度、内容策略、
模型不存在、provider 内部错误、序列化和配置错误。默认策略只重试瞬时错误，
并支持优先级更高的 `retry-after-ms`、秒数或 HTTP-date 格式的 `Retry-After`、
指数退避、抖动和可选的总 deadline。

### 显式 Registry 与 Fallback

Fallback 必须显式启用并声明顺序。每个注册项都声明模型能力，不兼容的 route
会在发请求之前跳过：

```python
from vv_llm import FallbackChatClient, FallbackRoute, ProviderRegistry

registry = ProviderRegistry()
registry.register(
    "primary",
    lambda: primary_client,
    capabilities=primary_client.capabilities,
)
registry.register(
    "secondary",
    lambda: secondary_client,
    capabilities=secondary_client.capabilities,
)
runtime = FallbackChatClient(
    registry,
    [
        FallbackRoute("primary", "primary-model"),
        FallbackRoute("secondary", "secondary-model"),
    ],
)
```

默认不会对认证和无效请求错误执行 fallback。流式调用只能在建立 stream 或首个
可见 chunk 之前切换 route；一旦已有输出，后续错误会直接返回，不会重放请求。

### 流式调用

```python
for chunk in client.create_stream([
    {"role": "user", "content": "写一首四行诗"}
]):
    if chunk.content:
        print(chunk.content, end="")
```

### 异步调用

```python
import asyncio
from vv_llm.chat_clients import create_async_chat_client, BackendType

async def main():
    client = create_async_chat_client(BackendType.OpenAI, model="gpt-4o")
    resp = await client.create_completion([
        {"role": "user", "content": "hello"}
    ])
    print(resp.content)

asyncio.run(main())
```

### Embedding 与 Rerank

```python
from vv_llm.settings import settings

settings.load({
    "VERSION": "2",
    "endpoints": [
        {
            "id": "siliconflow",
            "api_base": "https://api.siliconflow.cn/v1",
            "api_key": "sk-...",
        }
    ],
    "backends": {},
    "embedding_backends": {
        "siliconflow": {
            "models": {
                "BAAI/bge-large-zh-v1.5": {
                    "id": "BAAI/bge-large-zh-v1.5",
                    "endpoints": ["siliconflow"],
                    "protocol": "openai_embeddings",
                }
            }
        }
    },
    "rerank_backends": {
        "siliconflow": {
            "models": {
                "BAAI/bge-reranker-v2-m3": {
                    "id": "BAAI/bge-reranker-v2-m3",
                    "endpoints": ["siliconflow"],
                    "protocol": "custom_json_http",
                    "request_mapping": {
                        "method": "POST",
                        "path": "/rerank",
                        "body_template": {
                            "model": "${model_id}",
                            "query": "${query}",
                            "documents": "${documents}",
                        },
                    },
                    "response_mapping": {
                        "results_path": "$.results[*]",
                        "field_map": {
                            "index": "$.index",
                            "relevance_score": "$.relevance_score",
                        },
                    },
                }
            }
        }
    },
})
```

```python
from vv_llm.embedding_clients import create_embedding_client
from vv_llm.rerank_clients import create_rerank_client

embedding_client = create_embedding_client("siliconflow", model="BAAI/bge-large-zh-v1.5")
embedding_resp = embedding_client.create_embeddings(input="hello world")
print(len(embedding_resp.data[0].embedding))

rerank_client = create_rerank_client("siliconflow", model="BAAI/bge-reranker-v2-m3")
rerank_resp = rerank_client.rerank(
    query="Apple",
    documents=["apple", "banana", "fruit", "vegetable"],
)
print(rerank_resp.results[0].index, rerank_resp.results[0].relevance_score)
```

```python
import asyncio
from vv_llm.embedding_clients import create_async_embedding_client
from vv_llm.rerank_clients import create_async_rerank_client

async def main():
    embedding_client = create_async_embedding_client("siliconflow", model="BAAI/bge-large-zh-v1.5")
    rerank_client = create_async_rerank_client("siliconflow", model="BAAI/bge-reranker-v2-m3")

    emb = await embedding_client.create_embeddings(input=["a", "b"])
    rr = await rerank_client.rerank(query="Apple", documents=["apple", "banana"])
    print(len(emb.data), len(rr.results))

asyncio.run(main())
```

## 核心特性

- **统一接口** — 所有后端共享相同的 `create_completion` / `create_stream` API
- **Embedding 与 rerank** — 提供统一的同步/异步检索客户端与标准化输出
- **类型安全的工厂** — `create_chat_client(BackendType.X)` 返回对应的客户端类型
- **多端点管理** — 每个后端可配置多个端点，支持随机选择和故障转移
- **工具调用** — 跨后端标准化的 tool/function calling
- **多模态** — 支持文本 + 图片输入
- **思维链/推理** — 获取 Claude、DeepSeek Reasoner 等模型的推理过程
- **Token 统计** — 按模型使用对应分词器（tiktoken、deepseek-tokenizer、qwen-tokenizer）
- **速率限制** — RPM/TPM 控制，支持 memory、Redis、DiskCache 后端
- **上下文长度控制** — 自动截断消息以适配模型限制
- **Prompt 缓存** — 支持 Anthropic prompt caching
- **重试与退避** — 可配置的重试逻辑
- **版本化 middleware** — provider adapter 外稳定的 `v1` 请求、响应和错误 hook
- **统一错误分类** — 带重试语义和请求上下文的 provider-neutral 错误
- **显式 fallback** — 只按注册顺序执行 capability-aware route，不隐式切换 provider
- **Scripted 测试** — 用确定性的响应、错误和 stream 脚本进行契约测试

## 使用示例

[`examples/`](examples/README.md) 提供类型化 thinking、同步/异步流式输出、
middleware metadata 和显式 fallback 的可运行示例。

## 缓存 Usage 语义

OpenAI-compatible chat completion 通过 `usage.prompt_tokens_details.cached_tokens` 表示缓存读取量。`usage.prompt_tokens` 始终是总输入 token 数，因此调用方可用 `prompt_tokens - cached_tokens` 计算未缓存输入。该路径不会填充 Anthropic 的 `cache_read_input_tokens` 字段，因为 Anthropic 将其基础 `input_tokens` 定义为未缓存输入，两者口径不同。

对通用 OpenAI-compatible 后端，缓存读取字段省略时仍保持未知，显式的 `cached_tokens: 0` 则保留为观测零。Moonshot 的冷请求可能同时省略顶层 `cached_tokens` 和 `prompt_tokens_details`；仅在两者都完全省略时，vv-llm 才依据 provider 契约投影 `prompt_tokens_details.cached_tokens = 0`。显式 `null` 或无效缓存值继续保持未知。

## 工具函数

```python
from vv_llm.chat_clients import format_messages, get_token_counts, get_message_token_counts
```

| 函数 | 说明 |
|---|---|
| `format_messages` | 多模态/工具消息格式标准化 |
| `get_token_counts` | 文本 token 统计 |
| `get_message_token_counts` | 消息级 token 统计 |

## 可选依赖

```bash
pip install 'vv-llm[redis]'      # Redis 限流后端
pip install 'vv-llm[diskcache]'  # DiskCache 限流后端
pip install 'vv-llm[server]'     # FastAPI token server
pip install 'vv-llm[vertex]'     # Google Vertex AI
pip install 'vv-llm[bedrock]'    # AWS Bedrock
```

## 目录结构

```
src/vv_llm/
  chat_clients/    # 各后端 client + 工厂
  embedding_clients/  # embedding client + 工厂
  rerank_clients/     # rerank client + 工厂
  retrieval_clients/  # retrieval 共享底层能力
  settings/        # 配置管理
  types/           # 类型定义与枚举
  utilities/       # 限流、重试、多媒体处理、token 统计
  server/          # 可选的 token 统计服务

tests/unit/        # 单元测试
tests/live/        # 在线连通测试（需要真实 API key）
```

## 开发

```bash
pdm install -d          # 安装开发依赖
pdm run lint            # Ruff 检查
pdm run format-check    # Ruff 格式检查
pdm run type-check      # Ty 类型检查
pdm run test            # 单元测试
pdm run test-live       # 在线测试（需要真实端点）
```

## 许可证

MIT
