# vv-llm

[English README](./README.md)

面向多模型场景的统一 LLM 接口层。一套 API，17 种后端，同步 & 异步。

```
pip install vv-llm
```

## 支持的后端

OpenAI | Anthropic | DeepSeek | Gemini | Qwen | Groq | Mistral | Moonshot | MiniMax | Yi | ZhiPuAI | Baichuan | StepFun | xAI | Xiaomi | Ernie | Local

同时支持 Azure OpenAI、Vertex AI 和 AWS Bedrock 部署。

## 快速开始

### 加载配置

```python
from vv_llm.settings import settings

settings.load({
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

### 类型化同步调用（规范化请求）

```python
from vv_llm.chat_clients import create_chat_client, BackendType
from vv_llm import ChatRequest, ChatRequestOptions, ThinkingPreference

client = create_chat_client(BackendType.OpenAI, model="gpt-4o")
resp = client.create(
    ChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "用一句话解释 RAG"}],
        options=ChatRequestOptions(
            thinking=ThinkingPreference.default(),
            max_tokens=512,
        ),
    )
)
print(resp.content)
```

`ChatRequest` 是规范化的运行时请求。跨 contract 边界时，使用
`ChatRequest.from_contract(...)` 解码 canonical JSON（其中 `model` 必填且
`options.stream` 嵌套在 options 内），使用 `to_contract()` 编码；headers、query
等运行时传输控制不会进入 canonical JSON。

`ThinkingPreference.default()` 保留 provider 默认行为，`enabled()` 或
`enabled(budget_tokens=...)` 显式开启，`disabled()` 显式关闭。

### 关键字参数 API

`create_completion(...)` 接受关键字参数。支持 Anthropic 风格 thinking 控制的
provider 可显式传入 `thinking`；不传时使用 provider 默认值：

```python
resp = client.create_completion(
    messages=[{"role": "user", "content": "直接回答"}],
    thinking={"type": "disabled"},
)
```

### Middleware、重试与 Metadata

需要 middleware hook、分类重试或执行 metadata 时，用
`MiddlewareChatClient` 包装 client：

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
from vv_llm import ChatRequest

for chunk in client.create(ChatRequest(
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一首四行诗"}],
    stream=True,
)):
    if chunk.content:
        print(chunk.content, end="")
```

### 异步调用

```python
import asyncio
from vv_llm.chat_clients import create_async_chat_client, BackendType
from vv_llm import ChatRequest

async def main():
    client = create_async_chat_client(BackendType.OpenAI, model="gpt-4o")
    resp = await client.create(ChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    ))
    print(resp.content)

asyncio.run(main())
```

### HTTP 传输 client

`http_client` 参数同步调用接受 `httpx2.Client`，异步调用接受
`httpx2.AsyncClient`。应用可以注入自定义传输（例如离线
`MockTransport`），同时让 OpenAI 3.x 与 Anthropic 1.x 使用同一套 HTTPX2
运行时：

```python
import httpx2
from vv_llm.chat_clients import BackendType, create_chat_client

transport = httpx2.MockTransport(
    lambda request: httpx2.Response(200, json={"choices": []}, request=request)
)
http_client = httpx2.Client(transport=transport)
client = create_chat_client(BackendType.OpenAI, model="gpt-4o", http_client=http_client)
```

需要由 vv-llm 自动创建传输 client 时，在 endpoint 中配置 `proxy`。旧版
`httpx.Client` 与 `httpx.AsyncClient` 不再接受，请改用对应的 HTTPX2 类型。

### Embedding 与 Rerank

```python
from vv_llm.settings import settings

settings.load({
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

## 推理强度

模型能力的 `reasoning_efforts` 缺省或为 null 表示未知，空列表表示不支持，非空列表声明实际选项。请求省略 `reasoning_effort` 使用服务端默认值；`none` 是显式值。

切换模型请使用请求的 `model` 或端点绑定的 `model_id`。`extra_body.model`
与选中的实际模型冲突时，即使使用 passthrough 也会报错。

`create` 和旧 `create_completion` 都支持 `capability_policy=CapabilityPolicy.STRICT`，默认 WARN，PASSTHROUGH 跳过模型范围校验，参数冲突始终报错。Responses 映射为 `reasoning.effort`，Anthropic 映射为 `output_config.effort`。绑定中的 `capabilities` 局部覆盖模型能力；registry 的 `model_capabilities` 按模型声明 fallback 能力，不自动降档。

现有关键字调用和类型化调用可以继续使用，不传 effort 仍使用服务端默认值。以前传入
未确认或不支持的档位，现在默认会发出警告并继续请求；显式选用 STRICT 后，未知
能力和非法档位都会在发送前报错。同步、异步、普通响应和流式响应使用同一套规则。

在 settings 中绑定 DeepSeek Flash 端点后：

```python
from vv_llm import CapabilityPolicy, ChatRequest, ChatRequestOptions
from vv_llm.chat_clients import BackendType, create_chat_client

client = create_chat_client(BackendType.DeepSeek, model="deepseek-flash")
print(client.capabilities.reasoning_efforts)
print(client.capabilities.reasoning_effort_aliases)

response = client.create(
    ChatRequest(
        model=client.model,
        messages=[{"role": "user", "content": "计算 37 * 19。"}],
        options=ChatRequestOptions(reasoning_effort="xhigh", max_tokens=256),
    ),
    capability_policy=CapabilityPolicy.STRICT,
)
# 请求仍传 xhigh，由服务商按文档映射为 high。
```

原来的关键字参数 API 同样支持严格校验：

```python
response = client.create_completion(
    messages=[{"role": "user", "content": "计算 37 * 19。"}],
    reasoning_effort="high",
    capability_policy=CapabilityPolicy.STRICT,
    max_tokens=256,
)
```

模型选择器展示实际档位，兼容输入不作为额外档位展示；列表中的 none 单独表示
关闭。effort 和 thinking 是两个控制：省略 effort 不会自动开启或关闭 thinking，
使用 `ThinkingPreference` 控制开关。GLM-5.3/FLASH 的 xhigh 会被严格校验拒绝，
这两个型号也不支持关闭 thinking；GLM-5.2 支持显式关闭。已记录的 GLM-5.2 实测中，
none/minimal 仍返回推理内容，而显式 disabled 没有返回推理内容。

新增的[离线能力/校验/fallback 示例](examples/reasoning_capabilities.py)和
[真实请求/开关/流式示例](examples/reasoning_effort.py)见[示例指南](examples/README.md#reasoning-effort)。

`reasoning_effort_aliases` 单独记录兼容输入及其实际目标。只有目标仍在 `reasoning_efforts` 中时才接受别名，请求原值由服务商映射。绑定中的档位列表和别名映射均整体替换。DeepSeek 实际为 low/high/max 三档，none 表示关闭；minimal → low，medium/xhigh → high，ultra → max。兼容别名不作为独立档位展示。

## 核心特性

- **统一接口** — 所有后端共享规范化 `ChatRequest` 执行入口，同时兼容 `create_completion` / `create_stream`
- **Embedding 与 rerank** — 提供统一的同步/异步检索客户端与标准化输出
- **类型安全的工厂** — `create_chat_client(BackendType.X)` 返回对应的客户端类型
- **多端点管理** — 按优先级数值升序选择可用端点，同级保持配置顺序
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

模型端点绑定支持可选 `priority`，必须为大于等于 1 的整数，默认按 1 处理。
显式指定 `endpoint_id` 时使用指定端点。
`from vv_llm.settings import order_endpoints` 提供统一排序函数：
`order_endpoints(endpoints, preferred_endpoint_id=None)` 返回新列表，
偏好端点仅在同优先级内提前。

包内包含 `vv-llm-contract` 1.2.2。通过 `vv_llm.contract` 读取 contract
metadata、模型目录和完整性状态：

```python
from vv_llm.contract import contract_info, load_catalog, verify_contract

assert contract_info().contract_version == "1.2.2"
assert verify_contract().ok
catalog = load_catalog()
```

维护者可用 `pdm run contract-check` 校验包内副本，并用
`pdm run contract-sync --source PATH` 从已校验的 release 目录更新。

## Python 能力矩阵

| 能力面 | Python 支持 | 边界 |
|---|---|---|
| Middleware | `MiddlewareChatClient` / `AsyncMiddlewareChatClient`；v1 request/response/error hook 与 metadata | 需要显式包装 chat client |
| Fallback | `FallbackChatClient` / `AsyncFallbackChatClient`；按顺序、按 capability 路由 | 流式只在建立阶段或首个可见 chunk 前切换 |
| Retry | `RetryPolicy` 与同步/异步执行器；错误分类、`Retry-After`、退避、jitter、deadline | 默认不重试认证和无效请求错误 |
| 确定性测试 | Scripted client、vendored protocol fixture、unit tests | 不访问网络；在线检查必须显式 opt-in |
| Chat provider | Anthropic 原生 adapter；15 个 OpenAI-compatible adapter；Local adapter | 同步/异步和流式统一；tool、结构化输出、多模态、thinking 由模型/provider 决定 |
| Embedding | 同步/异步配置型 client | 支持 OpenAI embeddings、SiliconFlow、Cohere、Voyage、自定义 JSON HTTP protocol |
| Rerank | 同步/异步配置型 client | 支持 OpenAI-compatible、Cohere、Jina、Voyage、SiliconFlow、自定义 JSON HTTP protocol |

### Chat Provider 矩阵

| Adapter | Provider | 传输与共同行为 |
|---|---|---|
| Native | Anthropic | 原生同步/异步 chat、stream、tools、vision、thinking、prompt cache 处理 |
| OpenAI-compatible | OpenAI、DeepSeek、Gemini、Groq、MiniMax、Mistral、Moonshot、Qwen、Yi、ZhiPuAI、Baichuan、StepFun、xAI、Xiaomi、Ernie | 共用同步/异步请求与 stream 标准化；tools、结构化输出、多模态、reasoning 以 vendored model catalog 和 provider endpoint 为准 |
| Local | Local | 同样的同步/异步与 stream adapter 形状；具体能力由部署 endpoint 决定 |

## 使用示例

可运行示例位于 [`examples/`](examples/README.md)：`basic_chat.py`、
`streaming.py`、`tools.py`、`multimodal.py` 和 `contract_json.py` 覆盖主要的
类型化请求路径；`async_streaming.py`、`typed_thinking.py`、
`middleware_metadata.py` 与 `registry_fallback.py` 覆盖扩展能力，其中最后一个
完全离线且使用确定性 scripted client。新增的 `reasoning_capabilities.py` 也完全
离线；`reasoning_effort.py` 使用本地配置发送一次请求，并支持流式输出。

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
  _contract/      # 版本化 schema、fixture、catalog 与 consumer lock
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

## 用户、维护者与发布流程

### 用户

安装包后通过公开 `Settings` API 配置 endpoint。contract artifact 从已安装包
内资源读取，运行时不要求额外源码来源。

### 维护者

```bash
pdm install -d          # 安装开发依赖
pdm run contract-check  # 只校验包内 vendored contract lock
pdm run contract-sync --source PATH  # 从显式 source tree 同步
# 或：VV_LLM_CONTRACT_SOURCE=PATH pdm run contract-sync
# 比较显式 source 与包内 vendor：
python scripts/sync_contract.py --check --source PATH
pdm run lint            # Ruff 检查
pdm run format-check    # Ruff 格式检查
pdm run type-check      # Ty 类型检查
pdm run test            # 单元测试
```

如需有意执行在线 smoke，可通过现有 `tests/dev_settings.py` 机制提供私有配置，
并显式 opt-in：

```bash
VV_LLM_RUN_LIVE_TESTS=1 python tests/live/run_live_tests.py test_deepseek_contract_smoke.py
```

smoke 只输出 provider/model、响应形状、usage 计数和退出状态，不输出凭据或响应正文。

#### 推理强度在线检查

显式选择私有配置进行推理参数在线烟测：

```bash
python tests/live/reasoning_effort_smoke.py --settings /secure/path/llm_settings.json \
  --backend deepseek --aliases --invalid-probe --limit 18
```

按有凭据的模型/协议路由逐档调用，默认最多 80 个请求。`--aliases` 包含兼容输入，`--invalid-probe` 探测非法值，`--model backend:model` 筛选模型，`--report` 保存脱敏结果。`--include-catalog` 才会包含本地未列出、通过默认端点绑定的目录模型。返回成功只能证明请求被接受；SDK timeout 不代表整个流程的墙钟时限。

[已记录的在线结果](tests/live/reasoning-effort-report.md) 区分接受、拒绝、未验证和网络/配置失败。

### 发布者

```bash
pdm build
python scripts/smoke_wheel.py
```

发布 CI 会执行 contract-check、单元测试、lint、构建和隔离 wheel smoke 后才
发布。在线 API 检查不属于发布 CI。

## 许可证

MIT


### Gemini 生成参数

Gemini 3 及后续模型的请求会省略 `temperature`、`top_p`、`top_k` 和旧的
`thinking_budget`，包括驼峰拼写与嵌套 provider 参数。只设置 budget 时使用模型默认值，
不会猜测数值到档位的映射。需要指定思考强度时使用 `reasoning_effort` 或
Google 的 `thinking_config.thinking_level`，不要同时设置两者。3.7 Flash 和 3.8 Flash
支持 low/medium/high，不支持 minimal；Gemini 2.5 保留原有预算与采样行为。
