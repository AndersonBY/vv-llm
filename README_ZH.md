# vv-llm

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

## 核心特性

- **统一接口** — 所有后端共享相同的 `create_completion` / `create_stream` API
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
