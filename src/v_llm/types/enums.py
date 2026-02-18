# @Author: Bi Ying
# @Date:   2024-07-26 23:52:52
from __future__ import annotations

from enum import Enum


class BackendType(str, Enum):
    """BackendType enum class definition."""

    # OpenAI
    OpenAI = "openai"

    # ZhiPuAI
    ZhiPuAI = "zhipuai"

    # MiniMax
    MiniMax = "minimax"

    # Moonshot
    Moonshot = "moonshot"

    # Anthropic
    Anthropic = "anthropic"

    # Mistral
    Mistral = "mistral"

    # DeepSeek
    DeepSeek = "deepseek"

    # Aliyun Qwen
    Qwen = "qwen"

    # Groq
    Groq = "groq"

    # Local
    Local = "local"

    # Yi
    Yi = "yi"

    # Gemini
    Gemini = "gemini"

    # Baichuan
    Baichuan = "baichuan"

    # StepFun
    StepFun = "stepfun"

    # XAI
    XAI = "xai"

    # Baidu Ernie
    Ernie = "ernie"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


class LLMType(str, Enum):
    """LLMType enum class definition."""

    # Embeddings
    OpenAIEmbedding = "openai_embedding"
    AzureOpenAIEmbedding = "azure_openai_embedding"

    # Raw Completion
    OpenAI = "openai"
    AzureOpenAI = "azure_openai"

    # Chat Completion
    OpenAIChat = "openai_chat"
    AzureOpenAIChat = "azure_openai_chat"

    # Debug
    StaticResponse = "static_response"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


class ContextLengthControlType(str, Enum):
    """ContextLengthControlType enum class definition."""

    # latest
    Latest = "latest"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'
