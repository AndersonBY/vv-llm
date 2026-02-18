import json
import logging
from typing import Any
from abc import ABC, abstractmethod

from v_llm.chat_clients import create_chat_client
from v_llm.types import BackendType

logger = logging.getLogger(__name__)


class BaseGenie(ABC):
    """Abstract Base Class for Genies.

    This class defines the common interface for all Genie implementations. Genies
    are responsible for generating text or structured JSON responses based on
    input prompts.

    Subclasses must implement the `generate` method. If structured JSON output
    is required, subclasses should also implement the `generate_json` method.
    The batch generation methods (`generate_batch`, `generate_json_batch`) are
    provided for convenience and typically do not need to be overridden.

    Methods:
        generate(prompt: str) -> str:
            Generates a text response for a single prompt. Must be implemented by subclasses.
        generate_batch(prompts: list[str]) -> list[str]:
            Generates text responses for a batch of prompts. Uses `generate`.
        generate_json(prompt: str, schema: Any) -> Any:
            Generates a structured JSON response conforming to the provided schema
            for a single prompt. Should be implemented by subclasses if JSON output
            is needed.
        generate_json_batch(prompts: list[str], schema: Any) -> list[Any]:
            Generates structured JSON responses for a batch of prompts. Uses `generate_json`.

    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        raise NotImplementedError

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate a batch of responses based on the given prompts."""
        return [self.generate(prompt) for prompt in prompts]

    def generate_json(self, prompt: str, schema: Any) -> Any:
        """Generate a JSON response based on the given prompt and BaseModel schema."""
        raise NotImplementedError

    def generate_json_batch(self, prompts: list[str], schema: Any) -> list[Any]:
        """Generate a batch of JSON responses based on the given prompts and BaseModel schema."""
        return [self.generate_json(prompt, schema) for prompt in prompts]


class VectorVeinGenie(BaseGenie):
    """VectorVein Genie - 使用 vectorvein 作为 LLM 接口的 Genie 实现

    这个类可以在不修改 chonkie 源码的情况下使用，作为外部扩展。

    Examples:
        >>> from vectorvein_genie import VectorVeinGenie
        >>> from chonkie import SlumberChunker
        >>> from v_llm.types import BackendType
        >>>
        >>> # 使用 OpenAI
        >>> genie = VectorVeinGenie(backend_type=BackendType.OpenAI, model="gpt-4o")
        >>> chunker = SlumberChunker(genie=genie, chunk_size=512)
        >>> chunks = chunker.chunk("your long text...")
        >>>
        >>> # 使用 Anthropic
        >>> genie = VectorVeinGenie(
        ...     backend_type="Anthropic",
        ...     model="claude-3-sonnet-20240229"
        ... )
        >>> chunker = SlumberChunker(genie=genie)

    Attributes:
        backend_type (str): 后端类型
        model (str): 模型名称
        max_tokens (int): 最大生成 token 数
        temperature (float): 温度参数
        client: vectorvein 客户端实例
    """

    def __init__(
        self,
        backend_type: BackendType = BackendType.OpenAI,
        model: str = "gpt-5-mini",
        **kwargs,
    ):
        """初始化 VectorVeinGenie

        Args:
            backend_type: 后端类型，BackendType
            model: 模型名称
            **kwargs: 传递给 vectorvein 客户端的其他参数
        """
        super().__init__()

        # 保存参数
        self.model = model
        # 转换后端类型
        self.backend_type = backend_type

        self.client = create_chat_client(
            backend=self.backend_type,
            model=model,
            stream=False,
            **kwargs,  # 传递其他参数
        )

    def generate(self, prompt: str) -> str:
        """生成文本响应

        Args:
            prompt: 提示文本

        Returns:
            生成的文本响应
        """
        messages = [{"role": "user", "content": prompt}]

        response = self.client.create_completion(
            messages=messages,
            extra_body={"enable_thinking": False},
            skip_cutoff=True,
        )

        return response.content if response.content else ""

    def generate_json(self, prompt: str, schema: Any) -> dict[str, Any]:
        """生成结构化 JSON 响应

        Args:
            prompt: 提示文本
            schema: Pydantic BaseModel 类，定义期望的 JSON 结构

        Returns:
            符合 schema 的字典

        Raises:
            ValueError: 如果响应无法解析为有效的 JSON
        """
        # 构建系统提示
        system_prompt = (
            "You are a helpful assistant that returns responses in valid JSON format. "
            "Always respond with a valid JSON object that matches the requested schema. "
            "Do not include any explanatory text, only return the JSON object."
        )

        # 构建 schema 描述
        schema_description = self._build_schema_description(schema)
        full_prompt = f"{prompt}\n\n{schema_description}"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": full_prompt}]

        # 调用 LLM
        response = self.client.create_completion(
            messages=messages,
            extra_body={"enable_thinking": False},
            skip_cutoff=True,
        )

        # 解析响应
        content = response.content.strip() if response.content else ""

        # 尝试从 markdown 代码块中提取 JSON
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()

        # 解析 JSON
        try:
            result = json.loads(content)
            return dict(result)
        except json.JSONDecodeError as e:
            # 尝试查找第一个 JSON 对象
            try:
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                    return dict(result)
            except Exception:
                logger.debug("Fallback JSON extraction failed", exc_info=True)

            raise ValueError(f"Failed to parse LLM response as JSON.\nResponse (first 200 chars): {content[:200]}...\nError: {str(e)}") from e

    def _build_schema_description(self, schema: Any) -> str:
        """构建 Pydantic schema 的文本描述

        Args:
            schema: Pydantic BaseModel 类

        Returns:
            schema 的文本描述
        """
        # 支持 Pydantic v1 和 v2
        if hasattr(schema, "model_fields"):
            # Pydantic v2
            fields_info = schema.model_fields
        elif hasattr(schema, "__fields__"):
            # Pydantic v1
            fields_info = schema.__fields__
        else:
            return "Return a JSON object matching the expected format."

        description = "Return a JSON object with the following fields:\n"
        for field_name, field_info in fields_info.items():
            if hasattr(field_info, "annotation"):
                field_type = field_info.annotation
            else:
                field_type = "any"
            description += f"  - {field_name}: {field_type}\n"

        return description

    def __repr__(self) -> str:
        """返回字符串表示

        Returns:
            VectorVeinGenie 的字符串表示
        """
        return f"VectorVeinGenie(backend={self.backend_type}, model={self.model})"
