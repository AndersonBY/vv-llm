# @Author: Bi Ying
# @Date:   2024-07-27 00:30:56
from copy import deepcopy
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from ..types import defaults as defs
from ..types.enums import BackendType, EmbeddingBackendType, RerankBackendType
from ..types.settings import SettingsDict
from ..types.llm_parameters import BackendSettings, EndpointSetting, RetrievalBackendSettings


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0


class DiskCacheConfig(BaseModel):
    cache_dir: str = ".rate_limit_cache"


class RateLimitConfig(BaseModel):
    enabled: bool = False

    backend: Literal["memory", "redis", "diskcache"] = "memory"
    redis: RedisConfig | None = Field(default=None)
    diskcache: DiskCacheConfig | None = Field(default=None)
    default_rpm: int = 60
    default_tpm: int = 1000000


class Server(BaseModel):
    host: str
    port: int
    url: str | None


class Backends(BaseModel):
    """Model containing all backend configurations in one place."""

    anthropic: BackendSettings = Field(default_factory=BackendSettings, description="Anthropic models settings.")
    deepseek: BackendSettings = Field(default_factory=BackendSettings, description="Deepseek models settings.")
    gemini: BackendSettings = Field(default_factory=BackendSettings, description="Gemini models settings.")
    groq: BackendSettings = Field(default_factory=BackendSettings, description="Groq models settings.")
    local: BackendSettings = Field(default_factory=BackendSettings, description="Local models settings.")
    minimax: BackendSettings = Field(default_factory=BackendSettings, description="Minimax models settings.")
    mistral: BackendSettings = Field(default_factory=BackendSettings, description="Mistral models settings.")
    moonshot: BackendSettings = Field(default_factory=BackendSettings, description="Moonshot models settings.")
    openai: BackendSettings = Field(default_factory=BackendSettings, description="OpenAI models settings.")
    qwen: BackendSettings = Field(default_factory=BackendSettings, description="Qwen models settings.")
    yi: BackendSettings = Field(default_factory=BackendSettings, description="Yi models settings.")
    zhipuai: BackendSettings = Field(default_factory=BackendSettings, description="Zhipuai models settings.")
    baichuan: BackendSettings = Field(default_factory=BackendSettings, description="Baichuan models settings.")
    stepfun: BackendSettings = Field(default_factory=BackendSettings, description="StepFun models settings.")
    xai: BackendSettings = Field(default_factory=BackendSettings, description="XAI models settings.")
    xiaomi: BackendSettings = Field(default_factory=BackendSettings, description="Xiaomi models settings.")
    ernie: BackendSettings = Field(default_factory=BackendSettings, description="Baidu Ernie models settings.")


def _normalize_endpoint_transport_flags(endpoint: dict[str, Any]) -> None:
    endpoint_type = str(endpoint.get("endpoint_type", "") or "").strip().lower()
    if endpoint_type in ("", "default"):
        if endpoint.get("is_azure"):
            endpoint_type = "openai_azure"
        elif endpoint.get("is_vertex"):
            endpoint_type = "anthropic_vertex"
        elif endpoint.get("is_bedrock"):
            endpoint_type = "anthropic_bedrock"

    if endpoint_type:
        endpoint["endpoint_type"] = endpoint_type

    endpoint["is_azure"] = endpoint_type == "openai_azure"
    endpoint["is_vertex"] = endpoint_type == "anthropic_vertex"
    endpoint["is_bedrock"] = endpoint_type == "anthropic_bedrock"


class Settings(BaseModel):
    VERSION: str | None = Field(default="2", description="Optional wire-format metadata.")
    endpoints: list[EndpointSetting] = Field(default_factory=list, description="Available endpoints for the LLM service.")
    token_server: Server | None = Field(default=None, description="Token server address. Format: host:port")
    rate_limit: RateLimitConfig | None = Field(default=None, description="Rate limit settings.")

    backends: Backends = Field(default_factory=Backends, description="All model backends in one place.")
    embedding_backends: dict[str, RetrievalBackendSettings] | None = Field(
        default=None,
        description="Embedding backend settings.",
    )
    rerank_backends: dict[str, RetrievalBackendSettings] | None = Field(
        default=None,
        description="Rerank backend settings.",
    )

    def __init__(self, **data):
        model_types = {
            "anthropic": defs.ANTHROPIC_MODELS,
            "deepseek": defs.DEEPSEEK_MODELS,
            "gemini": defs.GEMINI_MODELS,
            "groq": defs.GROQ_MODELS,
            "local": {},
            "minimax": defs.MINIMAX_MODELS,
            "mistral": defs.MISTRAL_MODELS,
            "moonshot": defs.MOONSHOT_MODELS,
            "openai": defs.OPENAI_MODELS,
            "qwen": defs.QWEN_MODELS,
            "yi": defs.YI_MODELS,
            "zhipuai": defs.ZHIPUAI_MODELS,
            "baichuan": defs.BAICHUAN_MODELS,
            "stepfun": defs.STEPFUN_MODELS,
            "xai": defs.XAI_MODELS,
            "xiaomi": defs.XIAOMI_MODELS,
            "ernie": defs.ERNIE_MODELS,
        }

        data = deepcopy(data)
        legacy_backend = next((name for name in model_types if name in data), None)
        if legacy_backend is not None:
            raise ValueError(f"Top-level provider setting '{legacy_backend}' is unsupported; use 'backends.{legacy_backend}'.")

        raw_backends = data.get("backends")
        if isinstance(raw_backends, Backends):
            backends = cast(dict[str, Any], raw_backends.model_dump())
        elif raw_backends is None:
            backends = {}
        else:
            backends = cast(dict[str, Any], raw_backends)

        for model_type, default_models in model_types.items():
            if model_type in backends:
                user_models = cast(dict[str, dict[str, Any]], backends[model_type].get("models", {}))
                model_settings = BackendSettings()
                model_settings.update_models(cast(dict[str, dict[str, Any]], default_models), user_models)
                default_endpoint = backends[model_type].get("default_endpoint", None)
                if default_endpoint is not None:
                    model_settings.default_endpoint = default_endpoint
                    for model_setting in model_settings.models.values():
                        if len(model_setting.endpoints) == 0:
                            model_setting.endpoints = [default_endpoint]
                backends[model_type] = model_settings
            else:
                backends[model_type] = BackendSettings(models=cast(dict[str, Any], default_models))

        data["backends"] = backends

        for endpoint in data.get("endpoints", []):
            _normalize_endpoint_transport_flags(endpoint)
            if not endpoint.get("api_base"):
                continue
            api_base = endpoint["api_base"]
            if api_base.startswith("https://generativelanguage.googleapis.com/v1beta"):
                if not api_base.endswith("openai/"):
                    endpoint["api_base"] = api_base.strip("/") + "/openai/"

        super().__init__(**data)
        self._endpoint_index: dict[str, EndpointSetting] = {ep.id: ep for ep in self.endpoints}

    def load(self, settings: "SettingsDict | Settings"):
        if isinstance(settings, Settings):
            settings_dict = settings.export()
        else:
            settings_dict = settings
        self.__init__(**settings_dict)

    @classmethod
    def load_from_dict(cls, settings_dict: SettingsDict):
        return cls(**settings_dict)

    def get_endpoint(self, endpoint_id: str) -> EndpointSetting:
        endpoint = self._endpoint_index.get(endpoint_id)
        if endpoint is not None:
            return endpoint
        raise ValueError(f"Endpoint {endpoint_id} not found.")

    def get_backend(self, backend: BackendType) -> BackendSettings:
        backend_name = backend.value.lower()
        return getattr(self.backends, backend_name)

    def get_embedding_backend(self, backend: EmbeddingBackendType | str) -> RetrievalBackendSettings:
        backend_name = backend.value.lower() if isinstance(backend, EmbeddingBackendType) else str(backend).lower()
        if self.embedding_backends is None:
            raise ValueError("embedding_backends is not configured.")
        if backend_name not in self.embedding_backends:
            raise ValueError(f"Embedding backend {backend_name} not found.")
        return self.embedding_backends[backend_name]

    def get_rerank_backend(self, backend: RerankBackendType | str) -> RetrievalBackendSettings:
        backend_name = backend.value.lower() if isinstance(backend, RerankBackendType) else str(backend).lower()
        if self.rerank_backends is None:
            raise ValueError("rerank_backends is not configured.")
        if backend_name not in self.rerank_backends:
            raise ValueError(f"Rerank backend {backend_name} not found.")
        return self.rerank_backends[backend_name]

    def export(self):
        return cast(
            SettingsDict,
            super().model_dump(),
        )


settings = Settings()


def normalize_settings(settings_input: "Settings | SettingsDict | None") -> Settings:
    """Normalize settings input so callers always operate on a Settings object."""
    if settings_input is None:
        return settings
    if isinstance(settings_input, Settings):
        return settings_input
    return Settings(**settings_input)
