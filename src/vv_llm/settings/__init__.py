# @Author: Bi Ying
# @Date:   2024-07-27 00:30:56
import warnings
from copy import deepcopy
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from ..types import defaults as defs
from ..types.enums import BackendType
from ..types.settings import SettingsDict
from ..types.llm_parameters import BackendSettings, EndpointSetting


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
    ernie: BackendSettings = Field(default_factory=BackendSettings, description="Baidu Ernie models settings.")


class Settings(BaseModel):
    VERSION: str | None = Field(default="2", description="Configuration version. If provided, will use the corresponding format.")
    endpoints: list[EndpointSetting] = Field(default_factory=list, description="Available endpoints for the LLM service.")
    token_server: Server | None = Field(default=None, description="Token server address. Format: host:port")
    rate_limit: RateLimitConfig | None = Field(default=None, description="Rate limit settings.")

    # V2 format: all model backend configs in a single dictionary
    backends: Backends | None = Field(default=None, description="All model backends in one place (V2 format).")

    # V1 format: each model backend config
    anthropic: BackendSettings | None = Field(default_factory=BackendSettings, description="Anthropic models settings.")
    deepseek: BackendSettings | None = Field(default_factory=BackendSettings, description="Deepseek models settings.")
    gemini: BackendSettings | None = Field(default_factory=BackendSettings, description="Gemini models settings.")
    groq: BackendSettings | None = Field(default_factory=BackendSettings, description="Groq models settings.")
    local: BackendSettings | None = Field(default_factory=BackendSettings, description="Local models settings.")
    minimax: BackendSettings | None = Field(default_factory=BackendSettings, description="Minimax models settings.")
    mistral: BackendSettings | None = Field(default_factory=BackendSettings, description="Mistral models settings.")
    moonshot: BackendSettings | None = Field(default_factory=BackendSettings, description="Moonshot models settings.")
    openai: BackendSettings | None = Field(default_factory=BackendSettings, description="OpenAI models settings.")
    qwen: BackendSettings | None = Field(default_factory=BackendSettings, description="Qwen models settings.")
    yi: BackendSettings | None = Field(default_factory=BackendSettings, description="Yi models settings.")
    zhipuai: BackendSettings | None = Field(default_factory=BackendSettings, description="Zhipuai models settings.")
    baichuan: BackendSettings | None = Field(default_factory=BackendSettings, description="Baichuan models settings.")
    stepfun: BackendSettings | None = Field(default_factory=BackendSettings, description="StepFun models settings.")
    xai: BackendSettings | None = Field(default_factory=BackendSettings, description="XAI models settings.")
    ernie: BackendSettings | None = Field(default_factory=BackendSettings, description="Baidu Ernie models settings.")

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
            "ernie": defs.ERNIE_MODELS,
        }

        data = deepcopy(data)

        version = data.get("VERSION")

        if len(data) == 0:
            version = "2"
            data["backends"] = {}

        # If V2 format, model configs are in the backends dictionary
        if version == "2":
            if "backends" not in data:
                raise ValueError("backends is required in V2 format.")

            backends = cast(dict[str, Any], data["backends"])
        else:
            backends = cast(dict[str, Any], data)
            warnings.warn("You're using vectorvein's deprecated V1 format. Please use V2 format.", stacklevel=2)

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

        for endpoint in data.get("endpoints", []):
            if endpoint.get("is_azure"):
                endpoint["endpoint_type"] = "openai_azure"
            if endpoint.get("is_vertex"):
                endpoint["endpoint_type"] = "anthropic_vertex"
            if endpoint.get("is_bedrock"):
                endpoint["endpoint_type"] = "anthropic_bedrock"
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

        # Use VERSION 2 format backends field first
        if self.VERSION == "2" and self.backends is not None:
            return getattr(self.backends, backend_name)

        # Compatible with VERSION 1 format
        return getattr(self, backend_name)

    def export(self):
        return cast(
            SettingsDict,
            super().model_dump(
                exclude={
                    "anthropic",
                    "deepseek",
                    "gemini",
                    "groq",
                    "local",
                    "minimax",
                    "mistral",
                    "moonshot",
                    "openai",
                    "qwen",
                    "yi",
                    "zhipuai",
                    "baichuan",
                    "stepfun",
                    "xai",
                    "ernie",
                },
            ),
        )

    def upgrade_to_v2(self) -> "Settings":
        """
        Upgrade settings from v1 format to v2 format.
        In v2 format, all backend settings are stored in the 'backends' field.

        Returns:
            Settings: Self with updated format
        """
        # If already v2, no need to upgrade
        if self.VERSION == "2" and self.backends is not None:
            return self

        # Initialize backends if not exists
        if self.backends is None:
            self.backends = Backends()

        # Move all backend settings to backends field
        backend_names = [
            "anthropic",
            "deepseek",
            "gemini",
            "groq",
            "local",
            "minimax",
            "mistral",
            "moonshot",
            "openai",
            "qwen",
            "yi",
            "zhipuai",
            "baichuan",
            "stepfun",
            "xai",
            "ernie",
        ]

        for backend_name in backend_names:
            backend_setting = getattr(self, backend_name)
            if backend_setting is not None:
                setattr(self.backends, backend_name, backend_setting)
                delattr(self, backend_name)

        # Set version to 2
        self.VERSION = "2"

        return self


settings = Settings()


def normalize_settings(settings_input: "Settings | SettingsDict | None") -> Settings:
    """Normalize settings input so callers always operate on a Settings object."""
    if settings_input is None:
        return settings
    if isinstance(settings_input, Settings):
        return settings_input
    return Settings(**settings_input)
