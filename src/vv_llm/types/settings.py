from typing import Literal
from typing_extensions import TypedDict, NotRequired  # Required by pydantic under Python < 3.12


class RedisConfigDict(TypedDict):
    """TypedDict representing the RedisConfig structure."""

    host: str
    port: int
    db: int


class DiskCacheConfigDict(TypedDict):
    """TypedDict representing the DiskCacheConfig structure."""

    cache_dir: str


class RateLimitConfigDict(TypedDict):
    """TypedDict representing the RateLimitConfig structure."""

    enabled: bool
    backend: Literal["memory", "redis", "diskcache"]
    redis: NotRequired[RedisConfigDict]
    diskcache: NotRequired[DiskCacheConfigDict]
    default_rpm: int
    default_tpm: int


class ServerDict(TypedDict):
    """TypedDict representing the Server structure."""

    host: str
    port: int
    url: NotRequired[str]


class EndpointOptionDict(TypedDict):
    """TypedDict representing the model endpoint option structure."""

    endpoint_id: str
    model_id: str
    enabled: NotRequired[bool]
    rpm: NotRequired[int]
    tpm: NotRequired[int]
    concurrent_requests: NotRequired[int]


class RequestMappingDict(TypedDict):
    """TypedDict representing custom request mapping rules."""

    method: NotRequired[str]
    path: NotRequired[str]
    headers: NotRequired[dict[str, str]]
    body_template: NotRequired[dict | list | str]
    query_template: NotRequired[dict | list | str]


class ResponseMappingDict(TypedDict):
    """TypedDict representing custom response mapping rules."""

    model_path: NotRequired[str]
    data_path: NotRequired[str]
    results_path: NotRequired[str]
    field_map: NotRequired[dict[str, str]]
    usage_map: NotRequired[dict[str, str]]


class ModelConfigDict(TypedDict):
    """TypedDict representing the model configuration structure."""

    id: str
    enabled: NotRequired[bool]
    endpoints: list[str | EndpointOptionDict]
    function_call_available: NotRequired[bool]
    response_format_available: NotRequired[bool]
    native_multimodal: NotRequired[bool]
    context_length: NotRequired[int | None]
    max_output_tokens: NotRequired[int | None]


class RetrievalModelConfigDict(TypedDict):
    """TypedDict representing retrieval model configuration."""

    id: str
    enabled: NotRequired[bool]
    endpoints: list[str | EndpointOptionDict]
    protocol: NotRequired[str | None]
    dimensions: NotRequired[int | None]
    default_top_n: NotRequired[int | None]
    request_mapping: NotRequired[RequestMappingDict | None]
    response_mapping: NotRequired[ResponseMappingDict | None]


class BackendSettingsDict(TypedDict):
    """TypedDict representing the BackendSettings structure."""

    models: dict[str, ModelConfigDict]
    default_endpoint: NotRequired[str | None]


class RetrievalBackendSettingsDict(TypedDict):
    """TypedDict representing retrieval backend settings."""

    models: dict[str, RetrievalModelConfigDict]
    default_endpoint: NotRequired[str | None]


class EndpointSettingDict(TypedDict):
    """TypedDict representing the EndpointSetting structure."""

    id: str
    enabled: NotRequired[bool]
    api_base: NotRequired[str | None]
    api_key: NotRequired[str]
    region: NotRequired[str]
    response_api: NotRequired[bool]
    endpoint_type: NotRequired[
        Literal[
            "default",
            "openai",
            "openai_azure",
            "openai_vertex",
            "anthropic",
            "anthropic_vertex",
            "anthropic_bedrock",
        ]
    ]
    credentials: NotRequired[dict]
    is_azure: NotRequired[bool]
    is_vertex: NotRequired[bool]
    is_bedrock: NotRequired[bool]
    rpm: NotRequired[int]
    tpm: NotRequired[int]
    concurrent_requests: NotRequired[int]
    proxy: NotRequired[str]
    headers: NotRequired[dict[str, str]]


class BackendsDict(TypedDict):
    """TypedDict representing all model backends in a single dictionary."""

    anthropic: NotRequired[BackendSettingsDict]
    deepseek: NotRequired[BackendSettingsDict]
    gemini: NotRequired[BackendSettingsDict]
    groq: NotRequired[BackendSettingsDict]
    local: NotRequired[BackendSettingsDict]
    minimax: NotRequired[BackendSettingsDict]
    mistral: NotRequired[BackendSettingsDict]
    moonshot: NotRequired[BackendSettingsDict]
    openai: NotRequired[BackendSettingsDict]
    qwen: NotRequired[BackendSettingsDict]
    yi: NotRequired[BackendSettingsDict]
    zhipuai: NotRequired[BackendSettingsDict]
    baichuan: NotRequired[BackendSettingsDict]
    stepfun: NotRequired[BackendSettingsDict]
    xai: NotRequired[BackendSettingsDict]
    xiaomi: NotRequired[BackendSettingsDict]
    ernie: NotRequired[BackendSettingsDict]


class SettingsV1Dict(TypedDict):
    """TypedDict representing the expected structure of the settings dictionary."""

    endpoints: list[EndpointSettingDict]
    token_server: NotRequired[ServerDict]
    rate_limit: NotRequired[RateLimitConfigDict]

    # V1 format: each model backend config
    anthropic: NotRequired[BackendSettingsDict]
    deepseek: NotRequired[BackendSettingsDict]
    gemini: NotRequired[BackendSettingsDict]
    groq: NotRequired[BackendSettingsDict]
    local: NotRequired[BackendSettingsDict]
    minimax: NotRequired[BackendSettingsDict]
    mistral: NotRequired[BackendSettingsDict]
    moonshot: NotRequired[BackendSettingsDict]
    openai: NotRequired[BackendSettingsDict]
    qwen: NotRequired[BackendSettingsDict]
    yi: NotRequired[BackendSettingsDict]
    zhipuai: NotRequired[BackendSettingsDict]
    baichuan: NotRequired[BackendSettingsDict]
    stepfun: NotRequired[BackendSettingsDict]
    xai: NotRequired[BackendSettingsDict]
    xiaomi: NotRequired[BackendSettingsDict]
    ernie: NotRequired[BackendSettingsDict]


class SettingsV2Dict(TypedDict):
    """TypedDict representing the expected structure of the settings dictionary."""

    VERSION: NotRequired[str]
    endpoints: list[EndpointSettingDict]
    token_server: NotRequired[ServerDict]
    rate_limit: NotRequired[RateLimitConfigDict]

    # V2 format: all model backend configs in a single dictionary
    backends: NotRequired[BackendsDict]
    embedding_backends: NotRequired[dict[str, RetrievalBackendSettingsDict]]
    rerank_backends: NotRequired[dict[str, RetrievalBackendSettingsDict]]


SettingsDict = SettingsV2Dict
