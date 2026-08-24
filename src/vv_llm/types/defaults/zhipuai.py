"""ZhiPuAI defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

ZHIPUAI_DEFAULT_MODEL: Final[str]
ZHIPUAI_MODELS: Final[dict[str, ModelSettingDict]]
ZHIPUAI_DEFAULT_MODEL, ZHIPUAI_MODELS = load_backend_defaults("zhipuai")
