"""Yi defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

YI_DEFAULT_MODEL: Final[str]
YI_MODELS: Final[dict[str, ModelSettingDict]]
YI_DEFAULT_MODEL, YI_MODELS = load_backend_defaults("yi")
