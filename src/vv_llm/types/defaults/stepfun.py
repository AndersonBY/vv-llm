"""StepFun defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

STEPFUN_DEFAULT_MODEL: Final[str]
STEPFUN_MODELS: Final[dict[str, ModelSettingDict]]
STEPFUN_DEFAULT_MODEL, STEPFUN_MODELS = load_backend_defaults("stepfun")
