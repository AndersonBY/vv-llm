# @Author: Bi Ying
# @Date:   2024-07-27 00:02:34
from typing import Final
from typing_extensions import NotRequired, TypedDict

from ..enums import ContextLengthControlType

CONTEXT_LENGTH_CONTROL: Final[ContextLengthControlType] = ContextLengthControlType.Latest

ENDPOINT_CONCURRENT_REQUESTS: Final[int] = 20
ENDPOINT_RPM: Final[int] = 60
ENDPOINT_TPM: Final[int] = 300000


class ModelSettingDict(TypedDict):
    id: str
    function_call_available: bool
    response_format_available: bool
    native_multimodal: bool
    context_length: int
    max_output_tokens: NotRequired[int]
