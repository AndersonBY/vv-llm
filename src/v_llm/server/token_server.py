import importlib
from typing import Any, cast

from pydantic import BaseModel

from ..chat_clients.utils import get_token_counts
from ..settings import settings


_SERVER_IMPORT_ERROR: ModuleNotFoundError | None = None
_uvicorn: Any = None
_fastapi: Any = None
_http_exception_cls: type[Exception] = RuntimeError

try:
    _uvicorn = importlib.import_module("uvicorn")
    _fastapi = importlib.import_module("fastapi")
    _http_exception_cls = _fastapi.HTTPException
except ModuleNotFoundError as import_error:  # pragma: no cover - import guard
    _SERVER_IMPORT_ERROR = import_error


def _ensure_server_dependencies() -> None:
    if _SERVER_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "FastAPI server dependencies are not installed. "
            "Install with: pip install v-llm[server]"
        ) from _SERVER_IMPORT_ERROR


def _create_fastapi_app() -> Any:
    _ensure_server_dependencies()
    return _fastapi.FastAPI()


token_server = _create_fastapi_app() if _SERVER_IMPORT_ERROR is None else None


class TokenCountRequest(BaseModel):
    text: str | dict
    model: str = ""


if token_server is not None:

    @token_server.post("/count_tokens")
    async def count_tokens(request: TokenCountRequest):
        try:
            token_count = get_token_counts(request.text, request.model, use_token_server_first=False)
            return {"total_tokens": token_count}
        except Exception as e:
            raise cast(Any, _http_exception_cls)(status_code=500, detail=str(e)) from None


def run_token_server(host: str | None = None, port: int | None = None):
    """
    启动一个简单的HTTP服务器来处理token计数请求。参数均留空则使用 settings.token_server 的配置。

    参数:
        host (str): 服务器主机地址。
        port (int): 服务器端口。
    """
    _ensure_server_dependencies()

    if host is None or port is None:
        if settings.token_server is None:
            raise ValueError("Token server is not enabled.")

        _host = settings.token_server.host
        _port = settings.token_server.port
    else:
        _host = host
        _port = port

    _uvicorn.run(token_server, host=_host, port=_port)


if __name__ == "__main__":
    run_token_server()
