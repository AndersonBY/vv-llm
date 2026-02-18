from __future__ import annotations

import threading
import time

from v_llm.chat_clients.utils import get_token_counts
from v_llm.server.token_server import run_token_server
from v_llm.settings import settings

from live_common import load_live_settings


def main() -> None:
    load_live_settings(settings, require_credentials=False)

    server_thread = threading.Thread(target=run_token_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    print("Token server 正在后台运行")
    print(get_token_counts("Hello Maker毕! 你好！", "claude-sonnet-4-6", True))
    print(get_token_counts("Hello Maker毕! 你好！", "gemini-3-flash-preview", True))


if __name__ == "__main__":
    main()
