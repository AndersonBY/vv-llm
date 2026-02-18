# @Author: Bi Ying
# @Date:   2024-07-27 18:26:05
try:
    from tests.dev_settings import (
        sample_settings,
        VECTORVEIN_API_KEY,
        VECTORVEIN_BASE_URL,
        VECTORVEIN_APP_ID,
        VECTORVEIN_WORKFLOW_ID,
        VECTORVEIN_VPP_API_KEY,
        VECTORVEIN_VAPP_ID,
    )
except ImportError:
    print("dev_settings.py not found, using default sample settings with empty API keys.")
    from v_llm.types import SettingsDict

    sample_settings: SettingsDict = {
        "VERSION": "2",
        # "token_server": {"host": "127.0.0.1", "port": 8338, "url": "http://127.0.0.1:8338"},
        "rate_limit": {
            "enabled": False,
            "backend": "redis",
            "redis": {
                "host": "127.0.0.1",
                "port": 6379,
                "db": 0,
            },
            "diskcache": {
                "cache_dir": ".rate_limit_cache",
            },
            "default_rpm": 60,
            "default_tpm": 1000000,
        },
        "endpoints": [
            {
                "id": "openai-test",
                "api_base": "https://api.openai.com/v1",
                "api_key": "sk-test-key",
            }
        ],
        "backends": {
            "openai": {
                "models": {
                    "gpt-5.2": {
                        "id": "gpt-5.2",
                        "endpoints": ["openai-test"],
                    }
                }
            }
        },
    }

    VECTORVEIN_API_KEY = "YOUR_API_KEY"
    VECTORVEIN_BASE_URL = "https://vectorvein.com/api/v1/open-api"
    VECTORVEIN_APP_ID = "YOUR_APP_ID"
    VECTORVEIN_WORKFLOW_ID = "YOUR_WORKFLOW_ID"
    VECTORVEIN_VPP_API_KEY = "YOUR_VPP_API_KEY"
    VECTORVEIN_VAPP_ID = "YOUR_VAPP_ID"
