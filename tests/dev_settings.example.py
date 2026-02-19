"""Template for local credentials.

Usage:
1) Copy this file to `tests/dev_settings.py`
2) Fill in your real keys/secrets
3) Never commit `tests/dev_settings.py`
"""

from __future__ import annotations

from vv_llm.types import SettingsDict

sample_settings: SettingsDict = {
    "VERSION": "2",
    "token_server": {"host": "127.0.0.1", "port": 8338, "url": "http://127.0.0.1:8338"},
    "rate_limit": {
        "enabled": False,
        "backend": "memory",
        "default_rpm": 60,
        "default_tpm": 1000000,
        "redis": {"host": "127.0.0.1", "port": 6379, "db": 0},
        "diskcache": {"cache_dir": ".rate_limit_cache"},
    },
    "endpoints": [
        {
            "id": "openai-default",
            "endpoint_type": "openai",
            "api_base": "https://api.openai.com/v1",
            "api_key": "YOUR_OPENAI_API_KEY",
        },
        {
            "id": "anthropic-default",
            "endpoint_type": "anthropic",
            "api_base": "https://api.anthropic.com",
            "api_key": "YOUR_ANTHROPIC_API_KEY",
        },
        {
            "id": "gemini-default",
            "endpoint_type": "openai",
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": "YOUR_GEMINI_API_KEY",
        },
    ],
    "backends": {
        "openai": {"models": {"gpt-4o": {"id": "gpt-4o", "endpoints": ["openai-default"]}}},
        "anthropic": {"models": {"claude-sonnet-4-6": {"id": "claude-sonnet-4-6", "endpoints": ["anthropic-default"]}}},
        "gemini": {"models": {"gemini-2.5-pro": {"id": "gemini-2.5-pro", "endpoints": ["gemini-default"]}}},
    },
}
