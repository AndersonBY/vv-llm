"""Check live reporting without sending requests."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from vv_llm.types.llm_parameters import ChatCompletionMessage, Usage


spec = importlib.util.spec_from_file_location("effort_smoke", Path(__file__).parents[1] / "live" / "reasoning_effort_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_live_report_uses_normalized_response(monkeypatch):
    from vv_llm import chat_clients

    class Client:
        raw_client = SimpleNamespace(close=lambda: None)

        def create_completion(self, **options):
            assert options["reasoning_effort"] == "low"
            assert options["thinking"] == {"type": "enabled"}
            return ChatCompletionMessage(content="OK", usage=Usage(completion_tokens=2, prompt_tokens=3, total_tokens=5))

    monkeypatch.setattr(chat_clients, "create_chat_client", lambda **options: Client())
    route = {"backend": "openai", "model": "gpt-5", "wire_model": "gpt-5", "transport": "openai_azure", "endpoint_id": "private-endpoint"}
    result = smoke.run_case(None, route, "low", False, SimpleNamespace(timeout=1, max_tokens=64, effort=None, thinking="enabled"))
    assert result["result"] == "ACCEPTED"
    assert result["content_present"] is True
    assert result["output_tokens"] == 2
    assert result["thinking"] == "enabled"
    assert "endpoint_id" not in result


def test_rejection_report_does_not_expose_error_body(monkeypatch):
    from vv_llm import chat_clients

    class ProviderError(Exception):
        status_code = 400

    class Client:
        raw_client = SimpleNamespace(close=lambda: None)

        def create_completion(self, **options):
            raise ProviderError("invalid reasoning_effort; Authorization: secret-example")

    monkeypatch.setattr(chat_clients, "create_chat_client", lambda **options: Client())
    route = {"backend": "openai", "model": "gpt-5", "wire_model": "gpt-5", "transport": "openai_azure", "endpoint_id": "private-endpoint"}
    result = smoke.run_case(None, route, "vv_invalid_effort", True, SimpleNamespace(timeout=1, max_tokens=64, effort=None, thinking=None))
    assert result["result"] == "REJECTED"
    assert result["http_status"] == 400
    assert "secret-example" not in str(result)
