"""Opt-in decision smoke test; logs only shapes, token counts and error classes."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path

from vv_llm import create_decision_client, create_async_decision_client, DecisionRequest, PredicateAnswer, ChoiceAnswer, ScoreAnswer
from vv_llm.contract import load_fixture


def _config():
    source = os.environ.get("VV_LLM_SETTINGS_JSON")
    if not source:
        raise RuntimeError("Set VV_LLM_SETTINGS_JSON to an explicit local settings file")
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    if not data.get("decision_backends"):
        backend = deepcopy(data.get("backends", {}).get("openai", {}))
        if not backend.get("default_endpoint"):
            bindings = [binding for model in backend.get("models", {}).values() for binding in model.get("endpoints", [])]
            for binding in bindings:
                endpoint_id = binding if isinstance(binding, str) else binding.get("endpoint_id")
                if isinstance(binding, dict) and binding.get("enabled") is False:
                    continue
                endpoint = next((entry for entry in data.get("endpoints", []) if entry.get("id") == endpoint_id), {})
                if endpoint.get("enabled", True) and endpoint.get("api_key") and endpoint.get("endpoint_type") in {None, "default", "openai"} and not any(endpoint.get(flag) for flag in ("is_azure", "is_vertex", "is_bedrock")):
                    backend["default_endpoint"] = endpoint_id
                    break
        data["decision_backends"] = {"openai": backend}
    backend = data["decision_backends"]["openai"]
    model = os.environ.get("VV_LLM_MODEL", "gpt-6-luna")
    selected = backend.get("models", {}).get(model)
    ids = {backend.get("default_endpoint")}
    for binding in (selected or {}).get("endpoints", []):
        ids.add(binding if isinstance(binding, str) else binding.get("endpoint_id"))
    return {"endpoints": [endpoint for endpoint in data.get("endpoints", []) if endpoint.get("id") in ids], "decision_backends": {"openai": {"default_endpoint": backend.get("default_endpoint"), "models": {model: selected} if selected else {}}}}


def _flags(response):
    assert len(response.answers) == 3
    damaged, intent, urgency = response.answers
    assert isinstance(damaged, PredicateAnswer) and damaged.probability > 0.5
    assert isinstance(intent, ChoiceAnswer) and intent.choice == "replacement"
    assert isinstance(urgency, ScoreAnswer) and 0 <= urgency.score <= 2
    return {"answer_types": [answer.type for answer in response.answers], "semantic_checks_passed": True, "usage": {key: value for key, value in response.usage.model_dump(exclude_unset=True).items() if key in {"input_tokens", "output_tokens", "total_tokens"}} if response.usage else None}


async def _async(config, request):
    async with create_async_decision_client(model=request.model, settings=config, endpoint_id=os.environ.get("VV_LLM_ENDPOINT", ""), max_retries=0, timeout=60) as client:
        return _flags(await client.create(request))


def main():
    if os.environ.get("VV_LLM_RUN_LIVE_TESTS") != "1":
        print("decision live test skipped; set VV_LLM_RUN_LIVE_TESTS=1")
        return 0
    try:
        config = _config()
        request = DecisionRequest.from_contract(load_fixture("decisions.v1.json")["request"])
        request.model = os.environ.get("VV_LLM_MODEL", "gpt-6-luna")
        results = {}
        try:
            with create_decision_client(model=request.model, settings=config, endpoint_id=os.environ.get("VV_LLM_ENDPOINT", ""), max_retries=0, timeout=60) as client:
                results["sync"] = {"status": "passed", **_flags(client.create(request))}
        except Exception as error:
            results["sync"] = {"status": "failed", "error_type": type(error).__name__, "http_status": getattr(error, "status_code", None)}
        try:
            results["async"] = {"status": "passed", **asyncio.run(_async(config, request))}
        except Exception as error:
            results["async"] = {"status": "failed", "error_type": type(error).__name__, "http_status": getattr(error, "status_code", None)}
        passed = all(result["status"] == "passed" for result in results.values())
        print(json.dumps({"decision_live_smoke": "passed" if passed else "failed", **results}))
        return 0 if passed else 1
    except Exception as error:
        print(json.dumps({"decision_live_smoke": "failed", "error_type": type(error).__name__, "status": getattr(error, "status_code", None)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
