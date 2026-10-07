from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import httpx2
import pytest
from openai import RateLimitError

from vv_llm import (DecisionRequest, DecisionResponse, PredicateQuestion, create_decision_client, create_async_decision_client)
from vv_llm.contract import load_fixture
from vv_llm.settings import Settings

FIXTURE = load_fixture("decisions.v1.json")


def config():
    return {"endpoints": [{"id": "disabled", "enabled": False}, {"id": "active", "api_base": "https://example.invalid/v1", "api_key": "test-key", "headers": {"x-test": "preserved"}}], "decision_backends": {"openai": {"models": {"gpt-6-luna": {"id": "gpt-6-luna", "endpoints": [{"endpoint_id": "disabled", "priority": 1}, {"endpoint_id": "active", "priority": 2, "model_id": "luna-deployment"}]}}}}}


def test_shared_decision_contract_and_invalid_payloads():
    for valid in FIXTURE["valid_requests"]:
        assert DecisionRequest.from_contract(valid).to_contract() == valid
    request = DecisionRequest.from_contract(FIXTURE["request"])
    for valid in FIXTURE["valid_responses"]:
        assert DecisionResponse.from_contract(valid).to_contract() == valid
    response = DecisionResponse.from_contract(FIXTURE["response"])
    response.validate_for(request)
    assert request.to_contract() == FIXTURE["request"]
    assert response.to_contract() == FIXTURE["response"]
    assert DecisionResponse.from_contract(FIXTURE["refusal_response"]).answers[0].type == "refusal"
    for raw in FIXTURE["invalid_requests"]:
        with pytest.raises(ValueError):
            DecisionRequest.from_contract(raw)
    for raw in FIXTURE["invalid_responses"]:
        with pytest.raises(ValueError):
            DecisionResponse.from_contract(raw)
    missing = DecisionResponse.model_validate({"model": "gpt-6-luna", "answers": [{"type": "predicate", "probability": 0}], "usage": {"output_tokens": 0}})
    assert "input_tokens" not in missing.to_contract()["usage"]
    assert missing.to_contract()["usage"]["output_tokens"] == 0
    raw = deepcopy(FIXTURE["request"])
    raw["questions"][1]["name"] = raw["questions"][0]["name"]
    with pytest.raises(ValueError, match="unique"):
        DecisionRequest.from_contract(raw)


def test_sync_and_async_decision_transport_reuses_settings_without_mutating_request():
    bodies = []
    def handle(request):
        assert str(request.url) == "https://example.invalid/v1/decisions"
        assert request.headers["x-test"] == "preserved"
        body = json.loads(request.content)
        assert body["questions"][1]["choices"] == ["replacement", "refund"]
        assert body["questions"][2]["rubric"] == [{"label": "low"}, {"label": "medium"}, {"label": "high"}]
        bodies.append(body)
        return httpx2.Response(200, json={**FIXTURE["response"], "future_provider_field": True})
    request = DecisionRequest.from_contract(FIXTURE["request"])
    with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
        with create_decision_client(settings=config(), http_client=transport) as client:
            response = client.create(request)
            assert response.to_contract() == FIXTURE["response"]
        assert not transport.is_closed
    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as transport:
            async with create_async_decision_client(settings=config(), http_client=transport) as client:
                response = await client.create(request)
                assert response.to_contract() == FIXTURE["response"]
            assert not transport.is_closed
    asyncio.run(run())
    assert len(bodies) == 2
    assert all(body["model"] == "luna-deployment" for body in bodies)
    assert request.to_contract() == FIXTURE["request"]
    assert not Settings(**config()).backends.openai.models["gpt-6-luna"].endpoints


def test_unsupported_decision_types_fail_before_network_and_errors_preserve_headers():
    raw = config()
    raw["decision_backends"]["openai"]["models"]["gpt-6-luna"]["endpoints"][1]["capabilities"] = {"decision_types": []}
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: pytest.fail("must not send"))) as transport:
        with create_decision_client(settings=raw, http_client=transport) as client:
            with pytest.raises(ValueError, match="declare support"):
                client.create(DecisionRequest.from_contract(FIXTURE["request"]))
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "2", "x-request-id": "fixture-id"}))) as transport:
        with create_decision_client(settings=config(), http_client=transport) as client:
            with pytest.raises(RateLimitError) as info:
                client.create(DecisionRequest.from_contract(FIXTURE["request"]))
            assert info.value.response.headers["retry-after"] == "2"
    request = DecisionRequest(input="hello", questions=[PredicateQuestion(instructions="Is this a greeting?", name="greeting")])
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json={"model": "gpt-6-luna", "answers": [{"type": "refusal", "name": "greeting"}]}))) as transport:
        with create_decision_client(settings=config(), http_client=transport) as client:
            assert client.create(request).answers[0].type == "refusal"


def test_decision_alias_inherits_catalog_capabilities_and_rejects_missing_wire_type():
    raw = config()
    models = raw["decision_backends"]["openai"]["models"]
    models["friendly"] = {**models.pop("gpt-6-luna"), "id": "gpt-6-luna"}
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=FIXTURE["response"]))) as transport:
        with create_decision_client(model="friendly", settings=raw, http_client=transport) as client:
            request = DecisionRequest.from_contract(FIXTURE["request"])
            request.model = "friendly"
            assert len(client.create(request).answers) == 3
    invalid = deepcopy(FIXTURE["request"])
    del invalid["questions"][0]["type"]
    with pytest.raises(ValueError):
        DecisionRequest.from_contract(invalid)
