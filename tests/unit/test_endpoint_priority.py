from copy import deepcopy

import pytest
from pydantic import ValidationError

from vv_llm.chat_clients.openai_client import OpenAIChatClient, AsyncOpenAIChatClient
from vv_llm.chat_clients.utils import _get_first_enabled_endpoint
from vv_llm.retrieval_clients.common import BaseRetrievalClient, BaseAsyncRetrievalClient
from vv_llm.settings import Settings, order_endpoints


def payload(section="backends", priority=2):
    return {
        "endpoints": [{"id": name, "api_key": "test"} for name in ("low", "high", "peer")],
        section: {
            "openai": {
                "models": {
                    "test": {
                        "id": "test",
                        "endpoints": [
                            {"endpoint_id": "low", "model_id": "low-model", "priority": priority},
                            {"endpoint_id": "high", "model_id": "high-model"},
                            "peer",
                        ],
                    }
                }
            }
        },
    }


@pytest.mark.parametrize("section", ["backends", "embedding_backends", "rerank_backends"])
def test_priority_round_trip(section):
    settings = Settings()
    settings.load(payload(section))
    exported = settings.export()
    restored = Settings()
    restored.load(exported)
    assert restored.export()[section]["openai"]["models"]["test"]["endpoints"] == exported[section]["openai"]["models"]["test"]["endpoints"]
    assert exported[section]["openai"]["models"]["test"]["endpoints"][0]["priority"] == 2


@pytest.mark.parametrize("section", ["backends", "embedding_backends", "rerank_backends"])
@pytest.mark.parametrize("priority", [0, -1, True, False, "2", 1.0, 1.5, None])
def test_invalid_priority(section, priority):
    with pytest.raises(ValidationError):
        Settings.load_from_dict(payload(section, priority))


def test_order_is_stable_and_preference_cannot_override_priority():
    endpoints = payload()["backends"]["openai"]["models"]["test"]["endpoints"]
    original = deepcopy(endpoints)
    assert order_endpoints(endpoints) == [endpoints[1], endpoints[2], endpoints[0]]
    assert order_endpoints(endpoints, "low") == [endpoints[1], endpoints[2], endpoints[0]]
    assert order_endpoints(endpoints, "peer") == [endpoints[2], endpoints[1], endpoints[0]]
    assert order_endpoints(endpoints, "missing") == order_endpoints(endpoints)
    assert endpoints == original
    assert order_endpoints(endpoints) is not endpoints
    assert order_endpoints([]) == []


@pytest.mark.parametrize("client_type", [OpenAIChatClient, AsyncOpenAIChatClient])
def test_chat_priority_and_explicit_endpoint(client_type):
    settings = Settings.load_from_dict(payload())
    client = client_type(model="test", settings=settings)
    client.endpoint_id = "low"
    assert client._set_endpoint()[0].id == "high"
    assert client_type(model="test", settings=settings, endpoint_id="low")._set_endpoint()[0].id == "low"
    settings.get_endpoint("high").enabled = False
    client = client_type(model="test", settings=settings)
    assert client._set_endpoint()[0].id == "peer"
    model = settings.backends.openai.models["test"]
    assert _get_first_enabled_endpoint(model, settings).id == "peer"


@pytest.mark.parametrize("section", ["embedding_backends", "rerank_backends"])
@pytest.mark.parametrize("client_type", [BaseRetrievalClient, BaseAsyncRetrievalClient])
def test_retrieval_priority_and_explicit_endpoint(section, client_type):
    settings = Settings.load_from_dict(payload(section))
    kwargs = dict(model="test", backend_name="openai", backend_settings=getattr(settings, section)["openai"], settings=settings)
    client = client_type(**kwargs)
    client.endpoint_id = "low"
    assert client._set_endpoint()[0].id == "high"
    assert client_type(**kwargs, endpoint_id="low")._set_endpoint()[0].id == "low"
    settings.get_endpoint("high").enabled = False
    assert client_type(**kwargs)._set_endpoint()[0].id == "peer"
