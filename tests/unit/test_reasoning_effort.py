import asyncio
from types import SimpleNamespace

import pytest
from anthropic import Anthropic, AsyncAnthropic

from vv_llm import CapabilityPolicy, ChatRequest, ChatRequestOptions, FallbackChatClient, FallbackRoute, ModelCapabilities, ProviderRegistry, ScriptedChatClient, ThinkingPreference
from vv_llm.contract import load_fixture
from vv_llm.settings import Settings
from vv_llm.chat_clients.anthropic_client import AnthropicChatClient, AsyncAnthropicChatClient
from vv_llm.chat_clients.openai_client import OpenAIChatClient, AsyncOpenAIChatClient
from vv_llm.chat_clients.reasoning import resolve_reasoning_effort
from vv_llm.types.llm_parameters import ModelSetting
from vv_llm.types.defaults import ZHIPUAI_MODELS


@pytest.mark.parametrize("case", load_fixture("reasoning-effort.v1.json")["capability_cases"])
def test_shared_reasoning_cases(case):
    capabilities = ModelCapabilities(reasoning_efforts=case["reasoning_efforts"], reasoning_effort_aliases=case.get("reasoning_effort_aliases"))
    request = ChatRequest(model=case["model"], messages=[], options=ChatRequestOptions(reasoning_effort=case["reasoning_effort"]))
    if case["valid"]:
        request.validate_capabilities(capabilities, CapabilityPolicy.STRICT)
    else:
        with pytest.raises(ValueError, match="reasoning_effort"):
            request.validate_capabilities(capabilities, CapabilityPolicy.STRICT)
        with pytest.warns(UserWarning, match="reasoning_effort"):
            request.validate_capabilities(capabilities, CapabilityPolicy.WARN)
        request.validate_capabilities(capabilities, CapabilityPolicy.PASSTHROUGH)


def test_zhipuai_catalog_distinguishes_efforts_aliases_and_thinking():
    for model in ("glm-5.2", "glm-5.3", "glm-5.3-flash"):
        capabilities = ModelCapabilities(**ZHIPUAI_MODELS[model]["capabilities"])
        aliases = {"minimal": "none", "low": "high", "medium": "high", "xhigh": "max"} if model == "glm-5.2" else {}
        assert capabilities.reasoning_efforts == (["none", "high", "max"] if model == "glm-5.2" else ["low", "high", "max"])
        assert (capabilities.reasoning_effort_aliases or {}) == aliases
        for effort in [None, *capabilities.reasoning_efforts, *aliases]:
            request = ChatRequest(model=model, messages=[], options=ChatRequestOptions(reasoning_effort=effort, thinking=ThinkingPreference.enabled()))
            request.validate_capabilities(capabilities, CapabilityPolicy.STRICT)
        request.options.thinking = ThinkingPreference.disabled()
        request.options.reasoning_effort = None
        if model == "glm-5.2":
            request.validate_capabilities(capabilities, CapabilityPolicy.STRICT)
        else:
            with pytest.raises(ValueError, match="always enabled"):
                request.validate_capabilities(capabilities, CapabilityPolicy.STRICT)
            for effort in ("none", "minimal", "medium", "xhigh"):
                with pytest.raises(ValueError, match="reasoning_effort"):
                    capabilities.validate_reasoning_effort(effort, model, CapabilityPolicy.STRICT)
        with pytest.raises(ValueError, match="reasoning_effort"):
            capabilities.validate_reasoning_effort("ultra", model, CapabilityPolicy.STRICT)


def test_partial_capabilities_preserve_legacy_flags():
    model = ModelSetting(id="model", function_call_available=True, native_multimodal=True, capabilities={"reasoning_efforts": ["high"]})
    assert model.capabilities.tools
    assert model.native_multimodal
    assert model.capabilities.reasoning_efforts == ["high"]


class Captured(Exception):
    pass


def settings(protocol, levels=("low", "high")):
    backend = "anthropic" if protocol == "anthropic" else "openai"
    return Settings.load_from_dict(
        {
            "endpoints": [{"id": "test", "api_base": "https://example.invalid", "api_key": "test-key", "endpoint_type": backend, "response_api": protocol == "responses"}],
            "backends": {backend: {"models": {"test-model": {"id": "test-model", "endpoints": ["test"], "capabilities": {"reasoning_efforts": list(levels)}}}}},
        }
    )


def bind(client, protocol, asynchronous, captured, monkeypatch=None):
    def capture(**kwargs):
        captured.update(kwargs)
        raise Captured()

    async def acapture(**kwargs):
        return capture(**kwargs)

    create = acapture if asynchronous else capture
    client.endpoint = client.settings.get_endpoint("test")
    client.model_id = "test-model"
    if protocol == "anthropic":
        raw_type = AsyncAnthropic if asynchronous else Anthropic
        raw = raw_type.__new__(raw_type)
        raw.messages = SimpleNamespace(create=create)
    else:
        raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), responses=SimpleNamespace(create=create, stream=capture))
    if protocol == "anthropic":
        monkeypatch.setattr(type(client), "raw_client", property(lambda self: raw))
    else:
        client.raw_client = raw


@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_all_python_transport_paths_send_effort(protocol, asynchronous, stream, monkeypatch):
    classes = (AnthropicChatClient, AsyncAnthropicChatClient) if protocol == "anthropic" else (OpenAIChatClient, AsyncOpenAIChatClient)
    client = classes[asynchronous](model="test-model", random_endpoint=False, endpoint_id="test", settings=settings(protocol))
    captured = {}
    bind(client, protocol, asynchronous, captured, monkeypatch)
    kwargs = {
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "high",
        "capability_policy": CapabilityPolicy.STRICT,
        "stream": stream,
        "skip_cutoff": True,
        "max_tokens": 64,
    }

    async def run():
        result = await client.create_completion(**kwargs)
        if stream:
            await anext(result)

    with pytest.raises(Captured):
        if asynchronous:
            asyncio.run(run())
        else:
            result = client.create_completion(**kwargs)
            if stream:
                next(result)
    if protocol == "chat":
        assert captured["reasoning_effort"] == "high"
        assert "reasoning_effort" not in captured["extra_body"]
    else:
        container = "reasoning" if protocol == "responses" else "output_config"
        assert captured["extra_body"][container]["effort"] == "high"
        assert "reasoning_effort" not in captured


def test_model_switch_and_endpoint_override_are_checked_before_send():
    config = settings("chat")
    config.backends.openai.update_models(
        {},
        {
            "first": {"id": "first", "endpoints": ["test"], "capabilities": {"reasoning_efforts": ["low"]}},
            "second": {"id": "second", "endpoints": [{"endpoint_id": "test", "capabilities": {"reasoning_efforts": ["high"]}}], "capabilities": {"reasoning_efforts": ["low"]}},
        },
    )
    client = OpenAIChatClient(model="first", endpoint_id="test", random_endpoint=False, settings=config)
    captured = {}
    bind(client, "chat", False, captured)
    request = ChatRequest(model="second", messages=[{"role": "user", "content": "hello"}], skip_cutoff=True, options=ChatRequestOptions(reasoning_effort="high", max_tokens=64))
    with pytest.raises(Captured):
        client.create(request, capability_policy=CapabilityPolicy.STRICT)
    assert captured["model"] == "second"
    assert client.capabilities.reasoning_efforts == ["high"]
    with pytest.raises(ValueError, match="does not support reasoning_effort"):
        client.create_completion(messages=[], reasoning_effort="low", capability_policy=CapabilityPolicy.STRICT)


@pytest.mark.parametrize("extra_body", [None, {"reasoning_effort": "high"}, {"reasoning": {"effort": "high"}}, {"output_config": {"effort": "high"}}])
def test_fallback_checks_each_model_of_one_provider_without_downgrading(extra_body):
    registry = ProviderRegistry()
    scripted = ScriptedChatClient(["ok"])
    registry.register(
        "provider",
        lambda: scripted,
        capabilities=ModelCapabilities(),
        model_capabilities={
            "low": ModelCapabilities(reasoning_efforts=["low"]),
            "high": ModelCapabilities(reasoning_efforts=["high"]),
        },
    )
    client = FallbackChatClient(registry, [FallbackRoute("provider", "low"), FallbackRoute("provider", "high")])
    request = ChatRequest(messages=[], extra_body=extra_body, options=ChatRequestOptions(reasoning_effort="high" if extra_body is None else None))
    assert client.create(request) == "ok"
    assert len(scripted.requests) == 1
    assert scripted.requests[0].model == "high"
    assert scripted.requests[0].options.reasoning_effort == request.options.reasoning_effort
    assert scripted.requests[0].extra_body == extra_body


def test_native_fields_preserve_siblings_and_reject_conflicts():
    for protocol, container in [("responses", "reasoning"), ("anthropic", "output_config")]:
        body = {container: {"effort": "high", "other": True}}
        effort, resolved = resolve_reasoning_effort("high", body, protocol, "openai")
        assert effort == "high" and resolved == body
        with pytest.raises(ValueError, match="Conflicting"):
            resolve_reasoning_effort("low", body, protocol, "openai")
    with pytest.raises(ValueError, match="thinking_level/thinking_budget"):
        resolve_reasoning_effort("high", {"google": {"thinking_config": {"thinking_budget": 0}}}, "chat", "gemini")
    assert resolve_reasoning_effort(None, None, "chat", "openai") == (None, {})


def test_legacy_entry_rejects_extra_body_override_before_send():
    client = OpenAIChatClient(model="test-model", settings=settings("chat"))
    with pytest.raises(ValueError, match="Conflicting reasoning_effort"):
        client.create_completion(messages=[], reasoning_effort="high", extra_body={"reasoning_effort": "low"})


def test_aliases_are_preserved_on_wire_and_binding_maps_are_replaced():
    fixture = load_fixture("settings-resolution.v1.json")
    config = Settings.load_from_dict(fixture["settings"])
    from vv_llm.chat_clients.deepseek_client import DeepSeekChatClient

    client = DeepSeekChatClient(model="chat-alias", endpoint_id="shared-endpoint", random_endpoint=False, settings=config)
    assert client.capabilities.reasoning_effort_aliases == {"ultra": "low"}
    client.capabilities.validate_reasoning_effort("ultra", client.model, CapabilityPolicy.STRICT)
    with pytest.raises(ValueError, match="reasoning_effort"):
        client.capabilities.validate_reasoning_effort("max", client.model, CapabilityPolicy.STRICT)
    client = OpenAIChatClient(model="test-model", settings=settings("chat", levels=("high",)))
    client.backend_settings.get_model_setting(client.model).capabilities.reasoning_effort_aliases = {"xhigh": "high"}
    captured = {}
    bind(client, "chat", False, captured)
    with pytest.raises(Captured):
        client.create_completion(messages=[], reasoning_effort="xhigh", max_tokens=64, skip_cutoff=True, capability_policy=CapabilityPolicy.STRICT)
    assert captured["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_thinking_merge_preserves_budget(asynchronous, stream, monkeypatch):
    client_type = AsyncAnthropicChatClient if asynchronous else AnthropicChatClient
    client = client_type(model="test-model", settings=settings("anthropic"))
    captured = {}
    bind(client, "anthropic", asynchronous, captured, monkeypatch)
    body = {"thinking": {"budget_tokens": 1024}}
    kwargs = {"messages": [{"role": "user", "content": "hello"}], "thinking": {"type": "enabled"}, "extra_body": body, "max_tokens": 2048, "skip_cutoff": True, "stream": stream}

    async def run():
        result = await client.create_completion(**kwargs)
        if stream:
            await anext(result)

    with pytest.raises(Captured):
        if asynchronous:
            asyncio.run(run())
        else:
            result = client.create_completion(**kwargs)
            if stream:
                next(result)
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert body == {"thinking": {"budget_tokens": 1024}}


def test_capabilities_include_automatic_binding_before_first_request():
    config = settings("chat")
    config.backends.openai.models["test-model"].endpoints = [
        {"endpoint_id": "missing", "capabilities": {"reasoning_efforts": ["low"]}},
        {"endpoint_id": "test", "capabilities": {"reasoning_efforts": ["high"]}},
    ]
    client = OpenAIChatClient(model="test-model", settings=config)
    assert client.capabilities.reasoning_efforts == ["high"]


@pytest.mark.parametrize("explicit_tools", [None, True])
def test_partial_effort_override_preserves_explicit_legacy_flags(explicit_tools):
    capabilities = {"reasoning_efforts": ["high"]}
    if explicit_tools is not None:
        capabilities["tools"] = explicit_tools
    config = Settings.load_from_dict(
        {
            "backends": {
                "deepseek": {
                    "models": {
                        "deepseek-flash": {
                            "function_call_available": False,
                            "response_format_available": False,
                            "native_multimodal": False,
                            "capabilities": capabilities,
                        }
                    }
                }
            }
        }
    )
    model = config.backends.deepseek.models["deepseek-flash"]
    assert model.capabilities.tools is bool(explicit_tools)
    assert model.function_call_available is bool(explicit_tools)
    assert model.response_format_available is False
    assert model.native_multimodal is False
    assert model.capabilities.reasoning_efforts == ["high"]


@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic"])
def test_extra_body_cannot_change_the_validated_model(protocol, monkeypatch):
    client_type = AnthropicChatClient if protocol == "anthropic" else OpenAIChatClient
    client = client_type(model="test-model", settings=settings(protocol))
    captured = {}
    bind(client, protocol, False, captured, monkeypatch)
    with pytest.raises(ValueError, match="model"):
        client.create_completion(
            messages=[{"role": "user", "content": "hello"}], extra_body={"model": "other-model"}, max_tokens=64, skip_cutoff=True, capability_policy=CapabilityPolicy.PASSTHROUGH
        )
    assert captured == {}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_fallback_checks_provider_options_for_the_actual_backend(asynchronous, stream):
    from vv_llm import AsyncFallbackChatClient, AsyncScriptedChatClient, ScriptedStream
    from vv_llm.types.enums import BackendType
    from vv_llm.types.llm_parameters import ChatCompletionDeltaMessage

    scripted_type = AsyncScriptedChatClient if asynchronous else ScriptedChatClient
    scripted = scripted_type([ScriptedStream([ChatCompletionDeltaMessage(content="ok")]) if stream else "ok"])
    scripted.backend_name = BackendType.OpenAI
    registry = ProviderRegistry()
    registry.register(
        "custom-name",
        lambda: scripted,
        capabilities=ModelCapabilities(),
        model_capabilities={
            "low": ModelCapabilities(reasoning_efforts=["low"]),
            "high": ModelCapabilities(reasoning_efforts=["high"]),
        },
    )
    fallback_type = AsyncFallbackChatClient if asynchronous else FallbackChatClient
    client = fallback_type(registry, [FallbackRoute("custom-name", "low"), FallbackRoute("custom-name", "high")])
    request = ChatRequest(messages=[], stream=stream, options=ChatRequestOptions(provider_options={"openai": {"reasoning_effort": "high"}}))

    async def run():
        response = await client.create(request)
        if stream:
            return [chunk async for chunk in response]
        return response

    if asynchronous:
        asyncio.run(run())
    elif stream:
        list(client.create(request))
    else:
        client.create(request)
    assert len(scripted.requests) == 1
    assert scripted.requests[0].model == "high"


@pytest.mark.parametrize("first_enabled", [False, True])
def test_repeated_requests_keep_the_selected_binding_and_wire_model(first_enabled):
    config = settings("chat")
    config.backends.openai.models["test-model"].endpoints = [
        {"endpoint_id": "test", "model_id": "low-wire", "enabled": first_enabled, "priority": 2, "capabilities": {"reasoning_efforts": ["low"]}},
        {"endpoint_id": "test", "model_id": "high-wire", "priority": 1, "capabilities": {"reasoning_efforts": ["high"]}},
    ]
    client = OpenAIChatClient(model="test-model", settings=config)
    captured = {}
    bind(client, "chat", False, captured)
    client.endpoint = None
    for _ in range(2):
        with pytest.raises(Captured):
            client.create_completion(messages=[], reasoning_effort="high", max_tokens=64, skip_cutoff=True, capability_policy=CapabilityPolicy.STRICT)
        assert captured["model"] == "high-wire"
        assert client.capabilities.reasoning_efforts == ["high"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_keyword_entries_forward_capability_policy(asynchronous, wrapped, monkeypatch):
    from vv_llm import AsyncMiddlewareChatClient, AsyncScriptedChatClient, MiddlewareChatClient

    inner = AsyncScriptedChatClient(["ok"]) if asynchronous else ScriptedChatClient(["ok"])
    original = inner.create
    policies = []

    def capture(request, **kwargs):
        policies.append(kwargs.get("capability_policy"))
        return original(request, **kwargs)

    monkeypatch.setattr(inner, "create", capture)
    wrapper = AsyncMiddlewareChatClient if asynchronous else MiddlewareChatClient
    client = wrapper(inner) if wrapped else inner
    response = client.create_completion(messages=[], reasoning_effort="high", capability_policy=CapabilityPolicy.STRICT)
    if asynchronous:
        response = asyncio.run(response)
    assert response == "ok"
    assert policies == [CapabilityPolicy.STRICT]
    assert inner.requests[0].options.reasoning_effort == "high"


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "google/gemini-4-flash", "gemini-2.5-flash", "gpt-5.5"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_gemini_wire_parameters(model, asynchronous, stream, nested):
    from copy import deepcopy
    from vv_llm.types.llm_parameters import NOT_GIVEN

    config = settings("chat")
    config.backends.openai.models["test-model"].id = model
    client = (OpenAIChatClient, AsyncOpenAIChatClient)[asynchronous](model="test-model", endpoint_id="test", random_endpoint=False, settings=config)
    captured = {}
    bind(client, "chat", asynchronous, captured)
    google = {"google": {"thinking_config": {"thinkingBudget": 1024, "thinkingLevel": "high", "include_thoughts": True}}}
    body = {"extra_body": google} if nested else google
    body.update({"temperature": 0.4, "top_p": 0.8, "top_k": 20, "topP": 0.8, "topK": 20})
    original = deepcopy(body)
    kwargs = dict(messages=[{"role": "user", "content": "hello"}], temperature=0.3, top_p=0.7, extra_body=body, stream=stream, skip_cutoff=True, max_tokens=64)

    async def run():
        result = await client.create_completion(**kwargs)
        if stream:
            await anext(result)

    with pytest.raises(Captured):
        if asynchronous:
            asyncio.run(run())
        else:
            result = client.create_completion(**kwargs)
            if stream:
                next(result)
    modern = "gemini-3" in model or "gemini-4" in model
    assert captured["temperature"] == (NOT_GIVEN if modern else 0.3)
    assert captured["top_p"] == (NOT_GIVEN if modern else 0.7)
    sent = captured["extra_body"]
    for key in ("temperature", "top_p", "top_k", "topP", "topK"):
        assert (key in sent) is (not modern)
    thinking = (sent["extra_body"] if nested else sent)["google"]["thinking_config"]
    assert thinking == ({"thinking_level": "high", "include_thoughts": True} if modern else original.get("extra_body", original)["google"]["thinking_config"])
    assert body == original
    assert client.temperature == 0.3


def test_gemini_budget_uses_default_and_level_alias_conflicts_are_rejected():
    from vv_llm.chat_clients.reasoning import normalize_gemini_body

    assert normalize_gemini_body({"google": {"thinking_config": {"thinking_budget": 0, "include_thoughts": True}}}) == {"google": {"thinking_config": {"include_thoughts": True}}}
    with pytest.raises(ValueError, match="Conflicting Gemini"):
        normalize_gemini_body({"google": {"thinking_config": {"thinking_level": "low", "thinkingLevel": "high"}}})
    for field in ("thinkingLevel", "thinkingBudget"):
        with pytest.raises(ValueError, match="conflicts"):
            resolve_reasoning_effort("high", {"google": {"thinking_config": {field: "high"}}}, "chat", "gemini", "gemini-3.8-flash")
    for model in ("gemini-3.7-flash", "gemini-3.8-flash"):
        from vv_llm.types.defaults import GEMINI_MODELS
        capabilities = ModelCapabilities(**GEMINI_MODELS[model]["capabilities"])
        assert capabilities.reasoning_efforts == ["low", "medium", "high"]
        with pytest.raises(ValueError, match="reasoning_effort"):
            capabilities.validate_reasoning_effort("minimal", model, CapabilityPolicy.STRICT)
