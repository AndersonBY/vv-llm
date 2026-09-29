"""Inspect catalog choices and exercise validation/fallback without credentials."""

import warnings

from vv_llm import (
    CapabilityPolicy,
    ChatRequest,
    ChatRequestOptions,
    FallbackChatClient,
    FallbackRoute,
    ModelCapabilities,
    ProviderRegistry,
    ScriptedChatClient,
)
from vv_llm.contract import load_catalog


def main() -> None:
    catalog = load_catalog()
    models = [
        ("deepseek", "deepseek-flash"),
        ("deepseek", "deepseek-v4.1-flash"),
        ("zhipuai", "glm-5.2"),
        ("zhipuai", "glm-5.3"),
        ("zhipuai", "glm-5.3-flash"),
    ]
    capabilities = {}
    for backend, model in models:
        capability = ModelCapabilities(**catalog["backends"][backend]["models"][model]["capabilities"])
        capabilities[model] = capability
        print(model, "choices:", capability.reasoning_efforts)
        print("  aliases:", capability.reasoning_effort_aliases or {}, "thinking:", capability.thinking.value)

    request = ChatRequest(model="deepseek-flash", messages=[], options=ChatRequestOptions(reasoning_effort="xhigh"))
    request.validate_capabilities(capabilities[request.model], CapabilityPolicy.STRICT)
    assert request.options.reasoning_effort == "xhigh"
    print("Accepted alias stays unchanged:", request.options.reasoning_effort)

    request = ChatRequest(model="glm-5.3", messages=[], options=ChatRequestOptions(reasoning_effort="xhigh"))
    try:
        request.validate_capabilities(capabilities[request.model], CapabilityPolicy.STRICT)
    except ValueError as error:
        print("STRICT:", error)
    else:
        raise AssertionError("GLM-5.3 must reject xhigh in strict mode")
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        request.validate_capabilities(capabilities[request.model], CapabilityPolicy.WARN)
    assert observed
    print("WARN:", observed[0].message)
    request.validate_capabilities(capabilities[request.model], CapabilityPolicy.PASSTHROUGH)
    print("PASSTHROUGH: model support validation skipped")

    scripted = ScriptedChatClient(["OK"], provider="zhipuai")
    registry = ProviderRegistry()
    registry.register("zhipuai", lambda: scripted, capabilities=ModelCapabilities(), model_capabilities={model: capabilities[model] for model in ("glm-5.3", "glm-5.2")})
    fallback = FallbackChatClient(registry, [FallbackRoute("zhipuai", "glm-5.3"), FallbackRoute("zhipuai", "glm-5.2")])
    result = fallback.create_with_metadata(ChatRequest(messages=[{"role": "user", "content": "hello"}], options=ChatRequestOptions(reasoning_effort="xhigh")))
    assert result.metadata.fallback_index == 1
    assert scripted.requests[0].model == "glm-5.2"
    assert scripted.requests[0].options.reasoning_effort == "xhigh"
    print("Fallback selected:", scripted.requests[0].model, "effort:", scripted.requests[0].options.reasoning_effort)


if __name__ == "__main__":
    main()
