from vv_llm import (
    ChatRequest,
    ErrorKind,
    FallbackChatClient,
    FallbackRoute,
    ModelCapabilities,
    ProviderRegistry,
    ScriptedChatClient,
    VvLlmError,
)


def main() -> None:
    primary = ScriptedChatClient(
        [VvLlmError("temporary failure", kind=ErrorKind.PROVIDER_INTERNAL)],
        provider="primary",
    )
    secondary = ScriptedChatClient(["fallback response"], provider="secondary")

    registry = ProviderRegistry()
    registry.register("primary", lambda: primary, capabilities=ModelCapabilities())
    registry.register("secondary", lambda: secondary, capabilities=ModelCapabilities())
    runtime = FallbackChatClient(
        registry,
        [
            FallbackRoute("primary", "primary-model"),
            FallbackRoute("secondary", "secondary-model"),
        ],
    )
    result = runtime.create_with_metadata(ChatRequest(messages=[{"role": "user", "content": "hello"}]))
    print(result.response)
    print("provider:", result.metadata.provider)
    print("fallback index:", result.metadata.fallback_index)


if __name__ == "__main__":
    main()
