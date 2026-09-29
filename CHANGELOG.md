# Changelog

## 0.7.1 - 2026-09-29

- Adopt catalog revision 13 with `claude-opus-5-5`, `claude-sonnet-5-5`, `gemini-3.8-flash`, `gpt-6-sol`, and `gpt-6-luna`.
- Send `max_completion_tokens` for the GPT-6 family, which rejects `max_tokens` on Chat Completions.

## 0.7.0 - 2026-09-29

- Adopt vv-llm-contract 1.2.0, catalog revision 10, with documented per-model reasoning effort choices and compatibility aliases.
- Add WARN (default), STRICT, and PASSTHROUGH effort policies to typed and keyword calls, including middleware and scripted clients. Preserve alias inputs on the provider request.
- Validate endpoint capability overrides and select fallback models without changing the requested effort. Preserve explicit legacy capability overrides and selected endpoint/model bindings.
- Preserve Anthropic thinking budgets when merging controls, and reject conflicting effort, thinking, or wire-model values.
- Add runnable reasoning examples and sanitized live observations. Cache the shared Qwen tokenizer and synchronize lazy tokenizer initialization.

### Compatibility

Existing entry points remain available and omitted effort retains the provider default. Conflicting controls now raise even under WARN/PASSTHROUGH; last-write-wins overrides are no longer accepted. Wrappers that supply default enabled thinking and separately pass disabled thinking must reconcile those values before calling. GLM-5.2 none/minimal off behavior remains unconfirmed in live observations; explicit disabled thinking succeeded.
