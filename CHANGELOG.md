# Changelog

## 0.10.3 - 2026-11-11

- Add `gemini-4-carbon` with a unified 1,048,576-token context, 65,536-token synchronous output limit, text/image/video/audio input, tools, structured output, and low/medium/high reasoning efforts.
- Adopt shared contract 1.3.6 and catalog revision 24.

## 0.10.2 - 2026-11-10

- Adopt shared contract 1.3.5 and catalog revision 23 with `gpt-6.1-terra`: 1,050,000-token context, 128,000-token maximum output, text/image input, structured output, and none/low/medium/high/xhigh/max reasoning efforts.
- Tool calling requires the Responses API; existing runtime transport support is unchanged.

## 0.10.1 - 2026-11-07

- Report `stream_terminated=None` when OpenAI `[DONE]` observation degrades and no other terminal signal is observed, including after normal exhaustion. Reduced observation cannot establish the absence of a terminal signal.
- Preserve `True` for observed terminal signals and `False` for fully observed streams without terminal signals. Keep observer diagnostics and warnings unchanged across sync/async clients, middleware metadata, and fallback replay.
- Document live intermediate evidence and verify degraded observation with real OpenAI SDK SSE parsing over MockTransport.

## 0.10.0 - 2026-11-07

- Preserve raw and normalized provider finish reasons and explicit refusal text in optional response/chunk metadata across sync/async chat clients.
- Expose per-call stream termination signals and iterator exhaustion, including `[DONE]`-only and empty streams, without adding, delaying, or removing chunks or changing provider exceptions.
- Preserve live evidence through middleware, fallback routing, Anthropic native/compatible transports, and the Responses API. Evidence records provider facts; callers retain responsibility for deciding success.
- Guard private SSE observation against SDK interface drift; select finish/refusal evidence by choice index 0 and count exhaustion after caller replay.
- Classify Anthropic context-window exhaustion as length, and await owned SDK stream closure on asynchronous interruption or finalization.
- Verify public finish-evidence exports and local sync/async consumption in the wheel smoke.
- Keep canonical contract 1.3.4 unchanged: runtime metadata is separate from portable response JSON and can use the existing `x_*` extension mechanism at that boundary.

## 0.9.0 - 2026-11-05

- Remove upper bounds on the OpenAI, Anthropic, and HTTPX2 dependencies.
- Require Python 3.11 or newer.
- Construct Responses function tools with the OpenAI SDK's updated typed dictionary.
- Align settings dictionary types with nullable runtime fields and Vertex token cache fields.

## 0.8.5 - 2026-11-01

- Preserve provider tool-call indices across sync/async streaming chunks.
- Keep native Anthropic tool IDs and incremental arguments associated with their
  content-block indices, including interleaved sparse blocks.
- Adopt the locally validated, unpublished shared contract 1.3.4 candidate with
  sparse and reordered streaming tool-call conformance fixtures (no schema change).

## 0.8.4 - 2026-10-24

- Adopt shared contract 1.3.3 and catalog revision 22 with `kimi-k3.1-pro`: 1,048,576-token
  context and maximum output, text/image/video input, tools, strict JSON Schema
  output, always-on thinking, and low/high/max reasoning efforts.

## 0.8.3 - 2026-10-20

- Adopt shared contract 1.3.2 and catalog revision 21 with `MiniMax-M3.1-Flash`,
  a 1M-context multimodal agent model with 512K output, tools, mandatory
  thinking, and five reasoning-effort choices.

## 0.8.2 - 2026-10-18

- Moonshot usage: treat an omitted cache-read count as zero even when
  `prompt_tokens_details` is present (kimi-k3 sends it with only
  `cache_write_tokens`); explicit null or invalid counts stay unknown.
- OpenAI-compatible usage: promote `prompt_tokens_details.cache_write_tokens`
  to `cache_creation_tokens` when no top-level value is reported.

## 0.8.1 - 2026-10-14

- Add `gemini-4-argon` with a unified 1,048,576-token context, 65,536-token
  synchronous output limit, text/image/video/audio input, tools, structured
  output, and low/medium/high reasoning efforts.
- Adopt shared contract 1.3.1 and catalog revision 20.

## 0.8.0 - 2026-10-08

- Add `claude-haiku-5-5` from shared catalog revision 19 with a 1M context
  window, 128K output limit, image input, tools, and five reasoning efforts.

- Add standalone decision clients with predicate, choice, score, and refusal
  responses, independent decision_backends settings, and opt-in live smoke tests.
- Adopt shared contract 1.3.0, catalog revision 19 and decision conformance fixtures.

## 0.7.6 - 2026-10-07

- Omit temperature, top_p, top_k and legacy thinking budgets from Gemini 3+
  requests, including nested provider overrides; preserve Gemini 2.5 behavior.
- Normalize thinkingLevel to thinking_level and reject conflicting controls.
  Budget-only requests use the provider default; select reasoning_effort or
  thinking_level explicitly when an effort level is required.
- Adopt contract 1.2.2, catalog revision 17 with low/medium/high efforts for
  Gemini 3.7 Flash and 3.8 Flash.


## 0.7.5 - 2026-10-03

- Adopt contract 1.2.1, catalog revision 16 with the single public ID `qwen3.8-flash-next`;
  DashScope uses endpoint `model_id` to select `qwen3.8-flash`.
- Keep live test reports as ignored local artifacts and remove tracked report files.

## 0.7.4 - 2026-10-01

- Preserve Responses API `input_tokens_details.cached_tokens` as
  `prompt_tokens_details.cached_tokens` across synchronous, asynchronous,
  streaming, and non-streaming calls. Keep explicit zero distinct from
  missing cache statistics without changing token totals.

## 0.7.3 - 2026-09-30

- Fix the Responses API adapter: assistant tool calls and tool results become
  `function_call` and `function_call_output` input items keyed by `call_id`,
  so a second tool turn no longer fails with `Unknown parameter: input[2].tool_calls`.
- Emit tool-call arguments exactly once: terminal `*.done` events only append
  fragments that were not streamed, and streamed tool calls expose the provider
  `call_id` instead of the `fc_*` item id.

## 0.7.2 - 2026-09-30

- Adopt catalog revision 14 with OpenAI `gpt-6.1-sol`.

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
