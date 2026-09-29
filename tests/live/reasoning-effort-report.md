# Reasoning effort live observations

Recorded on 2026-09-28 and 2026-09-29 (UTC), using locally configured real credentials.
No key values, endpoint addresses, endpoint IDs, response text or provider error bodies are included.

## Matrix results

The Python matrices recorded 221 requests across 35 model/transport routes: 164 ACCEPTED, 31 REJECTED and 26 FAILED. The ACCEPTED count includes one invalid-value probe accepted by Kimi K3. Failures are retained, including the initial two Fable 5 HTTP 503 responses that succeeded on a later sequential retry. Counts describe recorded attempts, not distinct supported levels.

Each request asks “Reply only OK.” and permits 256 output tokens. Accepted responses reaching that limit remain acceptance evidence, not complete response evidence. The SDK timeout is 25 seconds per request; actual wall-clock time can be greater (roughly 170 seconds on these Gemini routes). No intensity benchmark is implied.

| Provider | Model | Transport | Accepted inputs | Invalid probe | Failed attempts |
| --- | --- | --- | --- | --- | --- |
| gemini | `gemini-2.5-pro` | default | `minimal`, `low`, `medium`, `high` | REJECTED | — |
| gemini | `gemini-2.5-flash` | default | `none`, `minimal`, `low`, `medium`, `high` | REJECTED | — |
| gemini | `gemini-2.5-flash-lite` | default | — | REJECTED | `none`: 404; `minimal`: 404; `low`: 404; `medium`: 404; `high`: 404 |
| gemini | `gemini-3-flash` | default | `minimal`, `low`, `medium`, `high` | REJECTED | — |
| gemini | `gemini-3.1-pro-preview` | default | `low`, `medium`, `high` | REJECTED | `minimal`: 400 |
| gemini | `gemini-3.1-flash-lite` | default | `minimal`, `low`, `medium`, `high` | REJECTED | — |
| openai | `o3-mini` | openai_azure | `low`, `medium`, `high` | REJECTED | — |
| openai | `o4-mini` | openai_azure | `low`, `medium`, `high` | REJECTED | — |
| openai | `gpt-5` | openai_azure | `minimal`, `low`, `medium`, `high` | REJECTED | — |
| openai | `gpt-5-mini` | openai_azure | `minimal`, `low`, `medium`, `high` | REJECTED | — |
| openai | `gpt-5-nano` | openai_azure | `minimal`, `low`, `medium`, `high` | REJECTED | — |
| openai | `gpt-5-pro` | responses | `high` | REJECTED | — |
| openai | `gpt-5.1` | openai_azure | `none`, `low`, `medium`, `high` | REJECTED | — |
| openai | `gpt-5.2` | openai_azure | `none`, `low`, `medium`, `high`, `xhigh` | REJECTED | — |
| openai | `gpt-5.3-codex` | openai_azure | — | FAILED (unverified) | `low`: 400; `medium`: 400; `high`: 400; `xhigh`: 400; `vv_invalid_effort`: 400 |
| openai | `gpt-5.4` | openai_azure | `none`, `low`, `medium`, `high`, `xhigh` | REJECTED | — |
| openai | `gpt-5.4-pro` | openai_azure | — | FAILED (unverified) | `medium`: 404; `high`: 404; `xhigh`: 404; `vv_invalid_effort`: 404 |
| deepseek | `deepseek-v4-pro` | default | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra` | REJECTED | — |
| anthropic | `claude-opus-4-6` | anthropic_bedrock | `low`, `medium`, `high`, `max` | REJECTED | — |
| anthropic | `claude-sonnet-4-6` | anthropic_bedrock | `low`, `medium`, `high` | REJECTED | — |
| anthropic | `claude-opus-4-7` | anthropic_bedrock | `low`, `medium`, `high`, `xhigh`, `max` | REJECTED | — |
| anthropic | `claude-opus-4-8` | anthropic_bedrock | `low`, `medium`, `high` | Not run | — |
| xai | `grok-4.6` | default | `xhigh` | FAILED (unverified) | `low`: APITimeoutError; `medium`: APITimeoutError; `high`: APITimeoutError; `vv_invalid_effort`: 400 |
| mistral | `mistral-small-latest` | default | — | FAILED (unverified) | `none`: APIConnectionError; `high`: 401; `vv_invalid_effort`: 401 |
| deepseek | `deepseek-flash` | default | `none`, `low`, `high`, `max`, `minimal`, `medium`, `xhigh`, `ultra` | REJECTED | — |
| qwen | `qwen3.8-max` | default | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | REJECTED | — |
| qwen | `qwen3.8-27b` | default | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | REJECTED | — |
| zhipuai | `glm-5.2` | default | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | REJECTED | — |
| zhipuai | `glm-5.3` | default | `low`, `high`, `max` | REJECTED | — |
| zhipuai | `glm-5.3-flash` | default | `low`, `high`, `max` | FAILED (unverified) | `vv_invalid_effort`: 400 |
| moonshot | `kimi-k3` | default | `low`, `high`, `max` | ACCEPTED (not validated) | — |
| anthropic | `claude-fable-5` | anthropic_bedrock | `medium`, `high`, `xhigh`, `low`, `max` | REJECTED | `low`: 503; `max`: 503 |
| anthropic | `claude-sonnet-5` | anthropic_bedrock | `low`, `medium`, `high`, `xhigh`, `max` | REJECTED | — |
| anthropic | `claude-opus-5` | anthropic_bedrock | `low`, `medium`, `high`, `xhigh`, `max` | REJECTED | — |
| deepseek | `deepseek-v4.1-flash` | default | `none`, `low`, `high`, `max`, `minimal`, `medium`, `xhigh`, `ultra` | REJECTED | — |

## Effective behavior and limitations

- DeepSeek Flash and V4 Pro accepted low/high/max, none, and all four documented aliases. Each rejected the invalid probe with HTTP 422. None returned no reasoning content; the other inputs returned reasoning content. The compatibility mappings come from official documentation, not token-count comparisons.
- The `deepseek-v4.1-flash` catalog alias was tested using the official request ID `deepseek-flash` via an explicit endpoint binding. Its eight declared choices/aliases succeeded and the invalid probe was rejected with HTTP 422. The literal wire ID `deepseek-v4.1-flash` was not sent.
- Qwen 3.8 Max/27B accepted all declared choices/aliases and rejected invalid values. None produced no reasoning content.
- GLM 5.2 accepted all documented choices/aliases and rejected the invalid value. Repeating the seven documented inputs with explicit `thinking.type=enabled` on the ordinary BigModel API still returned reasoning content for none/minimal; the documented effort-based off behavior remains unconfirmed. Omitted effort succeeded both with omitted thinking and with explicit enabled thinking; these calls do not independently establish the server default intensity. Explicit `thinking.type=disabled` returned no reasoning content with both omitted effort and max. GLM 5.3 and GLM 5.3 Flash accepted low/high/max with thinking enabled. GLM 5.3 explicitly rejected the invalid value; GLM 5.3 Flash returned HTTP 400 without an identified effort field, so that probe remains FAILED rather than a confirmed effort rejection.
- Kimi K3 accepted all documented values and also the invalid value. This cannot establish that effort was honored.
- Claude 4.6/4.7/4.8 and Claude 5 were exercised through Bedrock Messages in Python. Fable 5 low/max initially returned HTTP 503; a sequential retry accepted both. Sonnet 5 and Opus 5 accepted all five documented values; all three rejected invalid values. Thinking was not separately enabled in these calls, so absence of returned reasoning does not establish effort impact.
- OpenAI o3-mini/o4-mini and GPT-5/mini/nano/5.1/5.2/5.4 accepted the declared values through configured Azure routes. GPT-5 Pro accepted high through Responses. Some reasoning responses exhausted the 256-token limit. GPT-5.3 Codex was incorrectly bound to Chat Completions locally (HTTP 400); GPT-5.4 Pro was unavailable on the configured Azure route (HTTP 404). These failures do not alter documented Responses-only capabilities.
- Gemini 2.5 Pro/Flash, 3 Flash, 3.1 Pro and 3.1 Flash-Lite accepted their tested canonical choices. The Gemini 3.1 Pro minimal input returned HTTP 400; a targeted diagnostic confirmed that the error mentions minimal and “not supported”. This conflicts with the official compatibility table, so the alias is excluded from the default catalog pending verification. Gemini 2.5 Flash-Lite was unavailable on this route (HTTP 404). Thinking summaries were not requested.
- Grok 4.6 accepted xhigh; low/medium/high timed out. The invalid input returned HTTP 400 without an identified effort-field error. Mistral was not verified because the configured key returned HTTP 401 and a connection failure occurred.
- Supported catalog entries not exercised with usable routes: `openai:o3`, `openai:gpt-5.5`, `openai:gpt-5.6-sol`, `openai:gpt-5.6-terra`, `openai:gpt-5.6-luna`, `openai:gpt-6-astra`. Unknown entries were not sent as declared-capability tests.

## Cross-runtime checks

- TypeScript: after removing local capability copies, DeepSeek Flash xhigh inherited catalog metadata and succeeded in strict completion/streaming. Xhigh and none both succeeded in completion and streaming, with strict capability validation. Xhigh returned reasoning content; none did not.
- Rust: the ignored `live_deepseek_reasoning_effort_alias_and_off` test passed both none and xhigh through resolved settings and strict validation.
- These representative requests supplement the Python matrix; they are not included in the matrix counters above.

## Reproduction

```bash
python tests/live/reasoning_effort_smoke.py --settings /secure/path/llm_settings.json \
  --model deepseek:deepseek-flash --model deepseek:deepseek-v4-pro \
  --aliases --invalid-probe --limit 18 --report /safe/path/report.json
```

Explicit provider thinking controls can be included with `--thinking enabled` or
`--thinking disabled`; the requested type is recorded with each observation.
GLM-5.2 default/control observations use null effort to identify an omitted value.

The targeted Gemini diagnostic is separate from the matrix counters above.

The selected private file must bind the exact model ID to the intended provider.
`--include-catalog` includes models supplied by the default endpoint; `--effort VALUE`
performs explicit exploratory calls with passthrough. A report stores sanitized
status/usage flags and distinguishes an effort-specific rejection from unrelated
network, authentication, model availability or transport failures.

## Raw sanitized observations

- [configured.json](reports/configured.json): 110 requests; 3 cases outside the request cap.
- [native-extra.json](reports/native-extra.json): 26 requests; 0 cases outside the request cap.
- [new-models.json](reports/new-models.json): 36 requests; 0 cases outside the request cap.
- [claude5.json](reports/claude5.json): 18 requests; 0 cases outside the request cap.
- [claude5-retry.json](reports/claude5-retry.json): 2 requests; 0 cases outside the request cap.
- [cross-runtime.json](reports/cross-runtime.json): six representative TypeScript/Rust observations.
- [deepseek-v41-flash.json](reports/deepseek-v41-flash.json): nine requests for the V4.1 Flash catalog alias bound to the official Flash request ID.
- [zhipu-thinking-enabled.json](reports/zhipu-thinking-enabled.json): 16 requests for GLM-5.2/5.3/5.3-FLASH with explicit enabled thinking; 13 accepted, two confirmed effort rejections and one unclassified HTTP 400.
- [zhipu-thinking-controls.json](reports/zhipu-thinking-controls.json): four accepted GLM-5.2 default/enabled/disabled control requests.
