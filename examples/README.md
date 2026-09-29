# vv-llm Examples

The examples use the typed `ChatRequest` API. Network examples read settings
from `VV_LLM_SETTINGS_JSON`; choose the configured backend and model with:

```powershell
$env:VV_LLM_SETTINGS_JSON = 'C:\path\to\llm_settings.json'
$env:VV_LLM_BACKEND = 'openai'
$env:VV_LLM_MODEL = 'gpt-4o-mini'
```

`VV_LLM_MODEL` may be omitted to use the package default for the selected
backend. `llm_settings.example.json` is a minimal OpenAI-compatible starting
point; replace its placeholder key and endpoint details.

Run examples from the repository root:

```powershell
pdm run python examples/basic_chat.py
pdm run python examples/streaming.py
pdm run python examples/tools.py
pdm run python examples/multimodal.py
pdm run python examples/contract_json.py
```

Additional focused examples are available:

```powershell
pdm run python examples/async_streaming.py
pdm run python examples/typed_thinking.py
pdm run python examples/middleware_metadata.py
pdm run python examples/registry_fallback.py
pdm run python examples/reasoning_capabilities.py
pdm run python examples/reasoning_effort.py --help
```

`contract_json.py` is offline. `registry_fallback.py` uses scripted clients and
is also offline. `multimodal.py` reads `VV_LLM_IMAGE_URL`; the default URL is a
placeholder, so set it to an image URL accessible to the selected provider.

Custom transports use HTTPX2 clients: pass `httpx2.Client` to sync
`create_chat_client(..., http_client=...)` and `httpx2.AsyncClient` to
`create_async_chat_client(..., http_client=...)`. Legacy `httpx` clients are
rejected. Endpoint-level `proxy` settings let vv-llm construct the HTTPX2
client automatically.

## Reasoning effort

Run the capability example without settings or keys:

```powershell
pdm run python examples/reasoning_capabilities.py
```

It reads the vendored catalog for DeepSeek Flash/V4.1 Flash and GLM-5.2/5.3/5.3-FLASH,
prints effective choices separately from compatibility inputs, and checks STRICT,
WARN and PASSTHROUGH. The scripted fallback skips GLM-5.3 for xhigh, selects
GLM-5.2 and preserves xhigh. No requests are sent.

The network example sends one request per invocation. The settings file must bind
the exact selected model to a credentialed endpoint. It prints
`client.capabilities.reasoning_efforts` and `reasoning_effort_aliases` before sending;
these include applicable binding overrides, whereas the offline example shows
package defaults. The example defaults to STRICT; the library still defaults to WARN.

| Option | Behavior |
| --- | --- |
| `--effort VALUE` | Send a model-specific choice or documented alias; omission uses the provider default. The string none is an explicit value, not omission. |
| `--thinking default/enabled/disabled` | Preserve the thinking default or explicitly select the separate thinking control. |
| `--policy strict/warn/passthrough` | Reject locally, warn and continue, or skip model-support validation. Provider parameter conflicts still fail. |
| `--stream` | Consume streaming output with the same effort validation. |

With a configured DeepSeek Flash binding:

```powershell
$env:VV_LLM_BACKEND = 'deepseek'
$env:VV_LLM_MODEL = 'deepseek-flash'
pdm run python examples/reasoning_effort.py
pdm run python examples/reasoning_effort.py --effort high
pdm run python examples/reasoning_effort.py --effort xhigh --stream
```

The last call preserves xhigh on the wire. DeepSeek's documented effective target
is high; aliases are not additional intensities in a model selector. Display an
included none as a separate off control. Unknown choices (null/omitted) and
unsupported effort (an empty list) must not be treated as unrestricted support.

With a configured GLM-5.2 binding, explicit thinking disable uses:

```powershell
$env:VV_LLM_BACKEND = 'zhipuai'
$env:VV_LLM_MODEL = 'glm-5.2'
pdm run python examples/reasoning_effort.py --thinking disabled
```

GLM-5.2 none/minimal returned reasoning content in the recorded live checks;
explicit disabled thinking did not. GLM-5.3/FLASH cannot disable thinking and
accept only low/high/max on the ordinary API. A successful response does not prove
that effort changed model behavior; see [the sanitized observations](../tests/live/reasoning-effort-report.md).
