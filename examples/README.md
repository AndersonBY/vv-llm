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
```

`contract_json.py` is offline. `registry_fallback.py` uses scripted clients and
is also offline. `multimodal.py` reads `VV_LLM_IMAGE_URL`; the default URL is a
placeholder, so set it to an image URL accessible to the selected provider.
