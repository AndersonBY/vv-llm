# vv-llm Examples

The examples use the current normalized request API while the existing
`create_completion(**kwargs)` API remains supported.

Copy `llm_settings.example.json`, replace the placeholder API key, and set:

```powershell
$env:VV_LLM_SETTINGS_JSON = 'C:\path\to\llm_settings.json'
```

Run an example from the repository root:

```powershell
pdm run python examples/01_typed_thinking.py
pdm run python examples/02_streaming.py
pdm run python examples/03_async_streaming.py
pdm run python examples/04_middleware_metadata.py
```

`05_registry_fallback.py` is deterministic and does not call a real API:

```powershell
pdm run python examples/05_registry_fallback.py
```

The streaming examples use `ChatRequest(stream=True)`. Retry or fallback is
allowed only before the first visible chunk; once output begins, a later error
is returned without replaying the request.
