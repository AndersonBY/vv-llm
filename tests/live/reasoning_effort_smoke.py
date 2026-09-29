"""Bounded, opt-in live effort matrix. Reports never include credentials or error bodies."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from time import monotonic
import warnings


def has_credentials(endpoint):
    key = endpoint.api_key or ""
    if len(key) >= 12 and not any(value in key.lower() for value in ("test-key", "your_", "placeholder", "${", "sk-...")):
        return True
    credentials = endpoint.credentials or {}
    return any(all(credentials.get(field) for field in fields.split()) for fields in ("access_key secret_key", "client_id client_secret refresh_token", "client_email private_key"))


def run_case(settings, route, effort, probe, args):
    from vv_llm import CapabilityPolicy
    from vv_llm.chat_clients import create_chat_client
    from vv_llm.types.enums import BackendType

    result = {"backend": route["backend"], "model": route["model"], "wire_model": route["wire_model"], "transport": route["transport"], "effort": effort, "probe": probe}
    if args.thinking is not None:
        result["thinking"] = args.thinking
    started = monotonic()
    client = None
    sdk = None
    try:
        client = create_chat_client(backend=BackendType(route["backend"]), model=route["model"], endpoint_id=route["endpoint_id"], random_endpoint=False, settings=settings)
        sdk = client.raw_client
        sdk.max_retries = 0
        sdk.timeout = args.timeout

        class SmokeClient(type(client)):
            @property
            def raw_client(self):
                return sdk

        client.__class__ = SmokeClient
        options = {"max_completion_tokens": args.max_tokens} if route["backend"] == "openai" and route["transport"] != "responses" else {"max_tokens": args.max_tokens}
        if args.thinking is not None:
            options["thinking"] = {"type": args.thinking}
        response = client.create_completion(
            messages=[{"role": "user", "content": "Reply only OK."}],
            reasoning_effort=effort,
            capability_policy=CapabilityPolicy.PASSTHROUGH if probe or args.effort else CapabilityPolicy.STRICT,
            stream=False,
            skip_cutoff=True,
            timeout=args.timeout,
            **options,
        )
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        result.update(
            result="ACCEPTED",
            content_present=bool(response.content),
            reasoning_present=bool(response.reasoning_content),
            output_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
            output_limit_reached=(usage.completion_tokens >= args.max_tokens) if usage is not None else None,
        )
    except Exception as error:
        status = getattr(error, "status_code", None)
        # Inspect privately; provider messages can contain request data or auth details.
        error_text = str(error).lower()
        effort_error = any(field in error_text for field in ("reasoning_effort", "reasoning.effort", "output_config.effort", "vv_invalid_effort"))
        rejected = probe and status in (400, 422) and effort_error
        result.update(result="REJECTED" if rejected else "FAILED", http_status=status, error_type=type(error).__name__, reasoning_parameter_error=effort_error)
    finally:
        if sdk is not None:
            try:
                sdk.close()
            except Exception:
                pass
    result["latency_ms"] = round((monotonic() - started) * 1000)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--backend", action="append", help="Limit providers; repeat to select several")
    parser.add_argument("--model", action="append", help="Limit models using backend:model; repeat as needed")
    parser.add_argument("--effort", action="append", help="Explicit exploratory values, sent with passthrough")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), help="Explicit thinking.type for providers that support it")
    parser.add_argument("--include-catalog", action="store_true", help="Include default-endpoint bindings absent from the local model list")
    parser.add_argument("--aliases", action="store_true", help="Also test declared compatibility aliases")
    parser.add_argument("--invalid-probe", action="store_true", help="Also send an invalid effort to each selected route")
    parser.add_argument("--limit", type=int, default=80, help="Maximum number of paid requests")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.limit, args.workers, args.max_tokens, args.timeout) <= 0:
        parser.error("limits must be positive")
    if args.model and any(":" not in model for model in args.model):
        parser.error("--model requires backend:model")
    os.environ.pop("OPENAI_LOG", None)
    os.environ.pop("ANTHROPIC_LOG", None)
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    from vv_llm.settings import Settings
    from vv_llm.types.enums import BackendType

    try:
        raw = json.loads(args.settings.read_text(encoding="utf-8"))
        # Legacy local/custom blocks in live fixtures are outside the shared Settings shape.
        fields = ("VERSION", "endpoints", "backends", "token_server", "rate_limit", "embedding_backends", "rerank_backends")
        settings = Settings.load_from_dict({field: raw[field] for field in fields if field in raw})
    except Exception as error:
        print(f"FAIL: settings ({type(error).__name__})")
        return 1
    if settings.rate_limit:
        settings.rate_limit.enabled = False
    cases = []
    skipped = []
    seen = set()
    for backend_name, backend_data in raw.get("backends", {}).items():
        if args.backend and backend_name not in args.backend:
            continue
        backend = BackendType(backend_name)
        models = settings.get_backend(backend).models
        for model, config in models.items():
            model_key = f"{backend_name}:{model}"
            if args.model and model_key not in args.model:
                continue
            if not args.include_catalog and model not in backend_data.get("models", {}):
                continue
            if not config.enabled:
                continue
            for binding in config.endpoints:
                if isinstance(binding, dict) and not binding.get("enabled", True):
                    continue
                endpoint_id = binding if isinstance(binding, str) else binding["endpoint_id"]
                try:
                    endpoint = settings.get_endpoint(endpoint_id)
                except ValueError:
                    continue
                if not endpoint.enabled or not has_credentials(endpoint):
                    continue
                transport = "responses" if endpoint.response_api else endpoint.endpoint_type or "default"
                wire_model = binding.get("model_id") or config.id if isinstance(binding, dict) else config.id
                # One route per model/transport; aliases are never assumed equivalent.
                route_key = (model_key, transport, wire_model)
                if route_key in seen:
                    continue
                seen.add(route_key)
                levels = config.capabilities.reasoning_efforts
                if isinstance(binding, dict):
                    levels = (binding.get("capabilities") or {}).get("reasoning_efforts", levels)
                aliases = config.capabilities.reasoning_effort_aliases or {}
                if isinstance(binding, dict):
                    aliases = (binding.get("capabilities") or {}).get("reasoning_effort_aliases", aliases) or {}
                if args.aliases and levels:
                    levels = [*levels, *(alias for alias, target in aliases.items() if target in levels)]
                levels = args.effort or levels
                if not levels:
                    skipped.append({"backend": backend_name, "model": model, "transport": transport, "reason": "unknown_efforts" if levels is None else "unsupported_efforts"})
                    continue
                route = {"backend": backend_name, "model": model, "wire_model": wire_model, "endpoint_id": endpoint_id, "transport": transport}
                cases.extend((route, effort, False) for effort in levels)
                if args.invalid_probe:
                    cases.append((route, "vv_invalid_effort", True))
    selected = cases[: args.limit]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, settings, *case, args) for case in selected]
        for future in futures:
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "python",
        "prompt": "Reply only OK.",
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "configured_timeout_seconds": args.timeout,
        "request_limit": args.limit,
        "truncated_cases": len(cases) - len(selected),
        "results": results,
        "skipped": skipped,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not selected:
        print("SKIP: no selected credentialed route with declared efforts")
    counts = {status: sum(item["result"] == status for item in results) for status in ("ACCEPTED", "REJECTED", "FAILED")}
    print(json.dumps({"summary": counts, "requests": len(results), "truncated": report["truncated_cases"]}))
    return int(any(item["result"] == "FAILED" or item["probe"] and item["result"] != "REJECTED" for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
