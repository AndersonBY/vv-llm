"""Send one configured request; omit --effort to preserve the provider default."""

import argparse
import sys

from vv_llm import CapabilityPolicy, ChatRequest, ChatRequestOptions, ThinkingMode, ThinkingPreference

from common import load_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effort", help="Model-specific choice or documented compatibility input")
    parser.add_argument("--thinking", choices=("default", "enabled", "disabled"), default="default")
    parser.add_argument("--policy", choices=[policy.value for policy in CapabilityPolicy], default="strict")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    client = load_client()
    print("model:", client.model)
    print("choices:", client.capabilities.reasoning_efforts)
    print("aliases:", client.capabilities.reasoning_effort_aliases or {})
    print("requested effort:", args.effort if args.effort is not None else "(provider default)")
    response = client.create(
        ChatRequest(
            model=client.model,
            messages=[{"role": "user", "content": "Compute 37 * 19 and answer briefly."}],
            stream=args.stream,
            skip_cutoff=True,
            options=ChatRequestOptions(
                reasoning_effort=args.effort,
                thinking=ThinkingPreference(mode=ThinkingMode(args.thinking)),
                max_tokens=256,
                timeout=30,
            ),
        ),
        capability_policy=CapabilityPolicy(args.policy),
    )
    if args.stream:
        reasoning_present = False
        for chunk in response:
            reasoning_present = reasoning_present or bool(chunk.reasoning_content)
            print(chunk.content or "", end="", flush=True)
        print()
    else:
        reasoning_present = bool(response.reasoning_content)
        print(response.content or "")
    print("reasoning present:", reasoning_present)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Provider error bodies and configuration tracebacks can contain credentials.
        print(f"Failed: {type(error).__name__}; HTTP status: {getattr(error, 'status_code', None)}", file=sys.stderr)
        raise SystemExit(1) from None
