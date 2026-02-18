from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _enabled() -> bool:
    return os.getenv("VV_LLM_RUN_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live integration scripts under tests/live.")
    parser.add_argument("scripts", nargs="*", help="Specific live scripts to run (default: all test_*.py).")
    parser.add_argument("--backend", help="Override backend for scripts that support VV_LLM_BACKEND.")
    parser.add_argument("--model", help="Override model for scripts that support VV_LLM_MODEL.")
    parser.add_argument("--preset", help="Override model preset for scripts that support VV_LLM_MODEL_PRESET.")
    parser.add_argument("--allow-empty-keys", action="store_true", help="Allow running scripts even if no usable API keys are found.")
    parser.add_argument("--list", action="store_true", help="List available live scripts and exit.")
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args(sys.argv[1:])

    test_dir = Path(__file__).parent
    all_scripts = sorted(p for p in test_dir.glob("test_*.py") if p.name != "run_live_tests.py")

    if args.list:
        for script in all_scripts:
            print(script.name)
        return 0

    if not _enabled():
        print("Live tests are disabled. Set VV_LLM_RUN_LIVE_TESTS=1 to run.")
        return 1

    requested = args.scripts
    if requested:
        scripts = []
        for item in requested:
            candidate = Path(item)
            if not candidate.is_absolute():
                if (test_dir / candidate).exists():
                    candidate = (test_dir / candidate).resolve()
                else:
                    candidate = (test_dir.parent.parent / candidate).resolve()
            if not candidate.exists():
                print(f"Live test script not found: {item}")
                return 1
            scripts.append(candidate)
    else:
        scripts = all_scripts

    if not scripts:
        print("No live test scripts found.")
        return 0

    env = os.environ.copy()
    if args.backend:
        env["VV_LLM_BACKEND"] = args.backend
    if args.model:
        env["VV_LLM_MODEL"] = args.model
    if args.preset:
        env["VV_LLM_MODEL_PRESET"] = args.preset
    if args.allow_empty_keys:
        env["VV_LLM_ALLOW_EMPTY_KEYS"] = "1"

    if args.backend or args.model or args.preset:
        print(f"[live] overrides: backend={env.get('VV_LLM_BACKEND', '(default)')} model={env.get('VV_LLM_MODEL', '(default)')} preset={env.get('VV_LLM_MODEL_PRESET', '(default)')}")

    failed = []
    for script in scripts:
        print(f"[live] running {script}")
        result = subprocess.run([sys.executable, str(script)], cwd=test_dir.parent.parent, env=env)
        if result.returncode != 0:
            failed.append((script, result.returncode))

    if failed:
        print("\nLive tests finished with failures:")
        for script, code in failed:
            print(f"- {script} (exit={code})")
        return 1

    print("\nAll live test scripts completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
