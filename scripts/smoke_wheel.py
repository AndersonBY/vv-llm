#!/usr/bin/env python3
"""Install the built wheel into an isolated target and verify contract data."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def latest_wheel() -> Path:
    wheels = sorted((ROOT / "dist").glob("*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise FileNotFoundError("no wheel found under dist/")
    return wheels[-1]


def main() -> int:
    wheel = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_wheel()
    if not wheel.is_file():
        print(f"wheel does not exist: {wheel}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="vv-llm-wheel-smoke-") as temporary:
        target = Path(temporary) / "site-packages"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", str(target), str(wheel)],
            check=True,
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(target)
        smoke = """
from pathlib import Path
import vv_llm
from vv_llm.contract import CONTRACT_VERSION, CONSUMER_LOCK_SHA256, load_catalog, verify_contract
from vv_llm.types import defaults

result = verify_contract()
assert result.ok and result.artifact_count == 14
assert CONTRACT_VERSION == "1.0.0"
assert len(CONSUMER_LOCK_SHA256) == 64
catalog = load_catalog()
for backend, backend_data in catalog["backends"].items():
    default_name = f"{backend.upper()}_DEFAULT_MODEL"
    models_name = f"{backend.upper()}_MODELS"
    assert getattr(defaults, default_name) == catalog["default_models"][backend]
    assert getattr(defaults, models_name), f"{backend} has no public model defaults"
    assert isinstance(backend_data["models"], dict) and backend_data["models"]
assert "site-packages" in str(Path(vv_llm.__file__).resolve())
print("wheel smoke passed")
"""
        subprocess.run([sys.executable, "-c", smoke], check=True, cwd=temporary, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
