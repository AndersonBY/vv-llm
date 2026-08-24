#!/usr/bin/env python3
"""Sync or validate the vendored vv-llm contract using only the stdlib.

``--check`` validates the package's vendored contract tree and consumer lock.
Passing ``--source`` with ``--check`` additionally compares that explicit
source tree with the vendor.  Syncing requires an explicit ``--source`` directory or the
``VV_LLM_CONTRACT_SOURCE`` environment variable; no repository layout is
assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "src" / "vv_llm" / "_contract" / "v1_0_1"
META_FILES = ("manifest.json", "checksums.sha256", "consumer-lock.v1.json")
CONTRACT_VERSION = "1.0.1"
CONSUMER_LOCK_SHA256 = "4b63dfb29d28212a7e591dad4ccaabdf0ad29940e3eaa80176a59c59b774f0cb"
ARTIFACT_ROOTS = frozenset({"catalog", "fixtures", "schemas"})
ALLOWED_EXTRA_FILES = frozenset({"__init__.py"})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_paths(lock: dict[str, Any]) -> list[str]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not all(isinstance(path, str) and isinstance(value, str) for path, value in artifacts.items()):
        raise ValueError("consumer lock artifacts must map relative paths to SHA-256 strings")
    for relative, digest in artifacts.items():
        _validate_artifact_path(relative)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"consumer lock has invalid SHA-256 for {relative!r}")
    return [*META_FILES, *sorted(artifacts)]


def _validate_artifact_path(relative: str) -> str:
    """Validate the lock's paths before using them below a package root."""

    if not relative or "\\" in relative:
        raise ValueError(f"unsafe artifact path {relative!r}")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or relative != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) != 2
        or path.parts[0] not in ARTIFACT_ROOTS
        or not path.parts[1].endswith(".json")
    ):
        raise ValueError(f"unsafe artifact path {relative!r}")
    return relative


def _actual_files(root: Path) -> set[str]:
    """Return package files, ignoring interpreter-generated cache files."""

    if not root.is_dir():
        return set()
    files: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.add(path.relative_to(root).as_posix())
    return files


def _check_tree(
    root: Path,
    lock: dict[str, Any],
    *,
    reject_extra_files: bool = True,
) -> list[str]:
    problems: list[str] = []
    if lock.get("contract_version") != CONTRACT_VERSION:
        problems.append(f"{root}: expected contract version {CONTRACT_VERSION!r}")
    try:
        paths = _locked_paths(lock)
    except ValueError as exc:
        return [f"{root}: {exc}"]

    if reject_extra_files:
        expected_files = set(paths) | ALLOWED_EXTRA_FILES
        for relative in sorted(_actual_files(root) - expected_files):
            problems.append(f"{root}: unexpected file {relative}")

    for relative in paths:
        path = root / relative
        if not path.is_file():
            problems.append(f"{root}: missing {relative}")

    manifest = root / "manifest.json"
    checksums = root / "checksums.sha256"
    if manifest.is_file() and _sha256(manifest) != lock.get("manifest_sha256"):
        problems.append(f"{root}: manifest.json SHA-256 does not match lock")
    if checksums.is_file() and _sha256(checksums) != lock.get("checksums_sha256"):
        problems.append(f"{root}: checksums.sha256 SHA-256 does not match lock")

    artifacts = lock.get("artifacts", {})
    for relative, expected in sorted(artifacts.items()):
        path = root / relative
        if path.is_file() and _sha256(path) != expected:
            problems.append(f"{root}: SHA-256 mismatch for {relative}")

    if checksums.is_file():
        try:
            entries: dict[str, str] = {}
            for line in checksums.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    digest, relative = line.split(None, 1)
                    entries[relative] = digest
            if entries != artifacts:
                problems.append(f"{root}: checksums.sha256 entries do not match lock")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            problems.append(f"{root}: cannot parse checksums.sha256: {exc}")

    if manifest.is_file():
        try:
            manifest_value = _read_json(manifest)
            for field in ("contract_version", "schema_version", "fixture_version", "catalog_revision"):
                if manifest_value.get(field) != lock.get(field):
                    problems.append(f"{root}: manifest {field} does not match lock")
            listed: set[str] = set()
            for values in manifest_value.get("artifacts", {}).values():
                if isinstance(values, list):
                    listed.update(str(value) for value in values)
            if listed != set(artifacts):
                problems.append(f"{root}: manifest artifact list does not match lock")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            problems.append(f"{root}: cannot parse manifest.json: {exc}")

    return problems


def _lock_at(root: Path) -> dict[str, Any]:
    lock_path = root / "consumer-lock.v1.json"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    actual = _sha256(lock_path)
    if actual != CONSUMER_LOCK_SHA256:
        raise ValueError(f"{lock_path}: consumer lock SHA-256 mismatch")
    return _read_json(lock_path)


def sync(source: Path, target: Path) -> int:
    try:
        lock = _lock_at(source)
        source_problems = _check_tree(source, lock, reject_extra_files=False)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"contract source is invalid: {exc}", file=sys.stderr)
        return 1
    if source_problems:
        print("contract source failed validation:", file=sys.stderr)
        for problem in source_problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    extras = _actual_files(target) - (set(_locked_paths(lock)) | ALLOWED_EXTRA_FILES)
    if extras:
        print("contract vendor has unexpected files:", file=sys.stderr)
        for relative in sorted(extras):
            print(f"- {target}: unexpected file {relative}", file=sys.stderr)
        return 1

    target.mkdir(parents=True, exist_ok=True)
    for relative in _locked_paths(lock):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    print(f"synced contract {lock['contract_version']} ({len(lock['artifacts'])} artifacts) from {source}")
    return 0


def _compare_trees(source: Path, target: Path, lock: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for relative in _locked_paths(lock):
        source_path = source / relative
        target_path = target / relative
        if source_path.is_file() and target_path.is_file() and source_path.read_bytes() != target_path.read_bytes():
            problems.append(f"source/vendor mismatch for {relative}")
    return problems


def check(target: Path, source: Path | None = None) -> int:
    try:
        lock = _lock_at(target)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"cannot read contract consumer lock: {exc}", file=sys.stderr)
        return 1

    problems = _check_tree(target, lock)
    if source is not None:
        try:
            source_lock = _lock_at(source)
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
            problems.append(f"cannot read contract source lock: {exc}")
        else:
            if source_lock != lock:
                problems.append("source consumer lock does not match the vendored lock")
            problems.extend(_check_tree(source, source_lock, reject_extra_files=False))
            problems.extend(_compare_trees(source, target, lock))
    if problems:
        print("contract validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    if source is None:
        print(f"contract vendor is valid using the vendored lock ({len(lock['artifacts'])} artifacts)")
    else:
        print(f"contract source matches the vendor ({len(lock['artifacts'])} artifacts)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the vendor; with --source, compare source and vendor",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="contract source directory used for syncing or explicit check comparison",
    )
    args = parser.parse_args()
    if args.check:
        # ``--check`` is intentionally offline and vendor-only unless the
        # caller explicitly supplies ``--source`` for a comparison.  The sync
        # environment variable must not silently broaden a check.
        source = args.source.resolve() if args.source is not None else None
        return check(VENDOR_ROOT, source)
    source_value = str(args.source) if args.source is not None else os.getenv("VV_LLM_CONTRACT_SOURCE", "").strip()
    if not source_value:
        print("contract source is required for sync; pass --source PATH or set VV_LLM_CONTRACT_SOURCE", file=sys.stderr)
        return 2
    source = Path(source_value).resolve()
    if not source.is_dir():
        print(f"contract source does not exist: {source}", file=sys.stderr)
        return 1
    return sync(source, VENDOR_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
