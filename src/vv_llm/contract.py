"""Read-only access to the versioned ``vv-llm-contract`` artifacts.

The JSON files are deliberately shipped as data instead of being generated at
import time.  This module is the small, stable boundary used by runtime code
and protocol tests; callers receive a fresh decoded value for every load and
cannot mutate the vendored resources themselves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Final

_PACKAGE = "vv_llm._contract.v1_0_1"
_LOCK_NAME = "consumer-lock.v1.json"
_MANIFEST_NAME = "manifest.json"
_CHECKSUMS_NAME = "checksums.sha256"
_DEFAULT_CATALOG = "default-chat-catalog.json"
CONSUMER_LOCK_SHA256: Final[str] = "3407cc7d398885284f32c453a8e71c6dbb2f40a10eb0cc9f2d21a0a7c7dc6b49"


class ContractIntegrityError(RuntimeError):
    """Raised when the vendored contract does not match its consumer lock."""


@dataclass(frozen=True)
class ContractRevisions:
    """Immutable schema, fixture, and catalog revision numbers."""

    schema_version: int
    fixture_version: int
    catalog_revision: int


@dataclass(frozen=True)
class ContractInfo:
    """Immutable contract version, revisions, and top-level digests."""

    contract_version: str
    schema_version: int
    fixture_version: int
    catalog_revision: int
    manifest_sha256: str
    checksums_sha256: str
    consumer_lock_sha256: str

    @property
    def revisions(self) -> ContractRevisions:
        return ContractRevisions(self.schema_version, self.fixture_version, self.catalog_revision)


@dataclass(frozen=True)
class ContractVerification:
    """Result of checking every locked contract resource."""

    ok: bool
    artifact_count: int
    mismatches: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok


CONTRACT_VERSION: Final[str] = "1.0.1"
SCHEMA_VERSION: Final[int] = 2
FIXTURE_VERSION: Final[int] = 2
CATALOG_REVISION: Final[int] = 2


def _resource(name: str):
    return files(_PACKAGE).joinpath(*name.split("/"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(name: str) -> dict[str, Any]:
    value = json.loads(_resource(name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"contract resource {name!r} must contain a JSON object")
    return value


def _lock() -> dict[str, Any]:
    resource = _resource(_LOCK_NAME)
    lock_bytes = resource.read_bytes()
    actual = _sha256(lock_bytes)
    if actual != CONSUMER_LOCK_SHA256:
        raise ContractIntegrityError(f"consumer lock SHA-256 mismatch: expected {CONSUMER_LOCK_SHA256}, got {actual}")
    value = json.loads(lock_bytes.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"contract resource {_LOCK_NAME!r} must contain a JSON object")
    return value


def _artifact_name(kind: str, name: str | None, default: str | None = None) -> str:
    requested = default if name is None else name
    if not requested:
        raise ValueError(f"a {kind} artifact name is required")
    prefix = f"{kind}/"
    return requested if requested.startswith(prefix) else prefix + requested


def _locked_artifact(kind: str, name: str | None, default: str | None = None) -> str:
    path = _artifact_name(kind, name, default)
    artifacts = _lock().get("artifacts", {})
    if not isinstance(artifacts, dict) or path not in artifacts:
        raise KeyError(f"unknown {kind} contract artifact: {path}")
    return path


def load_artifact(path: str) -> dict[str, Any]:
    """Load a locked JSON artifact by its canonical relative path.

    Only paths present in ``consumer-lock.v1.json`` are accepted.  A new JSON
    object is decoded on each call, so modifying the returned value never
    changes the vendored contract or a later load.
    """

    canonical = path.replace("\\", "/")
    artifacts = _lock().get("artifacts", {})
    if not isinstance(artifacts, dict) or canonical not in artifacts:
        raise KeyError(f"unknown contract artifact: {path}")
    return _read_json(canonical)


def load_schema(name: str) -> dict[str, Any]:
    """Load a locked JSON Schema, accepting a basename or ``schemas/`` path."""

    return load_artifact(_locked_artifact("schemas", name))


def load_fixture(name: str) -> dict[str, Any]:
    """Load a locked protocol/test fixture, accepting a basename or path."""

    return load_artifact(_locked_artifact("fixtures", name))


def load_catalog(name: str = _DEFAULT_CATALOG) -> dict[str, Any]:
    """Load the locked model catalog, accepting its basename or path."""

    return load_artifact(_locked_artifact("catalog", name))


def contract_info() -> ContractInfo:
    """Return immutable version/revision metadata from the consumer lock."""

    lock = _lock()
    return ContractInfo(
        contract_version=str(lock["contract_version"]),
        schema_version=int(lock["schema_version"]),
        fixture_version=int(lock["fixture_version"]),
        catalog_revision=int(lock["catalog_revision"]),
        manifest_sha256=str(lock["manifest_sha256"]),
        checksums_sha256=str(lock["checksums_sha256"]),
        consumer_lock_sha256=CONSUMER_LOCK_SHA256,
    )


def contract_revisions() -> ContractRevisions:
    """Return immutable schema, fixture, and catalog revisions."""

    return contract_info().revisions


def _manifest_mismatches(lock: dict[str, Any], mismatches: list[str]) -> None:
    manifest = _read_json(_MANIFEST_NAME)
    for field in ("contract_version", "schema_version", "fixture_version", "catalog_revision"):
        if manifest.get(field) != lock.get(field):
            mismatches.append(f"manifest {field} does not match consumer lock")

    manifest_paths: set[str] = set()
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, dict):
        for values in artifacts.values():
            if isinstance(values, list):
                manifest_paths.update(str(value) for value in values)
    locked_paths = lock.get("artifacts", {})
    if isinstance(locked_paths, dict) and manifest_paths != set(locked_paths):
        mismatches.append("manifest artifact list does not match consumer lock")


def verify_contract() -> ContractVerification:
    """Check lock, manifest, checksums, and every locked artifact SHA-256."""

    mismatches: list[str] = []
    try:
        lock = _lock()
    except (ContractIntegrityError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return ContractVerification(False, 0, (f"cannot read consumer lock: {exc}",))

    artifacts = lock.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return ContractVerification(False, 0, ("consumer lock artifacts must be an object",))

    manifest_resource = _resource(_MANIFEST_NAME)
    checksums_resource = _resource(_CHECKSUMS_NAME)
    if not manifest_resource.is_file():
        mismatches.append("manifest.json is missing")
    elif _sha256(manifest_resource.read_bytes()) != lock.get("manifest_sha256"):
        mismatches.append("manifest.json SHA-256 does not match consumer lock")
    if not checksums_resource.is_file():
        mismatches.append("checksums.sha256 is missing")
    elif _sha256(checksums_resource.read_bytes()) != lock.get("checksums_sha256"):
        mismatches.append("checksums.sha256 SHA-256 does not match consumer lock")

    for path, expected in sorted(artifacts.items()):
        resource = _resource(str(path))
        if not resource.is_file():
            mismatches.append(f"missing artifact: {path}")
            continue
        actual = _sha256(resource.read_bytes())
        if actual != expected:
            mismatches.append(f"SHA-256 mismatch: {path}")

    try:
        checksums = checksums_resource.read_text(encoding="utf-8")
        checksum_entries = {}
        for line in checksums.splitlines():
            if not line.strip():
                continue
            digest, path = line.split(None, 1)
            checksum_entries[path] = digest
        if checksum_entries != artifacts:
            mismatches.append("checksums.sha256 entries do not match consumer lock")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        mismatches.append(f"cannot parse checksums.sha256: {exc}")

    try:
        _manifest_mismatches(lock, mismatches)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        mismatches.append(f"cannot validate manifest.json: {exc}")

    return ContractVerification(not mismatches, len(artifacts), tuple(mismatches))


def assert_contract_integrity() -> None:
    """Raise :class:`ContractIntegrityError` unless the vendored tree is valid."""

    result = verify_contract()
    if not result.ok:
        detail = "; ".join(result.mismatches)
        raise ContractIntegrityError(f"vendored vv-llm-contract is invalid: {detail}")


__all__ = [
    "CATALOG_REVISION",
    "CONSUMER_LOCK_SHA256",
    "CONTRACT_VERSION",
    "ContractInfo",
    "ContractIntegrityError",
    "ContractRevisions",
    "ContractVerification",
    "FIXTURE_VERSION",
    "SCHEMA_VERSION",
    "assert_contract_integrity",
    "contract_info",
    "contract_revisions",
    "load_artifact",
    "load_catalog",
    "load_fixture",
    "load_schema",
    "verify_contract",
]
