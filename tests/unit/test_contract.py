import shutil
import runpy
import sys
from pathlib import Path

import pytest

import vv_llm.contract as contract_api
from vv_llm.contract import CONSUMER_LOCK_SHA256, ContractIntegrityError
from vv_llm.contract import contract_info, load_catalog, load_fixture, load_schema, verify_contract
from vv_llm.types import defaults


BACKENDS = (
    "anthropic",
    "baichuan",
    "deepseek",
    "ernie",
    "gemini",
    "groq",
    "minimax",
    "mistral",
    "moonshot",
    "openai",
    "qwen",
    "stepfun",
    "xai",
    "xiaomi",
    "yi",
    "zhipuai",
)


def test_vendored_contract_matches_consumer_lock():
    info = contract_info()
    result = verify_contract()

    assert result.ok, result.mismatches
    assert result.artifact_count == 14
    assert info.contract_version == "1.0.0"
    assert info.schema_version == 2
    assert info.fixture_version == 2
    assert info.catalog_revision == 1
    assert info.consumer_lock_sha256 == CONSUMER_LOCK_SHA256


def test_runtime_defaults_are_exactly_the_locked_catalog():
    catalog = load_catalog()
    for backend in BACKENDS:
        assert getattr(defaults, f"{backend.upper()}_DEFAULT_MODEL") == catalog["default_models"][backend]
        assert getattr(defaults, f"{backend.upper()}_MODELS") == catalog["backends"][backend]["models"]


def test_contract_loaders_return_fresh_json_values():
    schema = load_schema("chat-response.v1.schema.json")
    fixture = load_fixture("openai-compatible.v2.json")
    schema["title"] = "caller mutation"
    fixture["request_case"]["canonical_request"]["model"] = "caller mutation"

    assert load_schema("chat-response.v1.schema.json")["title"] != "caller mutation"
    assert load_fixture("openai-compatible.v2.json")["request_case"]["canonical_request"]["model"] != "caller mutation"


def test_runtime_rejects_tampered_consumer_lock(monkeypatch):
    original_resource = contract_api._resource

    class TamperedResource:
        def read_bytes(self):
            return original_resource("consumer-lock.v1.json").read_bytes() + b"tampered"

    def resource(name):
        if name == "consumer-lock.v1.json":
            return TamperedResource()
        return original_resource(name)

    monkeypatch.setattr(contract_api, "_resource", resource)
    result = verify_contract()

    assert not result.ok
    assert "consumer lock SHA-256 mismatch" in result.mismatches[0]
    with pytest.raises(ContractIntegrityError, match="consumer lock SHA-256 mismatch"):
        contract_api.assert_contract_integrity()


def test_sync_checker_rejects_tampered_consumer_lock(tmp_path):
    repository_root = Path(__file__).parents[2]
    source = tmp_path / "vv-llm-contract"
    shutil.copytree(repository_root / "src" / "vv_llm" / "_contract" / "v1_0_0", source)
    lock = source / "consumer-lock.v1.json"
    lock.write_bytes(lock.read_bytes() + b"tampered")

    sync_contract = runpy.run_path(str(repository_root / "scripts" / "sync_contract.py"))
    with pytest.raises(ValueError, match="consumer lock SHA-256 mismatch"):
        sync_contract["_lock_at"](source)


@pytest.mark.parametrize(
    "relative",
    [
        "../escape.json",
        "schemas\\escape.json",
        "/absolute.json",
        "schemas/./escape.json",
        "catalog//extra.json",
    ],
)
def test_sync_checker_rejects_unsafe_artifact_paths(relative):
    repository_root = Path(__file__).parents[2]
    sync_contract = runpy.run_path(str(repository_root / "scripts" / "sync_contract.py"))

    with pytest.raises(ValueError, match="unsafe artifact path"):
        sync_contract["_locked_paths"]({"artifacts": {relative: "0" * 64}})


def test_sync_checker_rejects_unlocked_vendor_files(tmp_path):
    repository_root = Path(__file__).parents[2]
    source = tmp_path / "contract"
    shutil.copytree(repository_root / "src" / "vv_llm" / "_contract" / "v1_0_0", source)
    (source / "unexpected.json").write_text("{}", encoding="utf-8")
    sync_contract = runpy.run_path(str(repository_root / "scripts" / "sync_contract.py"))
    lock = sync_contract["_lock_at"](source)

    problems = sync_contract["_check_tree"](source, lock)

    assert any("unexpected file unexpected.json" in problem for problem in problems)


def test_check_with_source_compares_vendor_and_source(tmp_path):
    repository_root = Path(__file__).parents[2]
    vendor = repository_root / "src" / "vv_llm" / "_contract" / "v1_0_0"
    source = tmp_path / "contract"
    shutil.copytree(vendor, source)
    sync_contract = runpy.run_path(str(repository_root / "scripts" / "sync_contract.py"))

    assert sync_contract["check"](vendor, source) == 0

    artifact = source / "catalog" / "default-chat-catalog.json"
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    assert sync_contract["check"](vendor, source) == 1


def test_contract_sync_requires_explicit_source(monkeypatch, capsys):
    repository_root = Path(__file__).parents[2]
    sync_contract = runpy.run_path(str(repository_root / "scripts" / "sync_contract.py"))
    monkeypatch.delenv("VV_LLM_CONTRACT_SOURCE", raising=False)
    monkeypatch.setattr(sys, "argv", ["sync_contract.py"])

    assert sync_contract["main"]() == 2
    assert "pass --source PATH or set VV_LLM_CONTRACT_SOURCE" in capsys.readouterr().err


def test_contract_check_is_vendor_only_without_explicit_source(monkeypatch):
    repository_root = Path(__file__).parents[2]
    sync_contract = runpy.run_path(str(repository_root / "scripts" / "sync_contract.py"))
    monkeypatch.setenv("VV_LLM_CONTRACT_SOURCE", str(Path("does-not-exist")))
    monkeypatch.setattr(sys, "argv", ["sync_contract.py", "--check"])

    assert sync_contract["main"]() == 0
