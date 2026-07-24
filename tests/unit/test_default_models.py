import importlib

from vv_llm.types.defaults import ANTHROPIC_MODELS, MINIMAX_MODELS, MOONSHOT_MODELS, OPENAI_MODELS


def test_defaults_are_split_by_backend_modules():
    defaults_module = importlib.import_module("vv_llm.types.defaults")
    assert hasattr(defaults_module, "__path__")

    for backend_name in (
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
    ):
        importlib.import_module(f"vv_llm.types.defaults.{backend_name}")


def test_minimax_m3_defaults_match_m2_7_except_context_and_multimodal():
    base = MINIMAX_MODELS["MiniMax-M2.7"]
    model = MINIMAX_MODELS["MiniMax-M3"]

    assert model["id"] == "MiniMax-M3"
    assert model["context_length"] == 1_000_000
    assert model["max_output_tokens"] == base["max_output_tokens"]
    assert model["function_call_available"] == base["function_call_available"]
    assert model["response_format_available"] == base["response_format_available"]
    assert model["native_multimodal"] is True


def test_openai_gpt_56_models_are_available():
    for model_name in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        model = OPENAI_MODELS[model_name]
        assert model["id"] == model_name
        assert model["context_length"] == 1_050_000
        assert model["max_output_tokens"] == 128_000
        assert model["function_call_available"] is True
        assert model["response_format_available"] is True
        assert model["native_multimodal"] is True


def test_moonshot_kimi_k3_is_available():
    model = MOONSHOT_MODELS["kimi-k3"]

    assert model["id"] == "kimi-k3"
    assert model["context_length"] == 1_048_576
    assert model["max_output_tokens"] == 1_048_576
    assert model["function_call_available"] is True
    assert model["response_format_available"] is True
    assert model["native_multimodal"] is True


def test_anthropic_claude_opus_5_is_available():
    model = ANTHROPIC_MODELS["claude-opus-5"]

    assert model["id"] == "claude-opus-5"
    assert model["context_length"] == 1_000_000
    assert model["max_output_tokens"] == 128_000
    assert model["function_call_available"] is True
    assert model["response_format_available"] is False
    assert model["native_multimodal"] is True
