import importlib

from vv_llm.types.defaults import ANTHROPIC_MODELS, DEEPSEEK_MODELS, GEMINI_MODELS, MINIMAX_MODELS, MOONSHOT_MODELS, OPENAI_MODELS, QWEN_MODELS, XAI_MODELS, ZHIPUAI_MODELS


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


def test_deepseek_v4_models_expose_configurable_thinking_capability():
    for model_name in ("deepseek-v4-flash", "deepseek-v4-pro"):
        capabilities = DEEPSEEK_MODELS[model_name]["capabilities"]
        assert capabilities["thinking"] == "configurable"
        assert capabilities["structured_output"] == "json_schema"


def test_deepseek_v4_flash_vision_matches_flash_limits_and_accepts_images():
    flash = DEEPSEEK_MODELS["deepseek-v4-flash"]
    vision = DEEPSEEK_MODELS["deepseek-v4-flash-vision-exp"]

    assert vision["id"] == "deepseek-v4-flash-vision-exp"
    assert vision["context_length"] == flash["context_length"]
    assert vision["max_output_tokens"] == flash["max_output_tokens"]
    assert vision["max_image_dimension"] == 8192
    assert vision["function_call_available"] == flash["function_call_available"]
    assert vision["response_format_available"] == flash["response_format_available"]
    assert vision["native_multimodal"] is True
    assert vision["capabilities"]["tools"] == flash["capabilities"]["tools"]
    assert vision["capabilities"]["structured_output"] == flash["capabilities"]["structured_output"]
    assert vision["capabilities"]["thinking"] == flash["capabilities"]["thinking"]
    assert vision["capabilities"]["input_modalities"] == ["text", "image"]


def test_zhipuai_glm_53_defaults_match_documented_capabilities():
    model = ZHIPUAI_MODELS["glm-5.3"]

    assert model["id"] == "glm-5.3"
    assert model["context_length"] == 1_000_000
    assert model["max_output_tokens"] == 128_000
    assert model["function_call_available"] is True
    assert model["response_format_available"] is True
    assert model["native_multimodal"] is False
    assert model["capabilities"] == {
        "tools": True,
        "structured_output": "json_schema",
        "thinking": "always_enabled",
    }


def test_zhipuai_glm_53_flash_defaults_match_documented_capabilities():
    model = ZHIPUAI_MODELS["glm-5.3-flash"]

    assert model["id"] == "glm-5.3-flash"
    assert model["context_length"] == 1_000_000
    assert model["max_output_tokens"] == 128_000
    assert model["function_call_available"] is True
    assert model["response_format_available"] is True
    assert model["native_multimodal"] is True
    assert model["capabilities"] == {
        "tools": True,
        "structured_output": "json_schema",
        "input_modalities": ["text", "image", "video"],
        "thinking": "always_enabled",
    }


def test_qwen_38_models_match_hosted_api_capabilities():
    for model_name in ("qwen3.8-max", "qwen3.8-27b"):
        model = QWEN_MODELS[model_name]

        assert model["id"] == model_name
        assert model["context_length"] == 1_000_000
        assert model["max_output_tokens"] == 131_072
        assert model["function_call_available"] is True
        assert model["response_format_available"] is True
        assert model["native_multimodal"] is True
        assert model["capabilities"]["tools"] is True
        assert model["capabilities"]["structured_output"] == "json_schema"
        assert model["capabilities"]["input_modalities"] == ["text", "image", "video"]
        assert model["capabilities"]["thinking"] == "configurable"

    assert QWEN_MODELS["qwen3.8-max"]["capabilities"]["parallel_tool_calls"] is True


def test_gemini_37_flash_matches_documented_limits_and_capabilities():
    model = GEMINI_MODELS["gemini-3.7-flash"]

    assert model["id"] == "gemini-3.7-flash"
    assert model["context_length"] == 1_048_576
    assert model["max_output_tokens"] == 65_536
    assert model["function_call_available"] is True
    assert model["response_format_available"] is True
    assert model["native_multimodal"] is True
    assert model["capabilities"] == {
        "tools": True,
        "structured_output": "json_schema",
        "input_modalities": ["text", "image", "video", "audio"],
        "thinking": "configurable",
    }


def test_xai_grok_46_matches_documented_capabilities():
    model = XAI_MODELS["grok-4.6"]

    assert model["id"] == "grok-4.6"
    assert model["context_length"] == 500_000
    assert "max_output_tokens" not in model
    assert model["function_call_available"] is True
    assert model["response_format_available"] is True
    assert model["native_multimodal"] is True
    assert model["capabilities"] == {
        "tools": True,
        "structured_output": "json_schema",
        "input_modalities": ["text", "image"],
        "thinking": "configurable",
    }
