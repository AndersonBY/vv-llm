from vv_llm.types.defaults import MINIMAX_MODELS


def test_minimax_m3_defaults_match_m2_7_except_context_and_multimodal():
    base = MINIMAX_MODELS["MiniMax-M2.7"]
    model = MINIMAX_MODELS["MiniMax-M3"]

    assert model["id"] == "MiniMax-M3"
    assert model["context_length"] == 1_000_000
    assert model["max_output_tokens"] == base["max_output_tokens"]
    assert model["function_call_available"] == base["function_call_available"]
    assert model["response_format_available"] == base["response_format_available"]
    assert model["native_multimodal"] is True
