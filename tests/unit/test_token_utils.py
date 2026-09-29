from vv_llm.chat_clients import utils


def test_qwen_tokenizer_is_shared_across_model_aliases(monkeypatch) -> None:
    calls: list[str] = []

    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(text)

    def fake_get_tokenizer(model: str) -> FakeTokenizer:
        calls.append(model)
        return FakeTokenizer()

    monkeypatch.setattr("qwen_tokenizer.get_tokenizer", fake_get_tokenizer)
    utils._qwen_tokenizer_cache.clear()
    try:
        assert utils.get_token_counts("abc", "qwen-plus", use_token_server_first=False) == 3
        assert utils.get_token_counts("def", "qwen-max", use_token_server_first=False) == 3
        assert calls == ["qwen-plus"]
        assert len(utils._qwen_tokenizer_cache) == 1
    finally:
        utils._qwen_tokenizer_cache.clear()
