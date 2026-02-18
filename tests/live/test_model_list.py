from __future__ import annotations

from openai import AuthenticationError

from v_llm.settings import Settings

from sample_settings import sample_settings


def main() -> None:
    dev_settings = Settings(**sample_settings)
    test_endpoint = dev_settings.get_endpoint("siliconflow")
    temp_settings = Settings(VERSION="2", backends={}, endpoints=[test_endpoint.model_dump()])
    print(f"Temp settings endpoints length: {len(temp_settings.endpoints)}")
    try:
        print(test_endpoint.model_list())
    except AuthenticationError as exc:
        print(f"[live] model list request failed: {exc}")


if __name__ == "__main__":
    main()
