from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image

from vv_llm.settings import Settings

from live_common import resolve_bool, run_with_timer
from sample_settings import sample_settings

PROMPT = """画一个飞轮的图:
将合作伙伴、普通用户的动力都调动起来，形成多赢局面：
1）Servers：参考我们的campaign blog，策划campaign，既能强化各社区自身的凝聚力，促进老用户活跃，同时在discordhunt被feature，campaign吸引新用户参与，帮助社区拉新->社区获得新用户和知名度提升
2）Discordhunt：被通知到的servers->愿意参与的servers，吸引被通知/想参与的servers，吸引流量->Discordhunt平台获得内容和流量的冷启动
3）用户：所有关注到活动的用户，既可以upvote、评论，也可以到各个群参加活动，获得优惠、奖品等福利->用户获得奖励和荣誉感

直接画出来，不需要解释。"""


def main() -> None:
    config = Settings(**sample_settings)
    endpoint = config.get_endpoint("gemini-default")
    if not endpoint.api_key or not endpoint.api_base:
        raise RuntimeError("gemini-default endpoint is missing api_key/api_base in sample settings.")

    model = os.environ.get("VV_LLM_MODEL", "").strip() or "gemini-3-pro-image-preview"
    show_image = resolve_bool("VV_LLM_IMAGE_SHOW", False)
    output_file = Path(__file__).with_name("generated_image_gen2.png")

    def _run():
        client = OpenAI(api_key=endpoint.api_key, base_url=endpoint.api_base)
        response = client.images.generate(model=model, prompt=PROMPT, response_format="b64_json", n=1)
        if not response.data:
            raise RuntimeError("No image returned.")
        image = Image.open(BytesIO(base64.b64decode(response.data[0].b64_json)))
        image.save(output_file)
        print(f"[live] image saved to {output_file}")
        if show_image:
            image.show()

    run_with_timer("image_gen2", _run)


if __name__ == "__main__":
    main()
