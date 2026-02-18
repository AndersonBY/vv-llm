# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
from vv_llm.settings import settings
from vv_llm.chat_clients import (
    BackendType,
    format_messages,
    create_chat_client,
)

from live_common import load_live_settings, resolve_backend_model


load_live_settings(settings)

system_prompt = "# 角色\n你是播客制作人，你的主要任务是帮助用户整理各类新闻、报告、文档等信息，将其最终呈现为播客的形式。\n\n# 工作流程\n1. 用户会向你提出了解不同的信息需求（如 Bilibili 视频内容、科技新闻、Arxiv 论文等），你需要根据用户需求决定调用的工作流。\n  1.1 论文检索优先使用工作流【search_arxiv_papers】\n2. 在用户了解某个信息后你需要向用户明确是否要将该信息加入到最终的播客内容里。\n3. 每轮对话完成后都需要向用户提问是否还需要了解更多信息还是可以开始制作播客。\n4. 当用户决定制作播客时，向用户询问最终播客稿件的风格，如严谨、幽默等。\n5. 当用户已经明确稿件风格后先根据要求生成一份文字版的稿件，然后询问用户是否满意，如果用户回复满意则调用工作流【text_to_speech_conversion】进行音频生成。\n\n# 要求\n- 调用工作流时参数名称务必准确不能写错\n- 生成回复时必须始终和用户的语言一致！\n- 如果工作流的运行结果与用户语言不一致，则务必翻译后回复用户。"

messages = [
    {
        "role": "user",
        "content": "\n# vectorvein.workflow 模块源代码结构\n\n",
    },
    {
        "role": "assistant",
        "content": "我需要设计一个工作流来满足您的需求：允许上传多个小票图片，识别内容，用AI分析提取核心信息，并整合到一张表格中生成下载链接。\n\n让我先了解一下关键节点的具体实现方式。",
        "tool_calls": [
            {
                "id": "toolu_bdrk_01DxPc2FBUvg3z72txPRrweo",
                "function": {"arguments": '{"node": "FileUpload"}', "name": "get_node_documentation"},
                "type": "function",
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "toolu_bdrk_01DxPc2FBUvg3z72txPRrweo",
        "content": '# FileUpload 文档\n![上传文件](https://vector-vein-cdn.oss-cn-hangzhou.aliyuncs.com/documents/resources/images/nodes/file-processing-upload-file.jpg)\n\n该节点可以在使用界面上提供一个文件选择框，让用户选择一个文件，然后获取临时访问链接作为输出。\n\n### 参数详解\n\n#### 解压缩\n\n如果勾选则对上传的 .zip 压缩包进行解压缩，具体输出格式参考下文。\n\n#### 解压缩后输出格式\n\n- 列表\n  - 压缩包内每个文件（包含子文件夹内的文件）的临时链接作为一个列表元素\n- 字典\n  - 压缩包内每个文件（包含子文件夹内的文件）的临时链接作为一个字典元素，字典的键为文件名\n\n> [!TIP]\n> 示例列表结果：\n> \n> ```json\n> [\n>   "https://xxx.com/file1",\n>   "https://xxx.com/file2",\n>   "https://xxx.com/file3"\n> ]\n> ```\n>\n> 示例字典结果：\n>\n> ```json\n> {\n>   "file1": "https://xxx.com/file1",\n>   "子文件夹1/file2": "https://xxx.com/file2",\n>   "子文件夹2/file3": "https://xxx.com/file3"\n> }\n> ```\n\n### 输出类型\n\n` 字符串 `\n\n### 积分消耗\n\n0 积分/次',
    },
]

if __name__ == "__main__":
    presets = {
        "deepseek-reasoner": (BackendType.DeepSeek, "deepseek-reasoner"),
        "gpt-4o": (BackendType.OpenAI, "gpt-4o"),
        "claude-sonnet-4.6": (BackendType.Anthropic, "claude-sonnet-4-6"),
    }
    backend, model = resolve_backend_model(BackendType.DeepSeek, "deepseek-reasoner", presets=presets)

    model_settings = settings.get_backend(backend=backend).models[model]
    client = create_chat_client(backend=backend, model=model, stream=False)
    formatted_messages = format_messages(
        messages,
        backend=backend,
        native_multimodal=model_settings.native_multimodal,
        function_call_available=model_settings.function_call_available,
    )

    print(formatted_messages)
    response = client.create_completion(messages=formatted_messages, stream=True)
    reasoning_end = False
    for chunk in response:
        if not reasoning_end:
            if chunk.reasoning_content:
                print(chunk.reasoning_content, end="", flush=True)
            else:
                reasoning_end = True
                print("===thinking end===")
        else:
            print(chunk.content, end="", flush=True)
