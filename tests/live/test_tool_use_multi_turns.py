# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
import time

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
        "mid": "cea20838433f486dab6f17831b0df084",
        "author_type": "U",
        "content_type": "TXT",
        "status": "S",
        "create_time": 1722415381031,
        "update_time": 1722415381031,
        "metadata": {},
        "content": {"text": "Arxiv 上关于 AI Agent的论文总结成一个播客"},
        "attachments": [],
    },
    {
        "mid": "1749e50508ed483d9d8f5ef17ecd4fa8",
        "author_type": "A",
        "content_type": "WKF",
        "status": "S",
        "create_time": 1722415381031,
        "update_time": 1722415398031,
        "metadata": {
            "record_id": "2c92432544114dc4ab218dd3709a0585",
            "workflow_result": "[{'Title': 'A Tutorial on the Use of Physics-Informed Neural Networks to Compute the Spectrum of Quantum Systems', 'arXiv Link': 'https://arxiv.org/abs/2407.20669', 'arXiv ID': 'arXiv:2407.20669[pdf,other]', 'Abstract': 'Quantum many-body systems are of great interest for many research areas, including physics, biology and chemistry. However, their simulation is extremely challenging, due to the exponential growth of the Hilbert space with the system size, making it exceedingly difficult to parameterize the wave functions of large systems by using exact methods. Neural networks and machine learning in general are a way to face this challenge. For instance, methods like Tensor networks and Neural Quantum States are being investigated as promising tools to obtain the wave function of a quantum mechanical system. In this tutorial, we focus on a particularly promising class of deep learning algorithms. We explain how to construct a Physics-Informed Neural Network (PINN) able to solve the Schrödinger equation for a given potential, by finding its eigenvalues and eigenfunctions. This technique is unsupervised, and utilizes a novel computational method in a manner that is barely explored. PINNs are a deep learning method that exploits Automatic Differentiation to solve Integro-Differential Equations in a mesh-free way. We show how to find both the ground and the excited states. The method discovers the states progressively by starting from the ground state. We explain how to introduce inductive biases in the loss to exploit further knowledge of the physical system. Such additional constraints allow for a faster and more accurate convergence. This technique can then be enhanced by a smart choice of collocation points in order to take advantage of the mesh-free nature of the PINN. The methods are made explicit by applying them to the infinite potential well and the particle in a ring, a challenging problem to be learned by anAIagentdue to the presence of complex-valued eigenfunctions and degenerate states.△ Less'}, {'Title': 'Domain Adaptable PrescriptiveAIAgentfor Enterprise', 'arXiv Link': 'https://arxiv.org/abs/2407.20447', 'arXiv ID': 'arXiv:2407.20447[pdf,other]', 'Abstract': 'Despite advancements in causal inference and prescriptiveAI, its adoption in enterprise settings remains hindered primarily due to its technical complexity. Many users lack the necessary knowledge and appropriate tools to effectively leverage these technologies. This work at the MIT-IBM WatsonAILab focuses on developing the proof-of-conceptagent, PrecAIse, a domain-adaptable conversationalagentequipped with a suite of causal and prescriptive tools to help enterprise users make better business decisions. The objective is to make advanced, novel causal inference and prescriptive tools widely accessible through natural language interactions. The presented Natural Language User Interface (NLUI) enables users with limited expertise in machine learning and data science to harness prescriptive analytics in their decision-making processes without requiring intensive computing resources. We present anagentcapable of function calling, maintaining faithful, interactive, and dynamic conversations, and supporting new domains.△ Less'}, {'Title': 'Appraisal-Guided Proximal Policy Optimization: Modeling Psychological Disorders in Dynamic Grid World', 'arXiv Link': 'https://arxiv.org/abs/2407.20383', 'arXiv ID': 'arXiv:2407.20383[pdf]', 'Abstract': \"The integration of artificial intelligence across multiple domains has emphasized the importance of replicating human-like cognitive processes inAI. By incorporating emotional intelligence intoAIagents, their emotional stability can be evaluated to enhance their resilience and dependability in critical decision-making tasks. In this work, we develop a methodology for modeling psychological disorders using Reinforcement Learning (RL)agents. We utilized Appraisal theory to train RLagentsin a dynamic grid world environment with an Appraisal-Guided Proximal Policy Optimization (AG-PPO) algorithm. Additionally, we investigated numerous reward-shaping strategies to simulate psychological disorders and regulate the behavior of theagents. A comparison of various configurations of the modified PPO algorithm identified variants that simulate Anxiety disorder and Obsessive-Compulsive Disorder (OCD)-like behavior inagents. Furthermore, we compared standard PPO with AG-PPO and its configurations, highlighting the performance improvement in terms of generalization capabilities. Finally, we conducted an analysis of theagents' behavioral patterns in complex test environments to evaluate the associated symptoms corresponding to the psychological disorders. Overall, our work showcases the benefits of the appraisal-guided PPO algorithm over the standard PPO algorithm and the potential to simulate psychological disorders in a controlled artificial environment and evaluate them on RLagents.△ Less\"}, {'Title': 'MindSearch: Mimicking Human Minds Elicits DeepAISearcher', 'arXiv Link': 'https://arxiv.org/abs/2407.20183', 'arXiv ID': 'arXiv:2407.20183[pdf,other]', 'Abstract': 'Information seeking and integration is a complex cognitive task that consumes enormous time and effort. Inspired by the remarkable progress of Large Language Models, recent works attempt to solve this task by combining LLMs and search engines. However, these methods still obtain unsatisfying performance due to three challenges: (1) complex requests often cannot be accurately and completely retrieved by the search engine once (2) corresponding information to be integrated is spread over multiple web pages along with massive noise, and (3) a large number of web pages with long contents may quickly exceed the maximum context length of LLMs. Inspired by the cognitive process when humans solve these problems, we introduce MindSearch to mimic the human minds in web information seeking and integration, which can be instantiated by a simple yet effective LLM-based multi-agentframework. The WebPlanner models the human mind of multi-step information seeking as a dynamic graph construction process: it decomposes the user query into atomic sub-questions as nodes in the graph and progressively extends the graph based on the search result from WebSearcher. Tasked with each sub-question, WebSearcher performs hierarchical information retrieval with search engines and collects valuable information for WebPlanner. The multi-agentdesign of MindSearch enables the whole framework to seek and integrate information parallelly from larger-scale (e.g., more than 300) web pages in 3 minutes, which is worth 3 hours of human effort. MindSearch demonstrates significant improvement in the response quality in terms of depth and breadth, on both close-set and open-set QA problems. Besides, responses from MindSearch based on InternLM2.5-7B are preferable by humans to ChatGPT-Web and Perplexity.ai applications, which implies that MindSearch can already deliver a competitive solution to the proprietaryAIsearch engine.△ Less'}, {'Title': 'Quantum Machine Learning Architecture Search via Deep Reinforcement Learning', 'arXiv Link': 'https://arxiv.org/abs/2407.20147', 'arXiv ID': 'arXiv:2407.20147[pdf,other]', 'Abstract': \"The rapid advancement of quantum computing (QC) and machine learning (ML) has given rise to the burgeoning field of quantum machine learning (QML), aiming to capitalize on the strengths of quantum computing to propel ML forward. Despite its promise, crafting effective QML models necessitates profound expertise to strike a delicate balance between model intricacy and feasibility on Noisy Intermediate-Scale Quantum (NISQ) devices. While complex models offer robust representation capabilities, their extensive circuit depth may impede seamless execution on extant noisy quantum platforms. In this paper, we address this quandary of QML model design by employing deep reinforcement learning to explore proficient QML model architectures tailored for designated supervised learning tasks. Specifically, our methodology involves training an RLagentto devise policies that facilitate the discovery of QML models without predetermined ansatz. Furthermore, we integrate an adaptive mechanism to dynamically adjust the learning objectives, fostering continuous improvement in theagent'slearning process. Through extensive numerical simulations, we illustrate the efficacy of our approach within the realm of classification tasks. Our proposed method successfully identifies VQC architectures capable of achieving high classification accuracy while minimizing gate depth. This pioneering approach not only advances the study ofAI-driven quantum circuit design but also holds significant promise for enhancing performance in the NISQ era.△ Less\"}]",
            "selected_workflow": {
                "tid": "3b0ba398b5de4439b8a22fda7d0d35b4",
                "type": "WorkflowTemplate",
                "brief": '输入搜索关键词和数量，返回在 Arxiv 上检索到的论文基本信息。\n返回一个列表，列表里每个元素结构如下：\n```json\n{\n    "Title": "",\n    "arXiv Link": "",\n    "arXiv ID": "",\n    "Abstract": "",\n}\n```',
                "title": "搜索 Arxiv 论文",
                "params": {"count": 5, "search_text": "AI Agent"},
                "tool_call_id": "call_bF6aM6ChQ1-irt2DLY5Idw",
                "function_name": "search_arxiv_papers",
            },
        },
        "content": {"text": ""},
        "attachments": [],
    },
    {
        "mid": "fe89a143be9e482587e36ca309e25794",
        "author_type": "A",
        "content_type": "TXT",
        "status": "S",
        "create_time": 1722415401031,
        "update_time": 1722415404031,
        "metadata": {"selected_workflow": {}},
        "content": {"text": "好的，我找到了关于AI Agent的一些论文。您希望将哪篇论文的内容加入到最终的播客中呢？"},
        "attachments": [],
    },
    {
        "mid": "c223b530a53b4ef68662afb69018a59b",
        "author_type": "U",
        "content_type": "TXT",
        "status": "S",
        "create_time": 1722415560031,
        "update_time": 1722415560031,
        "metadata": {},
        "content": {"text": "都加入到播客里吧，然后画一个播客封面图吧我看看"},
        "attachments": [],
    },
]


tools = [
    {
        "type": "function",
        "function": {
            "name": "read_web_content",
            "description": "给定网址，输出这个网址的网页正文内容",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string"}, "use_oversea_crawler": {"type": "boolean"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_latest_tech_news",
            "description": "输入要抓取的科技新闻要求，输出相关的科技新闻标题、链接、总结等信息。",
            "parameters": {
                "type": "object",
                "required": ["user_requirements"],
                "properties": {"user_requirements": {"type": "string", "description": "检索新闻的要求"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv_papers",
            "description": '输入搜索关键词和数量，返回在 Arxiv 上检索到的论文基本信息。\n返回一个列表，列表里每个元素结构如下：\n```json\n{\n    "Title": "",\n    "arXiv Link": "",\n    "arXiv ID": "",\n    "Abstract": "",\n}\n```\n参数1：count\n参数2：search_text',
            "parameters": {
                "type": "object",
                "required": ["search_text"],
                "properties": {"count": {"type": "integer"}, "search_text": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dall_e_image_generation",
            "description": "输入一段文字，Dall-E 根据文字内容生成一张图片。",
            "parameters": {
                "type": "object",
                "required": ["prompt"],
                "properties": {"prompt": {"type": "string", "description": "简洁的画面描述词组、语句"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_speech_conversion",
            "description": "输入文字，生成合成的音频。",
            "parameters": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
        },
    },
]


if __name__ == "__main__":
    presets = {
        "deepseek-reasoner": (BackendType.DeepSeek, "deepseek-reasoner"),
        "gpt-4o": (BackendType.OpenAI, "gpt-4o"),
        "claude-3.5-sonnet": (BackendType.Anthropic, "claude-3-5-sonnet-20240620"),
        "gemini-1.5-flash": (BackendType.Gemini, "gemini-1.5-flash"),
        "yi-large-fc": (BackendType.Yi, "yi-large-fc"),
        "moonshot-v1-8k": (BackendType.Moonshot, "moonshot-v1-8k"),
    }
    backend, model = resolve_backend_model(BackendType.DeepSeek, "deepseek-reasoner", presets=presets)

    start_time = time.perf_counter()
    model_settings = settings.get_backend(backend=backend).models[model]
    client = create_chat_client(backend=backend, model=model, stream=False)
    formatted_messages = format_messages(
        messages,
        backend=backend,
        native_multimodal=model_settings.native_multimodal,
        function_call_available=model_settings.function_call_available,
    )
    response = client.create_completion(messages=formatted_messages, tools=tools)
    print(response)
    end_time = time.perf_counter()
    print(f"Stream time elapsed: {end_time - start_time} seconds")
