"""
LLM 客户端 —— 整个项目只有一个入口调大模型，就是这里。

为什么用 OpenAI 的包调阿里云？
→ DashScope 提供了 "compatible-mode" 端点，API 格式跟 OpenAI 一模一样。
   用 openai 官方 SDK 就能直接调，不需要额外装 dashscope 的包。
"""

import json
from openai import OpenAI
from config.settings import Settings


class LLMClient:
    """封装 DashScope OpenAI 兼容接口，给所有 Agent 和 RAG 模块用。"""

    def __init__(self, settings: Settings):
        # 核心就这一行：建立一个到阿里云的 HTTP 连接对象
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )

    # ================================================================
    # 方法 1：普通对话 → 返回文本
    # ================================================================
    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """
        发一次对话请求，返回 AI 的文本回复。

        参数：
            model:       用哪个模型（qwen-plus / qwen-turbo）
            system:      system prompt —— 定义 AI 的角色和行为
            user:        user prompt —— 这次具体要问什么
            temperature: 0=确定  1=随机  0.7=分析类任务常用
            max_tokens:  回复最多多少个 token
            json_mode:   是否启用 JSON Mode，强制模型只输出 JSON
        """
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # JSON Mode —— 让 API 层面就限制输出格式
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    # ================================================================
    # 方法 2：JSON 对话 → 返回 dict
    # ================================================================
    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict:
        """
        发一次对话请求，要求 AI 只输出 JSON，返回解析后的 dict。

        为什么需要这个？
        → 后面 4 个 Agent + 主编 都要输出结构化 JSON，
          用 response_format={"type": "json_object"} 强制 AI 只返回 JSON，
          不再需要手动从回复里抠 { 和 }。
        """
        text = self.chat(
            model=model,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return json.loads(text)
