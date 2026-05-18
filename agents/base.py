"""
Agent 抽象基类 —— 封装"调 LLM → 解析 JSON → 校验 → 重试"的公共流程。
每个具体 Agent 只写 3 样东西：名字、系统提示词、关心什么数据。
"""

import json
import re
from abc import ABC, abstractmethod
from pydantic import BaseModel
from llm.client import LLMClient


class BaseAgent(ABC):
    """
    所有 Agent 的父类。子类只需要：
    1. 设置 self.name / self.system_prompt / self.output_schema
    2. 实现 extract_focus_data() —— 从全量数据中摘出自己关心的部分
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        output_schema: type[BaseModel],
        model: str,
        llm: LLMClient,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.output_schema = output_schema
        self.model = model
        self.llm = llm

    # ================================================================
    # 子类必须实现：从全量数据中提取自己关注的字段
    # ================================================================
    @abstractmethod
    def extract_focus_data(self, market_data: dict) -> str:
        """
        每个 Agent 关注不同的数据子集：
        - 宏观分析师：PE + 涨跌比 + 历史对比
        - 技术分析师：趋势 + 支撑阻力 + 指数
        - 资金面：成交额 TOP + 板块 + 集中度
        - 风险：集中度 + 波动率 + 外部事件

        返回值是格式化的文本片段，会被嵌入 user prompt。
        """
        ...

    # ================================================================
    # 公共流程：拼 prompt → 调 LLM → 解析 → 校验 → 重试
    # ================================================================
    def run(self, market_data: dict, rag_context: str) -> BaseModel:
        """
        完整执行流程：
        1. 提取关注数据
        2. 拼 system + user prompt（含 JSON Schema）
        3. 调 LLM（JSON Mode）
        4. 解析 + Pydantic 校验
        """
        focus_data = self.extract_focus_data(market_data)

        # 把 Pydantic Schema 转成 LLM 能理解的 JSON 结构示例
        schema_json = self.output_schema.model_json_schema()
        schema_str = json.dumps(schema_json, ensure_ascii=False, indent=2)

        user_prompt = f"""## 今日市场数据
{focus_data}

## 相关新闻（RAG 检索）
{rag_context}

## 输出要求
必须严格输出以下 JSON Schema 定义的 JSON 对象，不要输出任何其他内容：
{schema_str}"""

        raw = self.llm.chat(
            model=self.model,
            system=self.system_prompt,
            user=user_prompt,
            max_tokens=2048,
            json_mode=True,  # 强制 JSON 输出
        )

        return self._parse_and_validate(raw)

    # ================================================================
    # JSON 解析 + 校验
    # ================================================================
    def _parse_and_validate(self, raw_text: str) -> BaseModel:
        """
        从 LLM 回复中提取 JSON，校验并返回 Pydantic 对象。

        LLM 有时候会在 JSON 外面包 markdown 代码块：
        ```json
        { ... }
        ```
        需要把外层的 ``` 去掉再解析。
        """
        # 1. 尝试直接解析
        try:
            return self.output_schema.model_validate_json(raw_text)
        except Exception:
            pass

        # 2. 尝试从 markdown 代码块中提取
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return self.output_schema.model_validate_json(json_str)

        # 3. 都失败了 —— 抛出明确错误
        raise ValueError(
            f"[{self.name}] 无法从回复中解析 JSON。\n"
            f"原始回复前 500 字:\n{raw_text[:500]}"
        )
