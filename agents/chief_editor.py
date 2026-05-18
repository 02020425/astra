"""
主编 —— 综合 4 份 Analyst 报告 + 消解矛盾 → 最终日报
"""
import json
from config.settings import Settings
from config.agent_prompts.chief_editor import CHIEF_EDITOR_SYSTEM
from agents.schemas import (
    MacroAnalysis, TechnicalAnalysis, FundFlowAnalysis, RiskAnalysis, FinalReport
)
from llm.client import LLMClient


class ChiefEditor:
    """
    主编跟其他 4 个 Agent 不同——它不继承 BaseAgent。
    因为它的输入不是原始 market_data，而是 4 份已经生成好的分析报告。
    它的任务是综合、消解矛盾、生成最终日报。
    """

    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.model = settings.analysis_model
        self.system_prompt = CHIEF_EDITOR_SYSTEM

    def synthesize(
        self,
        date_str: str,
        market_data: dict,
        rag_context: str,
        macro: MacroAnalysis,
        technical: TechnicalAnalysis,
        fund_flow: FundFlowAnalysis,
        risk: RiskAnalysis,
        question: str | None = None,
    ) -> FinalReport:
        """综合 4 份报告，回答用户问题。无 question 则生成日报。"""

        task_line = f"## 用户问题\n{question}\n\n请围绕以上问题，综合以下信息给出回答（JSON 格式）。" if question else "请综合以上所有信息，输出最终日报 JSON。"

        # 把 Pydantic 对象序列化为 JSON 字符串，嵌入 prompt
        user_prompt = f"""## 报告日期
{date_str}

## 原始市场数据摘要
{self._fmt_market_brief(market_data)}

## RAG 检索到的新闻
{rag_context}

## 宏观分析师报告 (置信度: {macro.confidence})
```json
{macro.model_dump_json(indent=2, ensure_ascii=False)}
```

## 技术分析师报告 (置信度: {technical.confidence})
```json
{technical.model_dump_json(indent=2, ensure_ascii=False)}
```

## 资金面分析师报告 (置信度: {fund_flow.confidence})
```json
{fund_flow.model_dump_json(indent=2, ensure_ascii=False)}
```

## 风险分析师报告 (置信度: {risk.confidence})
```json
{risk.model_dump_json(indent=2, ensure_ascii=False)}
```

{task_line}"""

        raw = self.llm.chat(
            model=self.model,
            system=self.system_prompt,
            user=user_prompt,
            max_tokens=3072,
            json_mode=True,
        )

        return FinalReport.model_validate_json(self._extract_json(raw))

    def _fmt_market_brief(self, data: dict) -> str:
        spot = data.get("spot_summary", {})
        return (
            f"上涨 {spot.get('up_count','?')} 家 / "
            f"下跌 {spot.get('down_count','?')} 家，"
            f"平均涨跌幅 {spot.get('avg_change_pct','?')}%"
        )

    def _extract_json(self, raw: str) -> str:
        """从 LLM 回复中提取纯 JSON"""
        import re
        raw = raw.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw
