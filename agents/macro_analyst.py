"""
宏观分析师 —— PE估值 + 市场广度 + 经济事件
"""
from config.settings import Settings
from config.agent_prompts.macro import MACRO_SYSTEM
from agents.base import BaseAgent
from agents.schemas import MacroAnalysis
from llm.client import LLMClient


class MacroAnalyst(BaseAgent):
    def __init__(self, llm: LLMClient, settings: Settings):
        super().__init__(
            name="MacroAnalyst",
            system_prompt=MACRO_SYSTEM,
            output_schema=MacroAnalysis,
            model=settings.analysis_model,
            llm=llm,
        )

    def extract_focus_data(self, market_data: dict) -> str:
        pe = market_data["pe"]
        breadth = market_data["spot_summary"]
        return f"""
PE 估值数据：
  当前 PE: {pe['current']}
  历史最高: {pe['max']}
  历史最低: {pe['min']}
  历史均值: {pe['mean']}
  当前分位数: {pe['percentile']}%

市场广度数据：
  上涨: {breadth['up_count']} 只
  下跌: {breadth['down_count']} 只
  平盘: {breadth['flat_count']} 只
  平均涨跌幅: {breadth['avg_change_pct']}%
"""
