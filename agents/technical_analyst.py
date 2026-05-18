"""
技术分析师 —— 趋势判断 + 支撑阻力 + 技术信号
"""
from config.settings import Settings
from config.agent_prompts.technical import TECHNICAL_SYSTEM
from agents.base import BaseAgent
from agents.schemas import TechnicalAnalysis
from llm.client import LLMClient


class TechnicalAnalyst(BaseAgent):
    def __init__(self, llm: LLMClient, settings: Settings):
        super().__init__(
            name="TechnicalAnalyst",
            system_prompt=TECHNICAL_SYSTEM,
            output_schema=TechnicalAnalysis,
            model=settings.analysis_model,
            llm=llm,
        )

    def extract_focus_data(self, market_data: dict) -> str:
        idx = market_data["index_stats"]
        return f"""
沪深300 指数数据：
  收盘价: {idx.get('close', 'N/A')}
  区间最低: {idx.get('close_min', 'N/A')}
  区间最高: {idx.get('close_max', 'N/A')}
  区间平均涨跌幅: {idx.get('avg_change_pct', 'N/A')}%

市场整体涨跌幅分布标准差: {market_data['spot_summary'].get('std_change', 'N/A')}
"""
