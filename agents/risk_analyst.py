"""
风险分析师 —— 集中度风险 + 波动率 + 外部事件 + 脆弱度评分
"""
from config.settings import Settings
from config.agent_prompts.risk import RISK_SYSTEM
from agents.base import BaseAgent
from agents.schemas import RiskAnalysis
from llm.client import LLMClient


class RiskAnalyst(BaseAgent):
    def __init__(self, llm: LLMClient, settings: Settings):
        super().__init__(
            name="RiskAnalyst",
            system_prompt=RISK_SYSTEM,
            output_schema=RiskAnalysis,
            model=settings.analysis_model,
            llm=llm,
        )

    def extract_focus_data(self, market_data: dict) -> str:
        conc = market_data.get("concentration", {})
        spot = market_data.get("spot_summary", {})
        top_losers = market_data.get("top_losers", [])

        # 跌幅超过 5% 的个股
        big_losers = [s for s in top_losers if abs(s.get("change_pct", 0)) > 5]

        return f"""
风险相关数据：
  全市场涨跌幅标准差: {spot.get('std_change', 'N/A')}
  成交集中度 Top10: {conc.get('top10_ratio', 'N/A')}%
  成交集中度 Top50: {conc.get('top50_ratio', 'N/A')}%

跌幅超 5% 个股数: {len(big_losers)} 只
跌幅前 5:
{self._fmt_stocks(big_losers[:5])}
"""

    def _fmt_stocks(self, stocks: list[dict]) -> str:
        if not stocks:
            return "  (无)"
        lines = []
        for s in stocks:
            lines.append(
                f"  {s.get('code','?')} {s.get('name','?')} "
                f"跌幅 {s.get('change_pct',0):.2f}%"
            )
        return "\n".join(lines)
