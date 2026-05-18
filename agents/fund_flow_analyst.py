"""
资金面分析师 —— 成交额TOP + 板块轮动 + 资金方向
"""
from config.settings import Settings
from config.agent_prompts.fund_flow import FUND_FLOW_SYSTEM
from agents.base import BaseAgent
from agents.schemas import FundFlowAnalysis
from llm.client import LLMClient


class FundFlowAnalyst(BaseAgent):
    def __init__(self, llm: LLMClient, settings: Settings):
        super().__init__(
            name="FundFlowAnalyst",
            system_prompt=FUND_FLOW_SYSTEM,
            output_schema=FundFlowAnalysis,
            model=settings.analysis_model,
            llm=llm,
        )

    def extract_focus_data(self, market_data: dict) -> str:
        top_amount = market_data.get("top_amount", [])
        sector_stats = market_data.get("sector_stats", [])
        conc = market_data.get("concentration", {})

        return f"""
成交额 TOP 10 个股：
{self._fmt_stocks(top_amount)}

板块资金流向：
{self._fmt_sectors(sector_stats)}

成交集中度：
  Top10 成交额占比: {conc.get('top10_ratio', 'N/A')}%
  Top50 成交额占比: {conc.get('top50_ratio', 'N/A')}%
"""

    def _fmt_stocks(self, stocks: list[dict]) -> str:
        lines = []
        for s in stocks[:10]:
            lines.append(
                f"  {s.get('code','?')} {s.get('name','?')} "
                f"成交额{s.get('amount',0):.0f}元 涨跌幅{s.get('change_pct',0):.2f}%"
            )
        return "\n".join(lines)

    def _fmt_sectors(self, sectors: list[dict]) -> str:
        lines = []
        for s in sectors[:8]:
            lines.append(
                f"  {s.get('industry_name','?')} "
                f"平均涨跌幅{s.get('avg_change',0):.2f}% "
                f"总成交额{s.get('total_amount',0):.0f}元"
            )
        return "\n".join(lines)
