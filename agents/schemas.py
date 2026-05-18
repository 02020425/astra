"""
Pydantic v2 结构化输出模型 —— Agent 之间的"合同"。

每个 Agent 输出不是自由文本，而是严格定义的 JSON。
好处：
  1. 结构可校验 —— 缺字段/Pydantic 直接报错
  2. Agent 间可靠传递 —— 主编读到的不是一段话，是精确的 dict
  3. 可存储、可对比 —— 历史报告可以按字段做统计分析

Field(description=...) 不只是给人看 —— 它会变成 JSON Schema，
传给 LLM 的 response_format 参数，告诉模型"按这个结构输出"。
"""

from pydantic import BaseModel, Field
from typing import Optional


# ================================================================
# 宏观分析师输出
# ================================================================
class PEAnalysis(BaseModel):
    """PE 估值子结构"""
    current_pe: float = Field(description="当前市场平均 PE")
    historical_max: float
    historical_min: float
    historical_mean: float
    percentile: float = Field(ge=0, le=100, description="当前 PE 在历史中的分位数")
    assessment: str = Field(description="估值判断: undervalued / fair / overvalued")


class MarketBreadth(BaseModel):
    """涨跌比子结构"""
    up_count: int
    down_count: int
    flat_count: int
    avg_change_pct: float
    breadth_signal: str = Field(description="广度信号: bullish / neutral / bearish")


class MacroAnalysis(BaseModel):
    """宏观分析师完整输出"""
    pe_valuation: PEAnalysis
    market_breadth: MarketBreadth
    economic_events: list[str] = Field(description="从 RAG 新闻中提取的宏观事件")
    sentiment: str = Field(description="综合情绪判断: 偏多 / 中性 / 偏空")
    key_drivers: list[str] = Field(description="影响今日市场的主要驱动因素，2-4条")
    confidence: float = Field(ge=0, le=1, description="对本报告分析的置信度，0最低1最高")


# ================================================================
# 技术分析师输出
# ================================================================
class TechnicalAnalysis(BaseModel):
    """技术分析师完整输出"""
    trend: str = Field(description="趋势方向: 上涨 / 下跌 / 震荡")
    trend_strength: str = Field(description="趋势强度: 强 / 中等 / 弱")
    support_level: Optional[float] = Field(default=None, description="关键支撑位")
    resistance_level: Optional[float] = Field(default=None, description="关键阻力位")
    index_summary: str = Field(description="沪深300指数概况，1-2句话")
    momentum_signals: list[str] = Field(description="技术信号列表，如 MACD死叉/KDJ超卖")
    confidence: float = Field(ge=0, le=1)


# ================================================================
# 资金面分析师输出
# ================================================================
class SectorFlow(BaseModel):
    """板块资金流子结构"""
    sector_name: str
    avg_change_pct: float
    total_amount: float  # 成交额（元）
    direction: str = Field(description="资金方向: inflow / outflow / neutral")


class StockFlow(BaseModel):
    """个股资金流子结构"""
    code: str
    name: str
    amount: float = Field(description="成交额（元）")
    change_pct: float


class FundFlowAnalysis(BaseModel):
    """资金面分析师完整输出"""
    top_turnover_stocks: list[StockFlow] = Field(description="成交额最大的几只股票，3-5只")
    sector_rotation: list[SectorFlow] = Field(description="资金流入/流出最明显的板块，3-5个")
    capital_direction: str = Field(description="资金整体方向: 科技主线 / 防御切换 / 分散")
    concentration_note: str = Field(description="对成交集中度的判断")
    confidence: float = Field(ge=0, le=1)


# ================================================================
# 风险分析师输出
# ================================================================
class RiskAnalysis(BaseModel):
    """风险分析师完整输出"""
    concentration_risk: str = Field(description="集中度风险评估")
    volatility_assessment: str = Field(description="波动率评估")
    external_risks: list[str] = Field(description="从RAG新闻中识别的外部风险，如政策/地缘")
    stock_warnings: list[str] = Field(description="高风险个股/ST预警")
    fragility_score: float = Field(ge=0, le=1, description="市场脆弱度评分，越高越脆弱")
    overall_risk_level: str = Field(description="综合风险等级: low / medium / high")
    confidence: float = Field(ge=0, le=1)


# ================================================================
# 主编汇总输出
# ================================================================
class ReportSection(BaseModel):
    """日报的一个章节 —— 标注来源 Agent"""
    heading: str = Field(description="章节标题")
    content: str = Field(description="章节正文")
    source_agent: str = Field(description="主要贡献的分析师，如 macro / technical / fund_flow / risk")


class FinalReport(BaseModel):
    """主编最终输出 —— 完整日报"""
    report_date: str = Field(description="报告日期 YYYY-MM-DD")
    summary: str = Field(description="一句话市场总览")
    sections: list[ReportSection] = Field(description="报告章节列表，4-6个")
    overall_sentiment: str = Field(description="综合情绪: 偏多 / 中性 / 偏空")
    risk_level: str = Field(description="综合风险等级: low / medium / high")
    key_highlights: list[str] = Field(description="关键要点，3-5条")
    contradictions_resolved: list[str] = Field(
        description="当分析师观点矛盾时，说明矛盾点及主编的判断"
    )
