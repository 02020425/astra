"""
评估指标 —— 衡量 RAG 检索质量和 Agent 输出质量。

核心方法：
  - 检索精度: LLM-as-judge 判断检索到的新闻是否与今日市场相关
  - 一致性: LLM-as-judge 判断 4 个 Agent 结论是否逻辑自洽
  - 报告质量: 规则检查（字数、章节完整性、可操作性）
"""

from agents.schemas import FinalReport
from llm.client import LLMClient


def evaluate_retrieval_precision(
    rag_context: str,
    market_data: dict,
    llm: LLMClient,
    model: str,
) -> float:
    """
    用轻量 LLM 评估检索结果的相关性。

    做法：把检索到的新闻内容和今日市场数据给 LLM，
    让它逐条打分（0-1），取平均值作为检索精度。
    """
    if not rag_context or rag_context.startswith("（暂无"):
        return 0.0

    prompt = f"""你是评估专家。以下是今日 A 股市场数据和 RAG 检索到的新闻。

## 市场数据
上涨 {market_data.get('spot_summary', {}).get('up_count', '?')} 家
下跌 {market_data.get('spot_summary', {}).get('down_count', '?')} 家
平均涨跌幅 {market_data.get('spot_summary', {}).get('avg_change_pct', '?')}%

## 检索到的新闻
{rag_context[:2000]}

请判断这些新闻中，与今日市场情况**明显相关**的有几条？
输出格式：{{"relevant": 数字, "total": 数字, "score": 0.0-1.0}}
只输出 JSON。"""

    try:
        raw = llm.chat(
            model=model,
            system="评估检索质量。只输出 JSON。",
            user=prompt,
            max_tokens=200,
            json_mode=True,
            temperature=0.1,
        )
        import json
        result = json.loads(raw)
        return float(result.get("score", 0.5))
    except Exception:
        return 0.5  # 评估失败，给中等分


def evaluate_consistency(
    agent_outputs: dict,
    llm: LLMClient,
    model: str,
) -> float:
    """
    评估 4 个 Agent 结论的一致性。

    完全一致 = 1.0，严重矛盾 = 0.0。
    """
    macro = agent_outputs.get("macro_result", {})
    tech = agent_outputs.get("technical_result", {})
    flow = agent_outputs.get("fund_flow_result", {})
    risk = agent_outputs.get("risk_result", {})

    prompt = f"""评估以下 4 份分析报告的逻辑一致性：

宏观: sentiment={macro.get('sentiment','?')}, confidence={macro.get('confidence','?')}
技术: trend={tech.get('trend','?')}, confidence={tech.get('confidence','?')}
资金面: capital_direction={flow.get('capital_direction','?')}, confidence={flow.get('confidence','?')}
风险: overall_risk_level={risk.get('overall_risk_level','?')}, confidence={risk.get('confidence','?')}

一致程度（0-1，1=完全一致，0=严重矛盾）？输出 JSON: {{"consistency": 0.0-1.0, "note": "简短说明"}}"""

    try:
        raw = llm.chat(
            model=model,
            system="评估一致性。只输出 JSON。",
            user=prompt,
            max_tokens=150,
            json_mode=True,
            temperature=0.1,
        )
        import json
        result = json.loads(raw)
        return float(result.get("consistency", 0.5))
    except Exception:
        return 0.5


def evaluate_report_quality(report: FinalReport, rag_context: str) -> dict:
    """规则检查报告质量，不调 LLM"""

    md_text = ""
    for s in report.sections:
        md_text += s.content + "\n"

    word_count = len(md_text)
    section_count = len(report.sections)
    has_highlights = len(report.key_highlights) >= 3
    has_contradictions = len(report.contradictions_resolved) > 0
    cites_news = any(
        keyword in md_text
        for keyword in ["新闻", "报道", "消息", "RAG"]
    )

    return {
        "word_count": word_count,
        "section_count": section_count,
        "has_highlights": has_highlights,
        "has_contradictions": has_contradictions,
        "cites_news": cites_news,
        "pass": (
            400 <= word_count <= 1000
            and section_count >= 4
            and has_highlights
        ),
    }
