"""
ASTRA Gradio 前端 —— 多智能体交互问答。

用法：
    python app.py
    浏览器打开 http://localhost:7860

两种模式：
  1. 问答模式：用户提问 → RAG 检索 → 4 Agent 分析 → 主编回答
  2. 日报模式：点击按钮 → 完整 6 阶段 pipeline → 生成日报
"""

import json
from datetime import datetime

import gradio as gr

from config.settings import Settings
from llm.client import LLMClient
from rag.embedder import DashScopeEmbedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from agents.macro_analyst import MacroAnalyst
from agents.technical_analyst import TechnicalAnalyst
from agents.fund_flow_analyst import FundFlowAnalyst
from agents.risk_analyst import RiskAnalyst
from agents.chief_editor import ChiefEditor
from agents.schemas import (
    MacroAnalysis, TechnicalAnalysis, FundFlowAnalysis, RiskAnalysis
)


# ================================================================
# 初始化（全局单例，避免每次请求重新初始化）
# ================================================================
settings = Settings()
llm = LLMClient(settings)
embedder = DashScopeEmbedder(settings)
store = VectorStore(settings.data_chroma, settings.chroma_collection)
retriever = Retriever(settings, llm, embedder, store)

agents = {
    "macro": MacroAnalyst(llm, settings),
    "technical": TechnicalAnalyst(llm, settings),
    "fund_flow": FundFlowAnalyst(llm, settings),
    "risk": RiskAnalyst(llm, settings),
}
editor = ChiefEditor(llm, settings)

AGENT_DISPLAY = {
    "macro": "🏛️ 宏观分析师",
    "technical": "📈 技术分析师",
    "fund_flow": "💰 资金面分析师",
    "risk": "⚠️ 风险分析师",
}


# ================================================================
# 核心问答逻辑
# ================================================================
def answer_question(question: str, history: list) -> str:
    """
    处理用户提问：
    1. RAG 检索相关新闻
    2. 4 个 Agent 各自分析（串行，避免 Gradio 异步问题）
    3. 主编综合
    4. 返回格式化的 Markdown 答案
    """
    if not question or not question.strip():
        return "请输入你的问题。"

    # Step 1: RAG 检索
    rag_context = retriever.retrieve_by_question(question)

    # Step 2: 构建一个简化的 market_data（问答模式不需要完整行情数据）
    #         用 RAG 检索到的新闻作为上下文
    market_data = {
        "spot_summary": {"note": "交互问答模式，使用 RAG 检索上下文"},
        "pe": {"note": "交互问答模式"},
        "sector_stats": [],
        "concentration": {},
        "index_stats": {},
        "top_amount": [],
    }

    # Step 3: 4 个 Agent 分别分析
    agent_results = {}
    for name in ["macro", "technical", "fund_flow", "risk"]:
        try:
            result = agents[name].run(market_data, rag_context)
            agent_results[name] = result
        except Exception as e:
            agent_results[name] = None

    # Step 4: 主编综合
    try:
        final = editor.synthesize(
            date_str=datetime.today().strftime("%Y-%m-%d"),
            market_data=market_data,
            rag_context=rag_context,
            macro=agent_results.get("macro") or _empty_macro(),
            technical=agent_results.get("technical") or _empty_technical(),
            fund_flow=agent_results.get("fund_flow") or _empty_fund_flow(),
            risk=agent_results.get("risk") or _empty_risk(),
        )
    except Exception as e:
        final = None
        error_msg = f"主编汇总失败: {e}"

    # Step 5: 格式化输出
    return _format_answer(question, agent_results, final, rag_context)


def generate_report() -> str:
    """一键生成完整日报（调用 CLI 主流程）"""
    from main import run
    date_str = datetime.today().strftime("%Y%m%d")
    try:
        run(date_str)
        return f"✅ 日报已生成！\n\n📄 查看: `data/reports/daily_report_{date_str}.md`"
    except Exception as e:
        return f"❌ 生成失败: {e}"


# ================================================================
# 格式化输出
# ================================================================
def _format_answer(
    question: str,
    agent_results: dict,
    final,
    rag_context: str,
) -> str:
    """把 Agent 结果和主编回答格式化为 Markdown"""

    lines = [f"## 🤖 多智能体分析\n", f"**你的问题**: {question}\n"]

    # --- 四个 Agent 的观点 ---
    lines.append("### 各分析师观点\n")
    lines.append("| 分析师 | 核心观点 | 置信度 |")
    lines.append("|--------|---------|--------|")

    for name in ["macro", "technical", "fund_flow", "risk"]:
        result = agent_results.get(name)
        display = AGENT_DISPLAY[name]
        if result is None:
            lines.append(f"| {display} | ❌ 分析失败 | - |")
        else:
            brief = _extract_brief(name, result)
            conf = result.confidence
            lines.append(f"| {display} | {brief} | {conf:.0%} |")

    lines.append("")

    # --- 主编综合 ---
    lines.append("### 📝 综合研判\n")
    if final is None:
        lines.append("（主编汇总失败）")
    else:
        lines.append(f"**市场情绪**: {final.overall_sentiment}")
        lines.append(f"**风险等级**: {final.risk_level}\n")
        lines.append(final.summary)
        lines.append("")

        for section in final.sections:
            lines.append(f"**{section.heading}**")
            lines.append(section.content)
            lines.append("")

        if final.key_highlights:
            lines.append("**关键要点**:")
            for h in final.key_highlights:
                lines.append(f"- {h}")

    # --- 参考新闻 ---
    if rag_context and rag_context.strip():
        lines.append("\n---\n")
        lines.append("### 📎 参考新闻 (RAG 检索)")
        # 截取前 3 条
        snippets = rag_context.split("---")[:3]
        lines.append("\n".join(snippets))

    return "\n".join(lines)


def _extract_brief(name: str, result) -> str:
    """从 Agent 结果中提取一句话摘要"""
    if name == "macro":
        return f"{result.sentiment}，PE分位{result.pe_valuation.percentile}%"
    elif name == "technical":
        return f"{result.trend}（{result.trend_strength}），{result.index_summary[:30]}"
    elif name == "fund_flow":
        return result.capital_direction
    elif name == "risk":
        return f"风险{result.overall_risk_level}，脆弱度{result.fragility_score:.0%}"
    return "—"


# ================================================================
# 空结果填充（Agent 失败时用）
# ================================================================
def _empty_macro():
    from agents.schemas import PEAnalysis, MarketBreadth
    return MacroAnalysis(
        pe_valuation=PEAnalysis(current_pe=30, historical_max=60, historical_min=10, historical_mean=35, percentile=50, assessment="fair"),
        market_breadth=MarketBreadth(up_count=0, down_count=0, flat_count=0, avg_change_pct=0, breadth_signal="neutral"),
        economic_events=[], sentiment="未知（分析失败）", key_drivers=[], confidence=0.1,
    )

def _empty_technical():
    return TechnicalAnalysis(trend="未知", trend_strength="弱", index_summary="分析失败", momentum_signals=[], confidence=0.1)

def _empty_fund_flow():
    return FundFlowAnalysis(top_turnover_stocks=[], sector_rotation=[], capital_direction="未知", concentration_note="分析失败", confidence=0.1)

def _empty_risk():
    return RiskAnalysis(concentration_risk="未知", volatility_assessment="未知", external_risks=[], stock_warnings=[], fragility_score=0.5, overall_risk_level="medium", confidence=0.1)


# ================================================================
# Gradio UI
# ================================================================
def build_ui():
    with gr.Blocks(
        title="ASTRA 多智能体 A 股分析",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🏦 ASTRA 多智能体 A 股分析系统

        向 AI 分析师团队提问，或一键生成今日完整日报。
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=550,
                    bubble_full_width=False,
                )
                msg = gr.Textbox(
                    label="",
                    placeholder="输入你的问题，例如：半导体板块最近走势如何？当前市场风险大吗？...",
                    scale=4,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空对话", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 快捷操作")
                report_btn = gr.Button("📊 生成今日日报", variant="secondary", size="lg")
                date_display = gr.Textbox(
                    label="今日日期",
                    value=datetime.today().strftime("%Y-%m-%d"),
                    interactive=False,
                )
                gr.Markdown("""
                ---
                ### 分析师团队
                - 🏛️ 宏观分析师
                - 📈 技术分析师
                - 💰 资金面分析师
                - ⚠️ 风险分析师
                - 📝 主编汇总
                ---
                ### 示例问题
                - 今天市场整体怎么样？
                - 科技股还能追吗？
                - 有什么风险需要注意？
                - 哪些板块资金在流入？
                """)

        # ---- 事件绑定 ----
        send_btn.click(
            fn=answer_question,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=lambda: "",
            outputs=[msg],
        )

        msg.submit(
            fn=answer_question,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=lambda: "",
            outputs=[msg],
        )

        clear_btn.click(
            fn=lambda: ([], "", "对话已清空"),
            outputs=[chatbot, msg, chatbot],
        )

        report_btn.click(
            fn=generate_report,
            outputs=[chatbot],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 如需公网链接改成 True
    )
