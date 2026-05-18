"""
LangGraph 多 Agent 编排 —— 整个项目最核心的调度文件。

图结构：
  START → load_context → ┬→ macro_analyst   ┐
                         ├→ technical       ├→ chief_editor → END
                         ├→ fund_flow       │
                         └→ risk_analyst    ┘
                          (4 个并行执行)

LangGraph 自动处理并行：当一个节点有多个下游，且下游之间无依赖，
它们会并发执行。当多个节点汇聚到同一个下游，下游等待全部完成后运行。
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from config.settings import Settings
from llm.client import LLMClient
from agents.macro_analyst import MacroAnalyst
from agents.technical_analyst import TechnicalAnalyst
from agents.fund_flow_analyst import FundFlowAnalyst
from agents.risk_analyst import RiskAnalyst
from agents.chief_editor import ChiefEditor
from rag.retriever import Retriever


# ================================================================
# 全局状态 —— Agent 之间靠它传递数据
# ================================================================
class PipelineState(TypedDict, total=False):
    # 输入（外部传入）
    market_data: dict
    date_str: str

    # 中间结果
    rag_context: str
    errors: list[str]

    # 4 个 Agent 的产出（Pydantic model 序列化为 dict）
    macro_result: Optional[dict]
    technical_result: Optional[dict]
    fund_flow_result: Optional[dict]
    risk_result: Optional[dict]

    # 最终产出
    final_report: Optional[dict]


# ================================================================
# 构建图
# ================================================================
def build_graph(settings: Settings, llm: LLMClient, retriever: Retriever) -> StateGraph:
    """构建并返回编译后的 LangGraph 工作流"""

    # ------- 初始化所有 Agent -------
    agents = {
        "macro": MacroAnalyst(llm, settings),
        "technical": TechnicalAnalyst(llm, settings),
        "fund_flow": FundFlowAnalyst(llm, settings),
        "risk": RiskAnalyst(llm, settings),
    }
    editor = ChiefEditor(llm, settings)

    workflow = StateGraph(PipelineState)

    # ============================================================
    # 节点 1: 加载上下文 + RAG 检索
    # ============================================================
    def load_context(state: PipelineState) -> dict:
        market_data = state["market_data"]
        try:
            rag = retriever.retrieve(market_data)
        except Exception as e:
            rag = f"[RAG 检索失败: {e}]"
        return {"rag_context": rag}

    # ============================================================
    # 工厂函数：为每个 Agent 创建一个 LangGraph 节点
    # ============================================================
    def make_agent_node(agent_name: str):
        """生产 Agent 节点函数。闭包捕获 agent_name。"""
        agent = agents[agent_name]

        def node(state: PipelineState) -> dict:
            try:
                result = agent.run(state["market_data"], state["rag_context"])
                return {f"{agent_name}_result": result.model_dump()}
            except Exception as e:
                return {
                    f"{agent_name}_result": None,
                    "errors": state.get("errors", []) + [f"{agent_name}: {e}"],
                }
        return node

    # ============================================================
    # 节点 6: 主编汇总
    # ============================================================
    def chief_editor_node(state: PipelineState) -> dict:
        # 检查是否有 Agent 失败了
        errors = state.get("errors", [])
        if errors:
            print(f"[ChiefEditor] 警告：以下 Agent 分析失败: {errors}")

        # 尝试反序列化各 Agent 的 dict → Pydantic 对象
        from agents.schemas import (
            MacroAnalysis, TechnicalAnalysis, FundFlowAnalysis, RiskAnalysis
        )

        def unpack(schema_cls, key: str):
            data = state.get(key)
            if data is None:
                raise ValueError(f"缺少 {key}，无法汇总")
            return schema_cls.model_validate(data)

        try:
            macro = unpack(MacroAnalysis, "macro_result")
            technical = unpack(TechnicalAnalysis, "technical_result")
            fund_flow = unpack(FundFlowAnalysis, "fund_flow_result")
            risk = unpack(RiskAnalysis, "risk_result")
        except Exception as e:
            return {
                "final_report": {"error": f"Agent 结果解析失败: {e}"},
                "errors": state.get("errors", []) + [f"chief_editor: {e}"],
            }

        final = editor.synthesize(
            date_str=state["date_str"],
            market_data=state["market_data"],
            rag_context=state["rag_context"],
            macro=macro,
            technical=technical,
            fund_flow=fund_flow,
            risk=risk,
        )
        return {"final_report": final.model_dump()}

    # ============================================================
    # 连线
    # ============================================================
    workflow.add_node("load_context", load_context)
    workflow.add_node("macro", make_agent_node("macro"))
    workflow.add_node("technical", make_agent_node("technical"))
    workflow.add_node("fund_flow", make_agent_node("fund_flow"))
    workflow.add_node("risk", make_agent_node("risk"))
    workflow.add_node("chief_editor", chief_editor_node)

    workflow.set_entry_point("load_context")

    # 扇出：load_context → 4 个 Agent
    for name in ["macro", "technical", "fund_flow", "risk"]:
        workflow.add_edge("load_context", name)

    # 扇入：4 个 Agent → chief_editor
    for name in ["macro", "technical", "fund_flow", "risk"]:
        workflow.add_edge(name, "chief_editor")

    workflow.add_edge("chief_editor", END)

    return workflow.compile()
