"""
日度评估卡 —— 每天生成一份，衡量系统运行质量。

存到 data/eval_cards/eval_YYYYMMDD.json
"""

import json
from pathlib import Path

from config.settings import Settings
from llm.client import LLMClient
from agents.schemas import FinalReport
from eval.metrics import (
    evaluate_retrieval_precision,
    evaluate_consistency,
    evaluate_report_quality,
)


def generate_eval_card(
    date_str: str,
    rag_context: str,
    market_data: dict,
    agent_outputs: dict,
    final_report: FinalReport,
    settings: Settings,
    llm: LLMClient,
) -> dict:
    """
    生成一份日度评估卡。
    """

    light_model = settings.lightweight_model

    # 1. 检索精度
    retrieval_score = evaluate_retrieval_precision(
        rag_context, market_data, llm, light_model
    )

    # 2. Agent 一致性
    consistency = evaluate_consistency(agent_outputs, llm, light_model)

    # 3. 报告质量（规则检查）
    quality = evaluate_report_quality(final_report, rag_context)

    # 4. 综合评分（加权）
    overall = round(
        retrieval_score * 0.25
        + consistency * 0.25
        + (1.0 if quality["pass"] else 0.5) * 0.50,
        2,
    )

    card = {
        "date": date_str,
        "retrieval_precision": retrieval_score,
        "cross_agent_consistency": consistency,
        "report_quality": quality,
        "overall_score": overall,
    }

    # 保存
    eval_dir = settings.data_eval_cards
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / f"eval_{date_str}.json"
    path.write_text(
        json.dumps(card, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[Eval] 评估卡 → {path}")
    print(f"  检索精度: {retrieval_score:.0%}")
    print(f"  Agent一致性: {consistency:.0%}")
    print(f"  报告质量: {'PASS' if quality['pass'] else 'FAIL'}")
    print(f"  综合评分: {overall:.0%}")

    return card
