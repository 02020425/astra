"""
输出写入 —— 把 FinalReport Pydantic 对象转成文件。

双输出：
  - JSON: 结构化、机器可读、可做历史趋势分析
  - Markdown: 人类阅读、可直接分享
"""

import json
from pathlib import Path
from datetime import datetime

from agents.schemas import FinalReport


def write_outputs(report: FinalReport, output_dir: Path, date_str: str):
    """写入 JSON + Markdown 双格式"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    json_path = output_dir / f"daily_report_{date_str}.json"
    json_path.write_text(
        report.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Writer] JSON → {json_path}")

    # --- Markdown ---
    md_path = output_dir / f"daily_report_{date_str}.md"
    md_content = render_markdown(report)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"[Writer] Markdown → {md_path}")


def render_markdown(report: FinalReport) -> str:
    """把 FinalReport 渲染为可读的 Markdown 格式"""

    lines = [
        f"# A股市场日报 ({report.report_date})",
        "",
        f"> **综合判断**: {report.overall_sentiment} | **风险等级**: {report.risk_level}",
        "",
        f"## 摘要",
        "",
        report.summary,
        "",
    ]

    for section in report.sections:
        lines.append(f"## {section.heading}")
        lines.append("")
        lines.append(section.content)
        lines.append("")
        # 来源标注
        lines.append(f"*({section.source_agent} 提供)*")
        lines.append("")

    lines.append("## 关键要点")
    lines.append("")
    for h in report.key_highlights:
        lines.append(f"- {h}")
    lines.append("")

    if report.contradictions_resolved:
        lines.append("## 分析师分歧说明")
        lines.append("")
        for c in report.contradictions_resolved:
            lines.append(f"- {c}")
        lines.append("")

    lines.append("---")
    lines.append(f"*报告由 ASTRA 多智能体系统自动生成*")
    return "\n".join(lines)
