"""
ASTRA —— A股多智能体分析系统 CLI 入口。

用法：
    python main.py                          # 分析今日
    python main.py --date 20260518          # 分析指定日期
    python main.py --date 20260518 --crawl-only   # 只爬数据
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.helpers import patch_akshare
patch_akshare()

from config.settings import Settings
from llm.client import LLMClient
from crawler.market_crawler import MarketCrawler
from crawler.news_crawler import NewsCrawler
from spark_jobs.etl import run_etl
from rag.embedder import DashScopeEmbedder
from rag.vector_store import VectorStore
from rag.news_loader import load_daily_news
from rag.retriever import Retriever
from agents.graph import build_graph
from agents.schemas import FinalReport
from output.writer import write_outputs
from eval.eval_card import generate_eval_card
from utils.helpers import read_processed_csv


# ================================================================
# 主流程
# ================================================================
def run(date_str: str, crawl_only: bool = False):
    print("=" * 60)
    print(f"  ASTRA 多智能体 A 股分析系统")
    print(f"  日期: {date_str}")
    print("=" * 60)

    # ---- 初始化 ----
    settings = Settings()
    llm = LLMClient(settings)

    # ================================================================
    # 阶段 1: 数据采集
    # ================================================================
    print("\n" + "=" * 40)
    print("  [1/6] 数据采集")
    print("=" * 40)

    mc = MarketCrawler(settings)
    mc.today = date_str
    mc.run_all()

    # 获取成交量 TOP20 的股票代码，用于新闻爬取
    try:
        spot_df = pd.read_csv(
            next(Path(settings.data_raw / "a_spot").glob(f"{date_str}.csv"))
        )
        top_codes = spot_df.nlargest(20, "成交额")["代码"].tolist()
    except Exception:
        top_codes = []

    nc = NewsCrawler(settings)
    nc.today = date_str
    nc.run_all(top_stock_codes=top_codes)

    if crawl_only:
        print("\n--crawl-only 模式，停止。")
        return

    # ================================================================
    # 阶段 2: Spark ETL
    # ================================================================
    print("\n" + "=" * 40)
    print("  [2/6] Spark ETL 分析")
    print("=" * 40)
    run_etl(settings)

    # ================================================================
    # 阶段 3: 加载处理后数据 → market_data dict
    # ================================================================
    print("\n" + "=" * 40)
    print("  [3/6] 加载分析数据")
    print("=" * 40)

    market_data = _load_processed(settings.data_processed)
    market_data["date_str"] = date_str

    # 从 PE 数据计算分位数
    pe_df = market_data.get("pe_raw")
    if pe_df is not None and len(pe_df) > 0:
        current_pe = float(pe_df["avg_pe_val"].iloc[-1])
        percentile = (pe_df["avg_pe_val"] < current_pe).mean() * 100
        market_data["pe"] = {
            "current": current_pe,
            "max": float(pe_df["avg_pe_val"].max()),
            "min": float(pe_df["avg_pe_val"].min()),
            "mean": float(pe_df["avg_pe_val"].mean()),
            "percentile": round(percentile, 1),
        }

    print(f"  全市场 {market_data['spot_summary']['total_count']} 只股票")
    print(f"  板块数: {len(market_data.get('sector_stats', []))}")

    # ================================================================
    # 阶段 4: RAG 知识库构建 + 检索
    # ================================================================
    print("\n" + "=" * 40)
    print("  [4/6] RAG 知识库")
    print("=" * 40)

    embedder = DashScopeEmbedder(settings)
    store = VectorStore(settings.data_chroma, settings.chroma_collection)

    # 4a: 新闻嵌入入库
    n = load_daily_news(date_str, embedder, store, settings.data_raw / "news")
    print(f"  知识库总量: {store.count()} 条")

    # 4b: 检索
    retriever = Retriever(settings, llm, embedder, store)
    rag_context = retriever.retrieve(market_data)
    print(f"  检索到上下文，长度: {len(rag_context)} 字")

    # ================================================================
    # 阶段 5: 多智能体分析
    # ================================================================
    print("\n" + "=" * 40)
    print("  [5/6] 多智能体分析")
    print("=" * 40)

    graph = build_graph(settings, llm, retriever)

    result = graph.invoke({
        "market_data": market_data,
        "date_str": date_str,
    })

    # 提取结果
    errors = result.get("errors", [])
    if errors:
        print(f"  WARN 部分 Agent 失败: {errors}")

    final_dict = result.get("final_report")
    if final_dict is None:
        print("  ERROR 主编汇总失败，退出。")
        return

    final_report = FinalReport.model_validate(final_dict)
    agent_outputs = {
        k: v
        for k, v in result.items()
        if k.endswith("_result") and v is not None
    }

    # ================================================================
    # 阶段 6: 输出 + 评估
    # ================================================================
    print("\n" + "=" * 40)
    print("  [6/6] 输出 + 评估")
    print("=" * 40)

    write_outputs(final_report, settings.data_reports, date_str)

    generate_eval_card(
        date_str=date_str,
        rag_context=rag_context,
        market_data=market_data,
        agent_outputs=agent_outputs,
        final_report=final_report,
        settings=settings,
        llm=llm,
    )

    print("\n" + "=" * 60)
    print(f"  完成！报告: data/reports/daily_report_{date_str}.md")
    print("=" * 60)


# ================================================================
# 数据加载工具
# ================================================================
def _load_processed(processed_dir: Path) -> dict:
    """把 Spark 输出的所有 CSV 读成 Python dict"""
    data = {}

    # 市场摘要
    try:
        df = read_processed_csv(processed_dir, "market_summary")
        row = df.iloc[0].to_dict()
        data["spot_summary"] = {
            "up_count": int(row["up_count"]),
            "down_count": int(row["down_count"]),
            "flat_count": int(row["flat_count"]),
            "total_count": int(row["total_count"]),
            "avg_change_pct": float(row["avg_change_pct"]),
            "std_change": float(row.get("std_change_pct", 0)),
            "total_amount": float(row["total_amount"]),
        }
    except Exception as e:
        print(f"  加载 market_summary 失败: {e}")
        data["spot_summary"] = {}

    # PE
    try:
        df = read_processed_csv(processed_dir, "pe_by_year")
        data["pe_raw"] = df
    except Exception:
        data["pe_raw"] = None

    # 涨幅 TOP
    try:
        df = read_processed_csv(processed_dir, "top_gainers")
        data["top_gainers"] = df.to_dict(orient="records")
    except Exception:
        data["top_gainers"] = []

    # 跌幅 TOP
    try:
        df = read_processed_csv(processed_dir, "top_losers")
        data["top_losers"] = df.to_dict(orient="records")
    except Exception:
        data["top_losers"] = []

    # 成交额 TOP
    try:
        df = read_processed_csv(processed_dir, "top_amount")
        data["top_amount"] = df.to_dict(orient="records")
    except Exception:
        data["top_amount"] = []

    # 板块聚合
    try:
        df = read_processed_csv(processed_dir, "sector_stats")
        data["sector_stats"] = df.to_dict(orient="records")
    except Exception:
        data["sector_stats"] = []

    # 集中度
    try:
        df = read_processed_csv(processed_dir, "concentration")
        row = df.iloc[0].to_dict()
        data["concentration"] = {
            "top10_ratio": float(row["top10_pct"]),
            "top50_ratio": float(row["top50_pct"]),
        }
    except Exception:
        data["concentration"] = {}

    # 指数统计
    try:
        df = read_processed_csv(processed_dir, "index_stats")
        row = df.iloc[0].to_dict()
        data["index_stats"] = {
            "close_min": float(row["close_min"]),
            "close_max": float(row["close_max"]),
            "close_avg": float(row.get("close_avg", 0)),
            "volatility": float(row.get("volatility", 0)),
        }
    except Exception:
        data["index_stats"] = {}

    return data


# ================================================================
# 入口
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRA 多智能体 A 股分析")
    parser.add_argument(
        "--date",
        default=datetime.today().strftime("%Y%m%d"),
        help="分析日期，格式 YYYYMMDD，默认今天",
    )
    parser.add_argument(
        "--crawl-only",
        action="store_true",
        help="只爬数据，不做分析和生成报告",
    )
    args = parser.parse_args()
    run(args.date, args.crawl_only)
