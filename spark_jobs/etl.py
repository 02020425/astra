"""
PySpark ETL —— 结构化数据清洗 + 多维聚合分析。
包含 pandas 回退方案，当 Spark 不可用时（如 Windows 无 Hadoop）自动降级。

输入：data/raw/（爬虫产出的 CSV）
输出：data/processed/（8 份分析结果 CSV）

本地运行：python spark_jobs/etl.py
Spark Submit：spark-submit spark_jobs/etl.py
"""

from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import Settings


def run_etl(settings: Settings):
    """先尝试 Spark，失败则用 pandas 回退"""
    if not _java_ok():
        print("  Java 版本不满足 PySpark 要求，使用 pandas 回退方案")
        _run_etl_pandas(settings)
        return
    try:
        _run_etl_spark(settings)
    except Exception as e:
        print(f"  Spark 不可用 ({e})，使用 pandas 回退方案")
        _run_etl_pandas(settings)


def _java_ok() -> bool:
    """检查 Java 版本是否 >= 17（PySpark 3.5 要求）"""
    import subprocess
    try:
        out = subprocess.check_output(
            ["java", "-version"], stderr=subprocess.STDOUT, text=True, timeout=10
        )
        for line in out.splitlines():
            if "version" in line:
                # "1.8.0_181" → 8, "17.0.1" → 17
                ver = line.split('"')[1] if '"' in line else line.split()[-1]
                major = int(ver.split(".")[0]) if ver.startswith("1.") else (
                    int(ver.split(".")[0]) if ver[0].isdigit() else int(ver.split(".")[0])
                )
                # 处理 "1.8.0" 格式
                if ver.startswith("1."):
                    major = int(ver.split(".")[1])
                else:
                    major = int(ver.split(".")[0])
                return major >= 17
    except Exception:
        return False
    return False


# ================================================================
# PySpark 实现
# ================================================================
def _run_etl_spark(settings: Settings):
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder \
        .appName("ASTRA_ETL") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    raw = settings.data_raw
    out = settings.data_processed

    # ================================================================
    # 1. 读取原始数据 + 重命名为统一英文列名
    # ================================================================
    print("=" * 40)
    print("  读取数据")
    print("=" * 40)

    # --- a_spot: 全市场实时行情 ---
    spot = spark.read.option("header", "true").option("inferSchema", "true") \
        .csv(f"{raw}/a_spot/*.csv") \
        .toDF("code", "name", "price", "change_amt", "change_pct", "bid", "ask",
              "prev_close", "open", "high", "low", "volume", "amount", "ts")
    print(f"  a_spot: {spot.count()} 条, {len(spot.columns)} 列")

    # --- a_hist: 沪深300 历史日线 ---
    hist_files = list(raw.glob("a_hist/*.csv"))
    if hist_files:
        hist = spark.read.option("header", "true").option("inferSchema", "true") \
            .csv(f"{raw}/a_hist/*.csv") \
            .toDF("date", "open", "close", "high", "low", "volume", "amount")
    else:
        hist = spark.createDataFrame(
            [], "date string, open double, close double, high double, low double, volume double, amount double"
        )
    print(f"  a_hist: {hist.count()} 条")

    # --- market_pe: 市场 PE ---
    pe = spark.read.option("header", "true").option("inferSchema", "true") \
        .csv(f"{raw}/market_pe/*.csv") \
        .toDF("date", "index_val", "avg_pe")
    print(f"  market_pe: {pe.count()} 条")

    # --- industry: 个股→行业映射 ---
    ind_files = list(raw.glob("industry/*.csv"))
    if ind_files:
        industry_map = spark.read.option("header", "true").option("inferSchema", "true") \
            .csv(f"{raw}/industry/*.csv")
        industry_map = industry_map.select(
            F.col("code"), F.col("industry").alias("industry_name")
        )
        print(f"  industry_map: {industry_map.count()} 条")
    else:
        industry_map = None

    # ================================================================
    # 2. 清洗
    # ================================================================
    print("\n" + "=" * 40)
    print("  清洗")
    print("=" * 40)

    spot_clean = spark.sql("""
        SELECT code, name,
            CAST(price AS double) AS price,
            CAST(change_pct AS double) AS change_pct,
            CAST(volume AS double) AS volume_num,
            CAST(amount AS double) AS amount_num
        FROM spot
        WHERE code IS NOT NULL AND price IS NOT NULL
          AND price > 0
    """)
    spot_clean.createOrReplaceTempView("spot_clean")
    print(f"  spot_clean: {spot_clean.count()} 条")

    hist_clean = spark.sql("""
        SELECT *,
            ROUND((close - open) / open * 100, 2) AS change_pct
        FROM hist
    """)
    hist_clean.createOrReplaceTempView("hist_clean")

    # PE：提取年份
    pe_clean = spark.sql("""
        SELECT *,
            CAST(SUBSTR(date, 1, 4) AS int) AS year
        FROM pe
    """)
    pe_clean.createOrReplaceTempView("pe_clean")

    # ================================================================
    # 3. 多维聚合分析
    # ================================================================
    print("\n" + "=" * 40)
    print("  分析")
    print("=" * 40)

    # --- 3.1 市场摘要（单行全景） ---
    print("\n--- 市场摘要 ---")
    market_summary = spark.sql("""
        SELECT
            SUM(CASE WHEN change_pct > 0 THEN 1 ELSE 0 END) AS up_count,
            SUM(CASE WHEN change_pct < 0 THEN 1 ELSE 0 END) AS down_count,
            SUM(CASE WHEN change_pct = 0 THEN 1 ELSE 0 END) AS flat_count,
            COUNT(*) AS total_count,
            ROUND(AVG(change_pct), 4) AS avg_change_pct,
            ROUND(STDDEV(change_pct), 4) AS std_change_pct,
            ROUND(SUM(amount_num), 0) AS total_amount
        FROM spot_clean
    """)
    market_summary.show(truncate=False)

    # --- 3.2 涨幅 TOP20 ---
    print("\n--- 涨幅 TOP20 ---")
    top_gainers = spark.sql("""
        SELECT code, name, change_pct
        FROM spot_clean ORDER BY change_pct DESC LIMIT 20
    """)
    top_gainers.show(truncate=False)

    # --- 3.3 跌幅 TOP20 ---
    print("\n--- 跌幅 TOP20 ---")
    top_losers = spark.sql("""
        SELECT code, name, change_pct
        FROM spot_clean ORDER BY change_pct ASC LIMIT 20
    """)
    top_losers.show(truncate=False)

    # --- 3.4 成交额 TOP20 ---
    print("\n--- 成交额 TOP20 ---")
    top_amount = spark.sql("""
        SELECT code, name, ROUND(amount_num, 0) AS amount, change_pct
        FROM spot_clean ORDER BY amount_num DESC LIMIT 20
    """)
    top_amount.show(truncate=False)

    # --- 3.5 PE 各年份均值 ---
    print("\n--- PE 年度均值 ---")
    pe_by_year = spark.sql("""
        SELECT year, ROUND(AVG(avg_pe), 2) AS avg_pe_val
        FROM pe_clean GROUP BY year ORDER BY year
    """)
    pe_by_year.show(30, truncate=False)

    # --- 3.6 沪深300 区间统计 ---
    print("\n--- 沪深300 区间统计 ---")
    index_stats = spark.sql("""
        SELECT
            ROUND(MIN(close), 2) AS close_min,
            ROUND(MAX(close), 2) AS close_max,
            ROUND(AVG(close), 2) AS close_avg,
            ROUND(STDDEV(change_pct), 2) AS volatility
        FROM hist_clean
    """)
    index_stats.show(truncate=False)

    # --- 3.7 行业板块聚合（NEW） ---
    print("\n--- 行业板块聚合 ---")
    if industry_map is not None:
        spot_with_industry = spot_clean.join(
            industry_map, on="code", how="left"
        )
        spot_with_industry.createOrReplaceTempView("spot_with_ind")

        sector_stats = spark.sql("""
            SELECT
                industry_name,
                COUNT(*) AS stock_count,
                ROUND(AVG(change_pct), 2) AS avg_change,
                ROUND(SUM(amount_num), 0) AS total_amount
            FROM spot_with_ind
            WHERE industry_name IS NOT NULL
            GROUP BY industry_name
            ORDER BY avg_change DESC
        """)
        sector_stats.show(20, truncate=False)
    else:
        # 无行业数据时创建空表
        sector_stats = spark.createDataFrame(
            [],
            "industry_name string, stock_count long, avg_change double, total_amount double"
        )
        print("  (无行业分类数据，生成空表)")
    sector_stats.createOrReplaceTempView("sector_stats")

    # --- 3.8 成交集中度（NEW） ---
    print("\n--- 成交集中度 ---")
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW spot_ranked AS
        SELECT *,
            ROW_NUMBER() OVER (ORDER BY amount_num DESC) AS rn,
            SUM(amount_num) OVER () AS total_market_amount
        FROM spot_clean
    """)

    concentration = spark.sql("""
        SELECT
            ROUND(
                SUM(CASE WHEN rn <= 10 THEN amount_num ELSE 0 END)
                / MAX(total_market_amount) * 100, 2
            ) AS top10_pct,
            ROUND(
                SUM(CASE WHEN rn <= 50 THEN amount_num ELSE 0 END)
                / MAX(total_market_amount) * 100, 2
            ) AS top50_pct
        FROM spot_ranked
    """)
    concentration.show(truncate=False)

    # ================================================================
    # 4. 写出结果
    # ================================================================
    print("\n" + "=" * 40)
    print("  写出结果")
    print("=" * 40)

    outputs = [
        ("market_summary", market_summary),
        ("top_gainers", top_gainers),
        ("top_losers", top_losers),
        ("top_amount", top_amount),
        ("pe_by_year", pe_by_year),
        ("index_stats", index_stats),
        ("sector_stats", sector_stats),
        ("concentration", concentration),
        ("spot_clean", spot_clean),
    ]

    for name, df in outputs:
        path = out / name
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(str(path))
        print(f"  [{name}] → {path}")

    print("\nSpark ETL 完成。")
    spark.stop()


# ================================================================
# Pandas 回退方案（Windows / 无 Java 环境自动使用）
# ================================================================
def _read_raw_csvs(raw: Path, date_glob: str = "*.csv"):
    """读取 raw 目录下所有 CSV 并合并"""
    files = list(raw.glob(date_glob))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _run_etl_pandas(settings: Settings):
    raw = settings.data_raw
    out = settings.data_processed
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. 读取 ----
    print("=" * 40)
    print("  [Pandas] 读取数据")
    print("=" * 40)

    spot = _read_raw_csvs(raw / "a_spot")
    if spot.empty:
        print("  ERROR 无 a_spot 数据，ETL 中止")
        return
    # 统一列名
    spot.columns = ["code", "name", "price", "change_amt", "change_pct", "bid", "ask",
                    "prev_close", "open", "high", "low", "volume", "amount", "ts"]
    spot["price"] = pd.to_numeric(spot["price"], errors="coerce")
    spot["change_pct"] = pd.to_numeric(spot["change_pct"], errors="coerce")
    spot["volume"] = pd.to_numeric(spot["volume"], errors="coerce")
    spot["amount"] = pd.to_numeric(spot["amount"], errors="coerce")
    spot = spot.dropna(subset=["code", "price"])
    spot = spot[spot["price"] > 0]
    print(f"  a_spot: {len(spot)} 条")

    hist = _read_raw_csvs(raw / "a_hist")
    if not hist.empty:
        hist.columns = ["date", "open", "close", "high", "low", "volume", "amount"]
        for c in ["open", "close", "high", "low", "volume", "amount"]:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
        hist["change_pct"] = (hist["close"] - hist["open"]) / hist["open"] * 100
    print(f"  a_hist: {len(hist)} 条")

    pe = _read_raw_csvs(raw / "market_pe")
    if not pe.empty:
        pe.columns = ["date", "index_val", "avg_pe"]
        pe["avg_pe"] = pd.to_numeric(pe["avg_pe"], errors="coerce")
        pe["year"] = pe["date"].astype(str).str[:4].astype(int)
    print(f"  market_pe: {len(pe)} 条")

    ind = _read_raw_csvs(raw / "industry")

    # ---- 2. 聚合分析 ----
    print("=" * 40)
    print("  [Pandas] 分析")
    print("=" * 40)

    total = len(spot)
    up = int((spot["change_pct"] > 0).sum())
    down = int((spot["change_pct"] < 0).sum())
    flat = int((spot["change_pct"] == 0).sum())

    market_summary = pd.DataFrame([{
        "up_count": up, "down_count": down, "flat_count": flat,
        "total_count": total,
        "avg_change_pct": round(float(spot["change_pct"].mean()), 4),
        "std_change_pct": round(float(spot["change_pct"].std()), 4),
        "total_amount": round(float(spot["amount"].sum()), 0),
    }])
    print(f"  市场摘要: ↑{up} ↓{down} —{flat}")

    top_gainers = spot.nlargest(20, "change_pct")[["code", "name", "change_pct"]]
    top_losers = spot.nsmallest(20, "change_pct")[["code", "name", "change_pct"]]
    top_amount = spot.nlargest(20, "amount")[["code", "name", "amount", "change_pct"]]
    top_amount["amount"] = top_amount["amount"].round(0)

    pe_by_year = pd.DataFrame()
    if not pe.empty:
        pe_by_year = pe.groupby("year")["avg_pe"].mean().round(2).reset_index()
        pe_by_year.columns = ["year", "avg_pe_val"]

    index_stats = pd.DataFrame()
    if not hist.empty:
        index_stats = pd.DataFrame([{
            "close_min": round(float(hist["close"].min()), 2),
            "close_max": round(float(hist["close"].max()), 2),
            "close_avg": round(float(hist["close"].mean()), 2),
            "volatility": round(float(hist["change_pct"].std()), 2),
        }])

    sector_stats = pd.DataFrame()
    if not ind.empty and "industry" in ind.columns:
        merged = spot.merge(ind[["code", "industry"]], on="code", how="left")
        merged = merged.dropna(subset=["industry"])
        if len(merged) > 0:
            sector_stats = merged.groupby("industry").agg(
                stock_count=("code", "count"),
                avg_change=("change_pct", "mean"),
                total_amount=("amount", "sum"),
            ).reset_index()
            sector_stats.columns = ["industry_name", "stock_count", "avg_change", "total_amount"]
            sector_stats["avg_change"] = sector_stats["avg_change"].round(2)
            sector_stats["total_amount"] = sector_stats["total_amount"].round(0)
            sector_stats = sector_stats.sort_values("avg_change", ascending=False)
    print(f"  板块聚合: {len(sector_stats)} 个行业")

    total_amt = float(spot["amount"].sum())
    top10_ratio = round(float(spot.nlargest(10, "amount")["amount"].sum()) / total_amt * 100, 2) if total_amt > 0 else 0
    top50_ratio = round(float(spot.nlargest(50, "amount")["amount"].sum()) / total_amt * 100, 2) if total_amt > 0 else 0
    concentration = pd.DataFrame([{"top10_pct": top10_ratio, "top50_pct": top50_ratio}])
    print(f"  集中度: TOP10={top10_ratio}%  TOP50={top50_ratio}%")

    spot_out = spot[["code", "name", "price", "change_pct", "volume", "amount"]].copy()
    spot_out.columns = ["code", "name", "price", "change_pct", "volume_num", "amount_num"]

    # ---- 3. 写出 ----
    print("=" * 40)
    print("  [Pandas] 写出结果")
    print("=" * 40)

    outputs = [
        ("market_summary", market_summary),
        ("top_gainers", top_gainers),
        ("top_losers", top_losers),
        ("top_amount", top_amount),
        ("pe_by_year", pe_by_year),
        ("index_stats", index_stats),
        ("sector_stats", sector_stats),
        ("concentration", concentration),
        ("spot_clean", spot_out),
    ]

    for name, df in outputs:
        path = out / name
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(path / "part-00000.csv", index=False)
        print(f"  [{name}] → {path}")

    print("\nPandas ETL 完成。")


# ================================================================
# 命令行入口
# ================================================================
if __name__ == "__main__":
    settings = Settings()
    run_etl(settings)
