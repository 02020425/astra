"""
PySpark ETL —— 结构化数据清洗 + 多维聚合分析。

输入：data/raw/（爬虫产出的 CSV）
输出：data/processed/（8 份分析结果 CSV）

本地运行：python spark_jobs/etl.py
Spark Submit：spark-submit spark_jobs/etl.py
"""

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config.settings import Settings


# ================================================================
# 初始化
# ================================================================
def run_etl(settings: Settings):
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
# 命令行入口
# ================================================================
if __name__ == "__main__":
    settings = Settings()
    run_etl(settings)
