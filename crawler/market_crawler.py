"""
A 股行情数据爬取 —— akshare 接口封装。

爬取内容：
  - 实时行情（全市场个股）
  - 历史日线（沪深300）
  - 市场 PE/PB
  - 行业分类映射（个股 → 行业板块）
  - 板块资金流向

存到 data/raw/{分类}/{YYYYMMDD}.csv，每天一个文件。
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

from config.settings import Settings
from utils.helpers import retry, today_str


class MarketCrawler:
    """行情数据爬取器"""

    def __init__(self, settings: Settings):
        self.raw_dir = settings.data_raw
        self.today = today_str()

    # ================================================================
    # 1. 全市场实时行情（Sina 源，比东方财富轻量）
    # ================================================================
    def fetch_a_spot(self):
        if self._skip("a_spot"):
            return
        print("正在获取 A股实时行情...")
        df = retry(lambda: ak.stock_zh_a_spot(), "a_spot")
        print(f"  获取到 {len(df)} 只股票")
        self._save(df, "a_spot")

    # ================================================================
    # 2. 沪深300 历史日线
    # ================================================================
    def fetch_a_hist(self):
        if self._skip("a_hist"):
            return
        start = (datetime.today() - timedelta(days=365)).strftime("%Y%m%d")
        end = self.today
        print(f"正在获取 沪深300 历史日线 ({start} ~ {end})...")
        df = retry(
            lambda: ak.stock_zh_index_daily_em(
                symbol="sh000300", start_date=start, end_date=end
            ),
            "a_hist",
        )
        print(f"  获取到 {len(df)} 条")
        self._save(df, "a_hist")

    # ================================================================
    # 3. 市场 PE/PB
    # ================================================================
    def fetch_market_pe(self):
        if self._skip("market_pe"):
            return
        print("正在获取 市场 PE/PB...")
        df = ak.stock_market_pe_lg()
        print(f"  获取到 {len(df)} 条")
        self._save(df, "market_pe")

    # ================================================================
    # 4. 行业分类映射（NEW）
    # ================================================================
    def fetch_industry_map(self):
        """获取所有行业板块的成分股，建立 个股→行业 映射"""
        if self._skip("industry"):
            return
        print("正在获取 行业分类...")

        try:
            # 获取所有行业板块名称
            boards = ak.stock_board_industry_name_em()
            print(f"  获取到 {len(boards)} 个行业板块")

            rows = []
            for _, row in boards.head(90).iterrows():  # 限制板块数，避免超时
                board_name = row["板块名称"]
                try:
                    cons = ak.stock_board_industry_cons_em(symbol=board_name)
                    for _, s in cons.iterrows():
                        rows.append({
                            "code": s["代码"],
                            "name": s["名称"],
                            "industry": board_name,
                        })
                except Exception:
                    continue
                time.sleep(0.3)  # 控制频率

            if rows:
                df = pd.DataFrame(rows)
                print(f"  建立 {len(rows)} 条 个股→行业 映射")
                self._save(df, "industry")
            else:
                print("  (无数据)")
        except Exception as e:
            print(f"  行业分类获取失败: {e}")

    # ================================================================
    # 5. 板块资金流向（NEW）
    # ================================================================
    def fetch_sector_flow(self):
        """获取板块资金流向排名"""
        if self._skip("sector_flow"):
            return
        print("正在获取 板块资金流向...")
        try:
            df = ak.stock_sector_fund_flow_rank(indicator="今日")
            print(f"  获取到 {len(df)} 个板块")
            self._save(df, "sector_flow")
        except Exception as e:
            print(f"  板块资金流向获取失败: {e}")

    # ================================================================
    # 一键全跑
    # ================================================================
    def run_all(self):
        print("=" * 40)
        print(f"  开始爬取行情数据 ({self.today})")
        print("=" * 40)

        tasks = [
            ("实时行情", self.fetch_a_spot),
            ("历史日线", self.fetch_a_hist),
            ("市场PE", self.fetch_market_pe),
            ("行业分类", self.fetch_industry_map),
            ("板块资金流", self.fetch_sector_flow),
        ]

        for name, task in tasks:
            try:
                task()
            except Exception as e:
                print(f"  [{name}] 失败: {e}")
            time.sleep(1.5)  # 请求间隔，避免被限流

        print("\n行情数据爬取完成。")

    # ================================================================
    # 内部工具
    # ================================================================
    def _skip(self, name: str) -> bool:
        """今天已爬则跳过"""
        out_dir = self.raw_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        return (out_dir / f"{self.today}.csv").exists()

    def _save(self, df: pd.DataFrame, name: str):
        """保存为 CSV"""
        out_dir = self.raw_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.today}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"  [{name}] → {path}")
