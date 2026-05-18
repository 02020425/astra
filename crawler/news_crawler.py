"""
财经新闻爬取 —— 构建 RAG 知识库的数据来源。

三个来源：
  - stock_news_em: 东财个股新闻（成交量 TOP 20 的股票）
  - stock_news_main_cx: 财新宏观新闻（宏观/产业）
  - news_cctv: CCTV 财经新闻

输出格式：
  data/raw/news/YYYYMMDD.jsonl  —— 每行一条 JSON，便于逐行读取和追加
"""

import time
import json
from pathlib import Path

import akshare as ak
import pandas as pd

from config.settings import Settings
from utils.helpers import retry, today_str


class NewsCrawler:
    """财经新闻爬取器"""

    def __init__(self, settings: Settings):
        self.raw_dir = settings.data_raw
        self.today = today_str()

    def run_all(self, top_stock_codes: list[str] | None = None):
        """
        爬取全部三个来源的新闻。

        top_stock_codes: 成交量 TOP20 的股票代码列表（如果不传则跳过个股新闻）
        """
        print("=" * 40)
        print(f"  开始爬取财经新闻 ({self.today})")
        print("=" * 40)

        all_articles = []

        # 1. 个股新闻
        if top_stock_codes:
            articles = self._fetch_stock_news(top_stock_codes[:20])
            all_articles.extend(articles)
            print(f"  个股新闻: {len(articles)} 条")

        # 2. 财新宏观新闻
        try:
            articles = self._fetch_cx_news()
            all_articles.extend(articles)
            print(f"  财新宏观: {len(articles)} 条")
        except Exception as e:
            print(f"  财新宏观失败: {e}")

        # 3. CCTV 财经
        try:
            articles = self._fetch_cctv_news()
            all_articles.extend(articles)
            print(f"  CCTV财经: {len(articles)} 条")
        except Exception as e:
            print(f"  CCTV财经失败: {e}")

        # 保存
        if all_articles:
            self._save(all_articles)
            print(f"\n共 {len(all_articles)} 条新闻 → data/raw/news/{self.today}.jsonl")
        else:
            print("\n无新闻数据。")

    # ================================================================
    # 数据源 1: 东财个股新闻
    # ================================================================
    def _fetch_stock_news(self, codes: list[str]) -> list[dict]:
        articles = []
        for code in codes:
            try:
                df = retry(lambda: ak.stock_news_em(symbol=code), f"news_{code}")
                for _, row in df.iterrows():
                    articles.append({
                        "title": str(row.get("新闻标题", "")),
                        "content": str(row.get("新闻内容", "")),
                        "publish_time": str(row.get("发布时间", "")),
                        "source": "stock_news_em",
                        "stock_code": code,
                        "url": str(row.get("新闻链接", "")),
                        "crawl_date": self.today,
                    })
            except Exception as e:
                print(f"    {code} 新闻获取失败: {e}")
            time.sleep(0.3)
        return articles

    # ================================================================
    # 数据源 2: 财新宏观新闻
    # ================================================================
    def _fetch_cx_news(self) -> list[dict]:
        df = retry(lambda: ak.stock_news_main_cx(), "cx_news")
        if df is None or df.empty:
            return []
        articles = []
        for _, row in df.head(50).iterrows():
            tag = str(row.get("tag", ""))
            summary = str(row.get("summary", ""))
            articles.append({
                "title": summary[:100] if summary else tag,
                "content": summary,
                "publish_time": self.today,
                "source": "stock_news_main_cx",
                "stock_code": None,
                "url": str(row.get("url", f"cx_{self.today}_{len(articles)}")),
                "crawl_date": self.today,
            })
        return articles

    # ================================================================
    # 数据源 3: CCTV 财经新闻
    # ================================================================
    def _fetch_cctv_news(self) -> list[dict]:
        df = retry(lambda: ak.news_cctv(), "news_cctv")
        if df is None or df.empty:
            return []
        articles = []
        for _, row in df.head(20).iterrows():
            articles.append({
                "title": str(row.get("title", row.get("content", "")))[:100],
                "content": str(row.get("content", "")),
                "publish_time": str(row.get("date", "")),
                "source": "news_cctv",
                "stock_code": None,
                "url": str(row.get("url", f"cctv_{self.today}_{len(articles)}")),
                "crawl_date": self.today,
            })
        return articles

    # ================================================================
    # 保存
    # ================================================================
    def _save(self, articles: list[dict]):
        """保存为 JSONL —— 每行一条 JSON，便于流式读取"""
        news_dir = self.raw_dir / "news"
        news_dir.mkdir(parents=True, exist_ok=True)
        path = news_dir / f"{self.today}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for a in articles:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
