"""
工具函数 —— 被所有模块复用，不写重复代码。
"""

import time
from pathlib import Path
from datetime import datetime


def retry(func, name: str, max_tries: int = 3):
    """
    带指数退避的重试机制。

    为什么需要？
    - akshare 接口偶尔抖动，直接抛异常会让整个管道挂掉
    - 间隔翻倍（2s → 4s → 8s），给上游恢复时间
    - 最后一次失败才真正抛异常

    用法：
        df = retry(lambda: ak.stock_zh_a_spot(), "a_spot")
    """
    wait = 2
    for i in range(max_tries):
        try:
            return func()
        except Exception as e:
            if i == max_tries - 1:
                raise e
            print(f"  [{name}] 第{i+1}次失败，{wait}秒后重试... ({e})")
            time.sleep(wait)
            wait *= 2


def today_str() -> str:
    """返回今天日期字符串 YYYYMMDD —— 用于文件命名、目录查找"""
    return datetime.today().strftime("%Y%m%d")


def read_processed_csv(dir_path: Path, subdir: str) -> "pd.DataFrame":
    """
    读取 Spark 输出的单个 CSV 文件。

    Spark 用 .coalesce(1).csv() 写入目录，会产出一个
    part-00000-xxx.csv 文件，名字不确定，需要用 glob 找到它。

    用法：
        df = read_processed_csv(settings.data_processed, "top_gainers")
    """
    import pandas as pd

    csv_files = list(dir_path.glob(f"{subdir}/part-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"没有找到 {dir_path / subdir}/part-*.csv")
    return pd.read_csv(csv_files[0])
