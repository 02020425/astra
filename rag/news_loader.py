"""
新闻增量入库 —— 读 JSONL → 嵌入 → ChromaDB upsert

每天爬虫产出一个 JSONL 文件（data/raw/news/YYYYMMDD.jsonl），
这个脚本负责把它嵌入并存入向量库。
"""

import json
from pathlib import Path

from config.settings import Settings
from rag.embedder import DashScopeEmbedder
from rag.vector_store import VectorStore, make_id


def load_daily_news(
    date_str: str,
    embedder: DashScopeEmbedder,
    store: VectorStore,
    news_dir: Path,
) -> int:
    """
    读取一天的新闻 JSONL，嵌入后存入 ChromaDB。

    参数：
        date_str: "20260518"
        embedder: DashScope 嵌入客户端
        store: ChromaDB 向量库
        news_dir: data/raw/news/ 目录

    返回：新增入库的文档数
    """
    news_file = news_dir / f"{date_str}.jsonl"
    if not news_file.exists():
        print(f"[NewsLoader] 文件不存在，跳过: {news_file}")
        return 0

    # 1. 读取 JSONL
    articles = []
    with open(news_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[NewsLoader] 跳过损坏行: {e}")

    if not articles:
        print("[NewsLoader] 没有可处理的文章")
        return 0

    print(f"[NewsLoader] 读取到 {len(articles)} 条新闻")

    # 2. 构建文本（标题 + 内容），去重
    texts = []
    ids = []
    metadatas = []

    for a in articles:
        url = a.get("url", "")
        if not url:
            continue
        doc_id = make_id(url)

        # 用 标题 + 内容 作为嵌入文本
        text = f"{a.get('title', '')}\n{a.get('content', '')}"
        if len(text) < 20:  # 跳过空内容
            continue

        texts.append(text)
        ids.append(doc_id)
        metadatas.append({
            "title": a.get("title", "")[:200],
            "source": a.get("source", "unknown"),
            "date": a.get("date", date_str),
            "url": url,
        })

    if not texts:
        print("[NewsLoader] 去重后无新增文档")
        return 0

    # 3. 批量嵌入
    print(f"[NewsLoader] 正在嵌入 {len(texts)} 条...")
    embeddings = embedder.embed(texts)

    # 4. 存入 ChromaDB（upsert = 同 id 自动覆盖）
    store.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"[NewsLoader] 完成，入库 {len(texts)} 条。当前知识库总量: {store.count()}")
    return len(texts)
