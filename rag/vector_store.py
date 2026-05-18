"""
ChromaDB 向量存储 —— 持久化知识库。

为什么选 ChromaDB？
- PersistentClient：文件持久化，不需要单独起服务
- 支持 upsert：用 URL 的 md5 做 ID，重跑不会产生重复数据
- 余弦相似度：适合中文文本语义检索
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    """ChromaDB 集合管理 —— 增、查、统计"""

    def __init__(self, persist_path: Path, collection_name: str):
        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        """
        批量插入/更新。

        id = md5(URL)，同一条新闻重跑不会产生新记录。
        """
        if not ids:
            return
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> dict:
        """
        检索最相似的 top_k 条记录。

        返回 ChromaDB 原生格式：
        {
            "ids": [["id1", "id2", ...]],
            "documents": [["文本1", "文本2", ...]],
            "metadatas": [[{...}, {...}, ...]],
            "distances": [[0.15, 0.23, ...]],  # 余弦距离，越小越相似
        }
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        """知识库文档总数"""
        return self.collection.count()

    def peek(self, n: int = 3):
        """快速预览几条记录，调试用"""
        return self.collection.peek(limit=n)


def make_id(url: str) -> str:
    """用 URL 的 md5 做文档 ID，实现自然去重"""
    return hashlib.md5(url.encode()).hexdigest()
