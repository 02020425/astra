"""
DashScope 文本嵌入 —— 把文本转成 1536 维向量。

DashScope 嵌入 API 不是 OpenAI 兼容的，需要用 httpx 直接调。
"""

import httpx
from config.settings import Settings


class DashScopeEmbedder:
    """
    封装 DashScope text-embedding-v2 API。

    一次调用最多 25 条文本，这里默认每批 20 条，留余量。
    """

    def __init__(self, settings: Settings):
        self.api_key = settings.dashscope_api_key
        self.model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.url = (
            "https://dashscope.aliyuncs.com/api/v1/services/"
            "embeddings/text-embedding/text-embedding"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        嵌入一批文本 → 返回向量列表。

        texts: ["新闻标题+内容1", "新闻标题+内容2", ...]
        返回: [[0.023, -0.451, ...], [0.187, 0.312, ...], ...]
              每个是 1536 维的 float 列表
        """
        if not texts:
            return []

        all_vectors = []
        # 分批发送，每批最多 batch_size 条
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self._embed_batch(batch)
            all_vectors.extend(vectors)

        return all_vectors

    def embed_single(self, text: str) -> list[float]:
        """嵌入单条文本"""
        return self.embed([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """发送一批文本到 DashScope"""
        resp = httpx.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": {"texts": texts},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["output"]["embeddings"]]
