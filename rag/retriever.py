"""
RAG 检索器 —— 查询改写 + 向量检索 + 上下文格式化。

流程：
  market_data → query_rewriting (qwen-turbo) → 嵌入 → ChromaDB 查询 → 格式化为上下文
"""

from config.settings import Settings
from llm.client import LLMClient
from rag.embedder import DashScopeEmbedder
from rag.vector_store import VectorStore


class Retriever:
    """
    执行 RAG 检索全流程。

    用法：
        retriever = Retriever(settings, llm, embedder, store)
        context = retriever.retrieve(market_data)
        # context 是一段可直接嵌入 Agent prompt 的格式化文本
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        embedder: DashScopeEmbedder,
        store: VectorStore,
    ):
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.top_k = settings.rag_top_k

    def retrieve(self, market_data: dict) -> str:
        """
        从市场数据出发，检索相关新闻，返回格式化的上下文文本。

        1. 用 qwen-turbo 把结构化数据转成自然语言检索查询
        2. 嵌入查询
        3. 在 ChromaDB 中搜索 top_k 条
        4. 格式化为可嵌入 prompt 的文本
        """
        # Step 1: 查询改写
        query = self._rewrite_query(market_data)
        print(f"[Retriever] 检索查询: {query}")

        # Step 2: 嵌入查询
        query_vec = self.embedder.embed_single(query)

        # Step 3: 向量检索
        results = self.store.query(query_vec, top_k=self.top_k)

        # Step 4: 格式化
        return self._format_results(results)

    def retrieve_by_question(self, question: str) -> str:
        """
        根据用户问题直接检索（Graio 问答模式）。

        不需要查询改写，用户的问题本身就是自然语言。
        """
        query_vec = self.embedder.embed_single(question)
        results = self.store.query(query_vec, top_k=self.top_k)
        return self._format_results(results)

    # ================================================================
    # 查询改写 —— 市场数据 → 自然语言检索词
    # ================================================================
    def _rewrite_query(self, market_data: dict) -> str:
        """
        为什么需要改写？
        市场数据是数字："涨 1990 跌 3350 avg -0.19"
        ChromaDB 里是新闻文本："央行降准...半导体走强..."
        直接嵌入数字去搜语义文本 = 搜不到。

        qwen-turbo 把数字转成人话检索词，语义匹配效果大幅提升。
        """
        spot = market_data.get("spot_summary", {})
        top_gainers = market_data.get("top_gainers", [])[:5]
        top_amount = market_data.get("top_amount", [])[:5]

        gainer_names = [s.get("name", "") for s in top_gainers]
        amount_names = [s.get("name", "") for s in top_amount]

        prompt = f"""今日A股市场概况：
上涨 {spot.get('up_count','?')} 家，下跌 {spot.get('down_count','?')} 家
平均涨跌幅 {spot.get('avg_change_pct','?')}%

涨幅前列: {'、'.join(gainer_names)}
成交额前列: {'、'.join(amount_names)}

请根据以上信息生成一个中文检索查询（15-30字），
用于在财经新闻库中搜索最相关的背景新闻。
直接输出查询，不要其他文字。"""

        return self.llm.chat(
            model=self.settings.lightweight_model,
            system="你是一个搜索查询生成器。根据A股市场数据生成中文检索词。",
            user=prompt,
            max_tokens=60,
            temperature=0.3,  # 低温，更确定
        ).strip()

    # ================================================================
    # 结果格式化
    # ================================================================
    def _format_results(self, results: dict) -> str:
        """
        把 ChromaDB 原始结果格式化为 Agent prompt 可用的文本。

        格式：
        【新闻标题】(来源 | 日期)
        新闻内容...
        """
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return "（暂无相关新闻）"

        lines = [f"以下是通过 RAG 检索到的 {len(docs)} 条相关财经新闻：\n"]
        for i, (doc_id, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, distances), 1
        ):
            # 余弦距离 → 相似度分数
            similarity = max(0, 1 - dist)
            title = meta.get("title", "无标题")
            source = meta.get("source", "?")
            date = meta.get("date", "?")
            lines.append(
                f"---\n"
                f"【{i}】{title}\n"
                f"来源: {source} | 日期: {date} | 相关度: {similarity:.0%}\n"
                f"{doc[:300]}"  # 截断过长内容
            )

        return "\n".join(lines)
