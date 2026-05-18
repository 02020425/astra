"""
全局配置中枢 —— 所有模块从这里拿参数，不用到处散落 os.getenv()。

核心：Pydantic BaseSettings 自动从 .env 文件 + 环境变量读取。
类型校验：dashscope_api_key 没设？启动直接报错，不会跑到一半才挂。
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """全局配置。实例化：settings = Settings()  自动读 .env"""

    # ============================================================
    # API
    # ============================================================
    dashscope_api_key: str
    #   ↑ 没有默认值 → 必填 → .env 里没写或环境变量没设，启动就报错
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    #   ↑ 有默认值 → 可选 → 绝大多数人不需要改

    # ============================================================
    # 模型
    # ============================================================
    analysis_model: str = "qwen-plus"
    #   4 个分析师 + 主编 用的模型
    lightweight_model: str = "qwen-turbo"
    #   查询改写、评估（LLM-as-judge）用的模型，比 plus 快且便宜
    embedding_model: str = "text-embedding-v2"
    #   DashScope 文本嵌入模型，输出 1536 维向量

    # ============================================================
    # 路径 —— 全部相对于项目根目录
    # ============================================================
    project_root: Path = Path(__file__).parent.parent
    #   config/ 的父目录 = 项目根

    data_raw: Path = project_root / "data" / "raw"
    data_processed: Path = project_root / "data" / "processed"
    data_reports: Path = project_root / "data" / "reports"
    data_chroma: Path = project_root / "data" / "chroma"
    data_eval_cards: Path = project_root / "data" / "eval_cards"

    # ============================================================
    # RAG 参数
    # ============================================================
    chroma_collection: str = "financial_news"
    rag_top_k: int = 10
    #   检索返回的最相关文档数
    embedding_batch_size: int = 20
    #   每批嵌入多少条文本（DashScope 限制单次最多 25）

    # ============================================================
    # Agent 参数
    # ============================================================
    agent_temperature: float = 0.7
    #   0 = 确定性输出，1 = 随机性强。0.7 是分析类任务的常见选择
    agent_max_tokens: int = 2048
    #   单个 Agent 输出上限。分析类 2048 足够，日报类 4096 更安全

    model_config = {"env_file": ".env"}
    #   告诉 BaseSettings 去读项目根目录的 .env 文件
