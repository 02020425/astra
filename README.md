# ASTRA — Multi-Agent RAG A 股智能分析系统

**A-Share Trading RAG Analyst** — 基于 Multi-Agent + RAG 的全自动 A 股市场情报系统。

PySpark 做数据工程，LangGraph 编排多智能体，ChromaDB 构建知识库，Qwen 驱动分析。

## 架构

```
                        ┌── Gradio 交互问答 (Demo/演示用) ──┐
                        └───────────────────────────────────┘

akshare 爬虫 ──→ PySpark ETL ──→ RAG 检索 ──→ 4 Agent 并行分析 ──→ 主编汇总 ──→ JSON + MD 日报
     │                               ▲
     └──→ 财经新闻 ──→ Embedding ──→ ChromaDB 知识库
```

### 多智能体设计

| Agent | 关注领域 | 核心能力 |
|-------|---------|---------|
| 🏛️ 宏观分析师 | PE 估值、涨跌比、历史分位 | 估值定位 + 经济事件关联 |
| 📈 技术分析师 | 趋势、支撑阻力、指数 | 价格动量 + 技术信号识别 |
| 💰 资金面分析师 | 成交额 TOP、板块轮动 | 资金方向 + 集中度分析 |
| ⚠️ 风险分析师 | 波动率、外部风险、ST | 脆弱度评分 + 黑天鹅预警 |
| 📝 主编 | 综合汇总、矛盾消解 | 加权采纳 + 统一叙事 |

4 个 Agent 并行运行，主编等待全部完成后进行综合研判。

### 技术栈

| 层 | 技术 | 说明 |
|---|------|------|
| 数据采集 | akshare | A 股行情 + 财经新闻 |
| 数据工程 | PySpark 3.5 | 结构化 ETL，板块聚合，集中度分析 |
| 知识库 | ChromaDB + DashScope Embedding | 文本嵌入（1536维），增量去重入库 |
| RAG | Query Rewriting + 向量检索 | qwen-turbo 改写查询，语义检索新闻 |
| Agent 编排 | LangGraph | 并行扇出 + 汇总扇入，StateGraph |
| LLM | DashScope Qwen | qwen-plus（分析）/ qwen-turbo（轻量任务） |
| 结构化输出 | Pydantic v2 | 每个 Agent 输出 JSON Schema 校验 |
| 评估 | LLM-as-Judge | 检索精度、跨 Agent 一致性、报告质量 |
| 前端 | Gradio | 轻量交互问答（演示用） |
| 部署 | Docker + cron | 交易日下午自动运行 |

### 数据流

```
15:30 cron 触发
  │
  ├─[1] 爬虫 ──→ data/raw/（行情CSV + 新闻JSONL）
  ├─[2] PySpark ETL ──→ data/processed/（8份分析结果CSV）
  ├─[3] 知识库构建 ──→ ChromaDB 增量入库
  ├─[4] RAG 检索 ──→ 检索相关新闻作为分析背景
  ├─[5] 多 Agent 分析 ──→ 4 Agent 并行 + 主编汇总
  ├─[6] 输出 ──→ data/reports/（JSON + Markdown）
  └─[7] 评估 ──→ data/eval_cards/（质量卡）
```

## 快速开始

### 环境准备

```bash
# 1. 克隆并进入项目
cd astra

# 2. 安装 uv（不需要 Python，独立二进制文件）
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 安装 Python >= 3.12（如果本地没有）
uv python install 3.12

# 4. 安装依赖（自动创建 venv）
uv sync

# 5. 设置 API Key（二选一）
# 方式 A：环境变量（推荐，全局生效）
setx DASHSCOPE_API_KEY "你的Key"
# 方式 B：项目 .env 文件
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY

# 6. (仅 Spark 需要) 确保 Java 17+ 已安装
java -version
```

### 使用

```bash
# ---- CLI 模式：生成今日日报（6 阶段全流程）----
python main.py                # 分析今日
python main.py --date 20260518   # 分析指定日期
python main.py --crawl-only      # 只爬数据，不分析

# ---- Gradio 交互模式：浏览器问答 ----
python app.py
# 打开 http://localhost:7860
# 两种模式：
#   ① 对话框提问 — 4 个 AI 分析师 + 主编回答
#   ② 一键生成日报 — 等同于 python main.py
```

> 首次运行会自动修复 akshare 1.18.x 在 Python 3.12+ 下的兼容性问题，无需手动操作。

### Docker

```bash
docker compose up -d
# 容器内 cron 每个交易日下午 3:30 自动运行
# Gradio 界面: http://localhost:7860
```

## 项目结构

```
astra/
├── main.py                     # CLI 入口：python main.py
├── app.py                      # Gradio UI：python app.py
│
├── config/                     # 配置层
│   ├── settings.py             # Pydantic BaseSettings
│   └── agent_prompts/          # 5 个 Agent 的 System Prompt
│
├── crawler/                    # 数据采集层
│   ├── market_crawler.py       # 行情数据（个股/指数/PE/行业）
│   └── news_crawler.py         # 财经新闻（东财/金十/CCTV）
│
├── spark_jobs/
│   └── etl.py                  # PySpark 清洗 + 板块聚合 + 集中度
│
├── rag/                        # RAG 知识库层
│   ├── embedder.py             # DashScope 文本嵌入
│   ├── vector_store.py         # ChromaDB 向量存储
│   ├── news_loader.py          # 增量入库协调
│   └── retriever.py            # 查询改写 + 检索
│
├── agents/                     # 多智能体层
│   ├── schemas.py              # Pydantic v2 输出模型
│   ├── base.py                 # Agent 抽象基类
│   ├── macro_analyst.py        # 宏观分析师
│   ├── technical_analyst.py    # 技术分析师
│   ├── fund_flow_analyst.py    # 资金面分析师
│   ├── risk_analyst.py         # 风险分析师
│   ├── chief_editor.py         # 主编（汇总+矛盾消解）
│   └── graph.py                # LangGraph 编排
│
├── llm/
│   └── client.py               # DashScope OpenAI 兼容客户端
│
├── output/
│   └── writer.py               # JSON + Markdown 双输出
│
├── eval/                       # 评估层
│   ├── metrics.py              # 检索精度/一致性/质量
│   └── eval_card.py            # 日度评估卡
│
├── utils/
│   └── helpers.py              # 工具函数
│
└── data/                       # 运行时生成（gitignore）
    ├── raw/                    # 原始爬取数据
    ├── processed/              # Spark ETL 结果
    ├── reports/                # 日报输出
    ├── chroma/                 # 向量数据库
    └── eval_cards/             # 评估卡
```

## 简历定位

**岗位方向**: AI Engineer / LLM Application Engineer / MLOps

**一句话描述**: 基于 Multi-Agent + RAG 的全自动 A 股金融情报系统，融合 PySpark 数据工程与 LangGraph 智能体编排

**关键技术词**: RAG, Multi-Agent, LangGraph, ChromaDB, PySpark, Pydantic, Structured Output, LLM-as-Judge, Financial NLP, Docker

## License

MIT
