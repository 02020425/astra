FROM python:3.12-slim

# Java 17（PySpark 依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk cron && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 依赖层（优先 COPY，利用 Docker 缓存）
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    'pyspark>=3.5,<4' akshare pandas pyarrow openpyxl \
    openai langgraph chromadb 'pydantic>=2' 'pydantic-settings>=2' \
    httpx gradio

# 代码层
COPY . .

# 数据持久化目录
RUN mkdir -p /app/data/raw /app/data/processed /app/data/reports \
    /app/data/chroma /app/data/eval_cards

# 定时任务：交易日 15:30 自动跑日报
RUN echo "30 15 * * 1-5 root cd /app && bash run.sh >> /app/data/cron.log 2>&1" \
    > /etc/cron.d/astra && chmod 0644 /etc/cron.d/astra

# 同时启动 cron + Gradio 前端
EXPOSE 7860
CMD ["sh", "-c", "cron && python app.py"]
