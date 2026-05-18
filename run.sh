#!/usr/bin/env bash
set -e

LOG="data/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date '+%Y-%m-%d %H:%M:%S') ASTRA 日报开始 ==="
python main.py
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ASTRA 日报完成 ==="
