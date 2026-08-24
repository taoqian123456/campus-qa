#!/usr/bin/env bash
# 高校学生事务智能问答系统 - 一键启动（Linux/macOS）
set -e
echo "============================================"
echo " 高校学生事务智能问答系统 - 一键启动"
echo "============================================"
echo "[1/2] 正在构建镜像（首次约 5-10 分钟）..."
docker compose up -d --build
echo "[2/2] 启动完成！"
echo "  浏览器打开: http://localhost:8000"
echo "  常用命令: docker compose logs -f / down / restart"
