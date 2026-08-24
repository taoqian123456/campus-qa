# ---------- 构建镜像 ----------
# 基础镜像：Python 3.11 slim（faiss-cpu / rapidocr / pypdfium2 都有 Linux wheel）
FROM python:3.11-slim

# RapidOCR（opencv/onnxruntime）与 pypdfium2 需要的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖清单并安装（利用 Docker 构建缓存：改代码不重装依赖）
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 复制代码（.dockerignore 已排除 .env/venv/数据库/索引等本地文件）
COPY backend /app/backend
COPY frontend /app/frontend

WORKDIR /app/backend

# 容器内数据目录（docker-compose 挂载卷到 /data；本地跑不影响，默认用 backend 下目录）
ENV DATABASE_URL=sqlite:////data/campus_qa.db \
    UPLOAD_DIR=/data/uploads \
    FAISS_INDEX_DIR=/data/faiss_index

# 入口：先初始化数据库（幂等），再启动服务（0.0.0.0 供容器外访问）
CMD ["sh", "-c", "python init_db.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
