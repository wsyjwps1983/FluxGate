# 使用阿里云的Python 3.12-slim镜像
FROM registry.cn-hangzhou.aliyuncs.com/library/python:3.12-slim

# 设置工作目录
WORKDIR /app

# 使用阿里云的apt源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME="all-MiniLM-L6-v2"
ENV ALPHA="0.3"
# SiliconFlow API配置
ENV SILICONFLOW_API_KEY=""
ENV SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"
# 设置pip使用阿里云镜像
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

# 复制必要的项目文件
COPY semantic_router /app/semantic_router
COPY final_training_data.json /app/
COPY pyproject.toml /app/
COPY README.md /app/

# 安装Python依赖
RUN pip install --no-cache-dir -e .[local]
RUN pip install --no-cache-dir faiss-cpu fastapi uvicorn python-dotenv

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "semantic_router.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
