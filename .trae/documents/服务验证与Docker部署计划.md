# 服务验证与Docker部署优化计划

## 1. 服务完整性验证

### 1.1 核心功能验证

* [x] FastAPI服务正常运行

* [x] 意图预测功能正常

* [x] 路由管理功能正常

* [x] 健康检查端点正常

* [x] 统计信息端点正常

### 1.2 依赖验证

* [x] 核心依赖已配置

* [x] 本地模型依赖已配置

* [x] Faiss索引依赖已配置

### 1.3 数据验证

* [x] 初始意图数据文件存在

* [x] 数据格式正确

## 2. 配置优化

### 2.1 代码修改

**修改硬编码路径**

* **文件**: `semantic_router/api/main.py`

* **修改点**: 第38行，将绝对路径改为相对路径

* **修改前**:

  ```python
  file_path = "/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/final_training_data.json"
  ```

* **修改后**:

  ```python
  file_path = os.path.join(os.path.dirname(__file__), "../../final_training_data.json")
  ```

**添加环境变量支持**

* **文件**: `semantic_router/api/main.py`

* **修改点**: 添加环境变量配置支持

* **修改内容**:

  ```python
  import os
  from dotenv import load_dotenv

  # 加载环境变量
  load_dotenv()

  # 使用环境变量配置
  MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
  ALPHA = float(os.getenv("ALPHA", "0.3"))

  # 初始化编码器
  encoder = LocalEncoder(name=MODEL_NAME)
  ```

## 3. Dockerfile设计（详细）

```dockerfile
# 使用Python 3.12作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app

# 安装Python依赖
RUN pip install --no-cache-dir -e .[local]
RUN pip install --no-cache-dir faiss-cpu

# 安装FastAPI和uvicorn
RUN pip install --no-cache-dir fastapi uvicorn python-dotenv

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME="all-MiniLM-L6-v2"
ENV ALPHA="0.3"

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "semantic_router.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## 4. 部署验证计划

### 4.1 本地测试

1. **构建镜像**

   ```bash
   ```

docker build -t semantic-router .

````

2. **运行容器**
```bash
docker run -d -p 8000:8000 --name semantic-router semantic-router
````

1. **验证服务**

   ```bash
   # 健康检查
   curl http://localhost:8000/health

   # 路由检查
   curl http://localhost:8000/routes

   # 预测测试
   curl -X POST -H "Content-Type: application/json" -d '{"query":"帮我总结附件中的内容","top_k":1}' http://localhost:8000/predict
   ```

2. **日志检查**

   ```bash
   ```

docker logs semantic-router

````

### 4.2 云服务器部署

1. **推送镜像到容器注册表**
```bash
# 登录Docker Hub或私有注册表
docker login

# 标记镜像
docker tag semantic-router:latest <registry>/semantic-router:latest

# 推送镜像
docker push <registry>/semantic-router:latest
````

1. **在云服务器上运行**

   ```bash
   # 登录云服务器
   ssh user@cloud-server

   # 拉取镜像
   docker pull <registry>/semantic-router:latest

   # 运行容器
   docker run -d -p 8000:8000 --name semantic-router <registry>/semantic-router:latest
   ```

2. **云服务器验证**

   ```bash
   # 健康检查
   curl http://<cloud-server-ip>:8000/health

   # 完整功能测试
   python validate_intent.py --api-url http://<cloud-server-ip>:8000
   ```

## 5. 配置文件管理

### 5.1 环境变量配置

创建`.env.example`文件，包含所有必要的环境变量：

```
# 模型配置
MODEL_NAME=all-MiniLM-L6-v2
ALPHA=0.3

# 服务配置
HOST=0.0.0.0
PORT=8000
WORKERS=4

# 日志配置
LOG_LEVEL=INFO
```

## 6. 监控与维护

### 6.1 日志管理

* 配置Docker日志驱动

* 考虑使用ELK或Loki收集日志

### 6.2 健康检查

* 使用Docker健康检查

* 配置云平台健康检查

### 6.3 自动重启

* 设置容器自动重启策略

  ```bash
  ```

docker run -d --restart unless-stopped ...

````

## 7. 安全最佳实践

- [ ] 使用非root用户运行容器
- [ ] 限制容器资源使用
- [ ] 配置网络隔离
- [ ] 定期更新镜像
- [ ] 考虑使用HTTPS

## 8. 执行步骤

1. **修改代码**
 - 更新硬编码路径
 - 添加环境变量支持

2. **创建配置文件**
 - 创建`.env.example`
 - 更新`.env`

3. **构建Docker镜像**
 - 本地构建测试
 - 推送至镜像仓库

4. **部署到云服务器**
 - 拉取镜像
 - 运行容器
 - 验证部署

5. **监控与维护**
 - 配置日志收集
 - 设置健康检查
 - 配置自动重启

## 9. 回滚计划

1. **停止当前容器**
 ```bash
docker stop semantic-router
````

1. **移除当前容器**

   ```bash
   ```

docker rm semantic-router

````

3. **运行旧版本镜像**
```bash
docker run -d -p 8000:8000 --name semantic-router <registry>/semantic-router:old-version
````

1. **验证回滚**

   ```bash
   ```

curl http\://<cloud-server-ip>:8000/health

```

## 10. 性能优化

- 使用多workers运行uvicorn
- 配置Gunicorn + uvicorn
- 考虑使用CDN加速静态资源
- 优化模型加载时间

This optimized plan provides a comprehensive approach to verifying the service and deploying it to a cloud server using Docker, with detailed steps for configuration, testing, deployment, and maintenance.
```

