# Docker 网络问题排查指南

## 1. 常见网络问题

### 1.1 镜像拉取超时

**错误信息**:
```
failed to resolve reference "docker.io/library/python:3.12-slim": failed to do request: Head "https://registry-1.docker.io/v2/library/python/manifests/3.12-slim": dial tcp 108.160.163.102:443: i/o timeout
```

**解决方案**:

1. **使用国内Docker镜像源**
   ```bash
   # 编辑Docker配置文件
   sudo mkdir -p /etc/docker
   sudo tee /etc/docker/daemon.json <<-'EOF'
   {
     "registry-mirrors": [
       "https://docker.mirrors.ustc.edu.cn",
       "https://hub-mirror.c.163.com",
       "https://mirror.baidubce.com",
       "https://ccr.ccs.tencentyun.com"
     ]
   }
   EOF
   
   # 重启Docker服务
   sudo systemctl daemon-reload
   sudo systemctl restart docker
   ```

2. **检查网络连接**
   ```bash
   # 检查DNS配置
   cat /etc/resolv.conf
   
   # 测试网络连通性
   ping -c 3 www.baidu.com
   
   # 测试Docker Hub连通性
   curl -v https://registry-1.docker.io
   ```

3. **使用代理服务器**
   ```bash
   # 设置Docker代理
   sudo mkdir -p /etc/systemd/system/docker.service.d
   sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<-'EOF'
   [Service]
   Environment="HTTP_PROXY=http://your-proxy:port/"
   Environment="HTTPS_PROXY=http://your-proxy:port/"
   Environment="NO_PROXY=localhost,127.0.0.1"
   EOF
   
   # 重启Docker服务
   sudo systemctl daemon-reload
   sudo systemctl restart docker
   ```

### 1.2 容器无法访问外部网络

**解决方案**:

1. **检查容器网络模式**
   ```bash
   # 查看容器网络
   docker network ls
   
   # 检查容器网络配置
   docker inspect <container-id> | grep -A 20 NetworkSettings
   ```

2. **重启Docker网络服务**
   ```bash
   sudo systemctl restart docker
   ```

3. **重置Docker网络**
   ```bash
   # 删除所有容器
   docker rm -f $(docker ps -aq)
   
   # 删除所有网络
   docker network prune -f
   
   # 重启Docker
   sudo systemctl restart docker
   ```

## 2. 离线部署方案

### 2.1 提前拉取镜像

在有网络的环境中拉取镜像并保存：

```bash
# 拉取镜像
docker pull python:3.12-slim

# 保存镜像到文件
docker save -o python-3.12-slim.tar python:3.12-slim
```

在目标服务器上加载镜像：

```bash
# 加载镜像
docker load -i python-3.12-slim.tar
```

### 2.2 使用本地Docker Registry

1. **启动本地Registry**
   ```bash
docker run -d -p 5000:5000 --name registry registry:2
   ```

2. **标记并推送镜像到本地Registry**
   ```bash
   # 标记镜像
docker tag python:3.12-slim localhost:5000/python:3.12-slim
   
   # 推送镜像
docker push localhost:5000/python:3.12-slim
   ```

3. **从本地Registry拉取镜像**
   ```bash
docker pull localhost:5000/python:3.12-slim
   ```

## 3. 优化Dockerfile以减少网络依赖

### 3.1 多阶段构建

```dockerfile
# 第一阶段：构建依赖
FROM python:3.12-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# 安装Python依赖到独立目录
RUN pip install --no-cache-dir --target=/app/deps -e .[local]
RUN pip install --no-cache-dir --target=/app/deps faiss-cpu fastapi uvicorn python-dotenv

# 第二阶段：运行时
FROM python:3.12-slim

WORKDIR /app

# 复制依赖
COPY --from=builder /app/deps /usr/local/lib/python3.12/site-packages

# 复制应用代码
COPY semantic_router /app/semantic_router
COPY final_training_data.json /app/

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME="all-MiniLM-L6-v2"
ENV ALPHA="0.3"

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "semantic_router.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 3.2 使用国内基础镜像

```dockerfile
# 使用国内镜像源的Python基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 使用国内apt源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 使用国内pip源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# 后续构建步骤...
```

## 4. 替代部署方案

### 4.1 使用Docker Compose

```yaml
version: '3.8'

services:
  semantic-router:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=all-MiniLM-L6-v2
      - ALPHA=0.3
    restart: unless-stopped
```

### 4.2 直接运行Python服务

如果Docker部署遇到困难，可以直接在服务器上运行Python服务：

```bash
# 安装依赖
pip install -e .[local]
pip install faiss-cpu fastapi uvicorn python-dotenv

# 启动服务
uvicorn semantic_router.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 5. 监控和日志

### 5.1 查看Docker日志

```bash
# 查看Docker守护进程日志
sudo journalctl -u docker.service -f

# 查看容器日志
docker logs -f <container-id>
```

### 5.2 检查Docker状态

```bash
# 查看Docker状态
sudo systemctl status docker

# 查看Docker信息
docker info
```

## 6. 其他常见问题

### 6.1 权限问题

**解决方案**:
```bash
# 将用户添加到docker组
sudo usermod -aG docker $USER

# 刷新组权限
newgrp docker
```

### 6.2 磁盘空间不足

**解决方案**:
```bash
# 清理未使用的镜像
docker image prune -a

# 清理未使用的容器
docker container prune

# 清理未使用的卷
docker volume prune

# 清理未使用的网络
docker network prune
```

## 7. 最佳实践

1. **定期更新Docker**
   ```bash
sudo apt-get update && sudo apt-get upgrade -y docker-ce docker-ce-cli containerd.io
   ```

2. **使用固定版本标签**
   ```dockerfile
   # 不推荐
   FROM python:slim
   
   # 推荐
   FROM python:3.12-slim
   ```

3. **最小化镜像大小**
   - 使用Alpine或Slim基础镜像
   - 清理临时文件
   - 使用多阶段构建

4. **使用健康检查**
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=3s \
     CMD curl -f http://localhost:8000/health || exit 1
   ```

5. **配置资源限制**
   ```dockerfile
   # 在docker run时添加
   docker run -d --name semantic-router \
     --memory="4g" \
     --cpus="2" \
     semantic-router
   ```

## 8. 联系方式

如果您遇到无法解决的问题，请联系技术支持团队，提供以下信息：

1. Docker版本：`docker --version`
2. 操作系统：`cat /etc/os-release`
3. 完整错误信息
4. Docker日志：`sudo journalctl -u docker.service -n 100`
5. 网络配置：`ifconfig` 或 `ip addr`

---

**文档更新时间**: 2025-11-30
**版本**: v1.0.0
