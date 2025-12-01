# Semantic Router Docker部署指南

## 1. 环境准备

### 1.1 安装Docker

#### Ubuntu/Debian系统
```bash
# 更新包列表
apt-get update

# 安装依赖包
apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common

# 添加Docker官方GPG密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

# 添加Docker仓库
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# 更新包列表并安装Docker
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# 启动Docker服务
systemctl start docker

# 设置Docker开机自启
systemctl enable docker

# 验证Docker安装
docker --version
```

#### CentOS/RHEL系统
```bash
# 安装依赖包
yum install -y yum-utils device-mapper-persistent-data lvm2

# 添加Docker仓库
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# 安装Docker
yum install -y docker-ce docker-ce-cli containerd.io

# 启动Docker服务
systemctl start docker

# 设置Docker开机自启
systemctl enable docker

# 验证Docker安装
docker --version
```

### 1.2 安装Docker Compose (可选)
```bash
# 下载Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# 添加执行权限
chmod +x /usr/local/bin/docker-compose

# 验证Docker Compose安装
docker-compose --version
```

## 2. 部署Semantic Router服务

### 2.1 克隆代码库
```bash
git clone <repository-url> semantic-router
cd semantic-router
```

### 2.2 构建Docker镜像
```bash
docker build -t semantic-router .
```

### 2.3 运行Docker容器

#### 基本运行
```bash
docker run -d -p 8000:8000 --name semantic-router semantic-router
```

#### 带环境变量配置
```bash
docker run -d \
  -p 8000:8000 \
  --name semantic-router \
  -e MODEL_NAME="all-MiniLM-L6-v2" \
  -e ALPHA="0.3" \
  semantic-router
```

#### 使用SiliconFlow向量化API
```bash
docker run -d \
  -p 8000:8000 \
  --name semantic-router \
  -e ENCODER_TYPE="siliconflow" \
  -e SILICONFLOW_API_KEY="your-siliconflow-api-key" \
  -e MODEL_NAME="BAAI/bge-large-zh-v1.5" \
  -e ALPHA="0.3" \
  semantic-router
```

#### 挂载数据卷 (可选，用于持久化数据)
```bash
docker run -d \
  -p 8000:8000 \
  --name semantic-router \
  -v ./final_training_data.json:/app/final_training_data.json \
  semantic-router
```

## 3. 验证部署

### 3.1 健康检查
```bash
curl http://localhost:8000/health
```

预期输出:
```json
{"status":"ok","message":"Semantic Router API is running","version":"1.0.0"}
```

### 3.2 路由检查
```bash
curl http://localhost:8000/routes
```

预期输出:
```json
{"routes":[{"name":"research_report",...}],"count":5}
```

### 3.3 预测测试
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query":"帮我总结附件中的内容","top_k":1}' http://localhost:8000/predict
```

预期输出:
```json
{"query":"帮我总结附件中的内容","intent":"summary","matched":true,"similarity_score":0.9987794160842896,"top_k_results":[{"intent":"summary","similarity_score":0.9987794160842896,"function_call":null}]}
```

### 3.4 日志检查
```bash
docker logs semantic-router
```

## 4. 服务管理

### 4.1 查看容器状态
```bash
docker ps
```

### 4.2 停止服务
```bash
docker stop semantic-router
```

### 4.3 重启服务
```bash
docker restart semantic-router
```

### 4.4 删除容器
```bash
docker rm semantic-router
```

### 4.5 更新服务
```bash
# 停止并删除旧容器
docker stop semantic-router
docker rm semantic-router

# 重新构建镜像
docker build -t semantic-router .

# 运行新容器
docker run -d -p 8000:8000 --name semantic-router semantic-router
```

## 5. 高级配置

### 5.1 使用Docker Compose

创建`docker-compose.yml`文件:
```yaml
version: '3.8'

services:
  semantic-router:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=all-MiniLM-L6-v2
      - ALPHA=0.3
    restart: unless-stopped
    volumes:
      - ./final_training_data.json:/app/final_training_data.json
```

启动服务:
```bash
docker-compose up -d
```

### 5.2 使用SiliconFlow向量化API

#### 5.2.1 支持的模型

SiliconFlowEncoder支持以下模型:

- **中文模型**:
  - `BAAI/bge-large-zh-v1.5` (推荐)
  - `BAAI/bge-base-zh-v1.5`
  - `BAAI/bge-small-zh-v1.5`

- **英文模型**:
  - `BAAI/bge-large-en-v1.5`
  - `text-embedding-ada-002`
  - `text-embedding-3-small`
  - `text-embedding-3-large`

#### 5.2.2 Docker Compose配置示例

```yaml
version: '3.8'

services:
  semantic-router:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENCODER_TYPE=siliconflow
      - SILICONFLOW_API_KEY=your-siliconflow-api-key
      - MODEL_NAME=BAAI/bge-large-zh-v1.5
      - ALPHA=0.3
      - SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
    restart: unless-stopped
    volumes:
      - ./final_training_data.json:/app/final_training_data.json
```

启动服务:
```bash
docker-compose up -d
```

#### 5.2.3 环境变量说明

| 环境变量名 | 类型 | 默认值 | 说明 |
|------------|------|--------|------|
| `ENCODER_TYPE` | string | `local` | 编码器类型，设置为`siliconflow`使用SiliconFlow向量化API |
| `SILICONFLOW_API_KEY` | string | - | SiliconFlow API密钥，必填 |
| `SILICONFLOW_BASE_URL` | string | `https://api.siliconflow.cn/v1` | SiliconFlow API基础URL |
| `MODEL_NAME` | string | `BAAI/bge-large-zh-v1.5` | 使用的模型名称 |
| `ALPHA` | float | `0.3` | 混合路由的权重参数 |

### 5.3 配置HTTPS

使用Nginx作为反向代理，配置SSL证书:

```bash
# 安装Nginx
apt-get install -y nginx

# 配置Nginx反向代理
cat > /etc/nginx/sites-available/semantic-router << EOF
server {
    listen 80;
    server_name your-domain.com;
    
    # 重定向HTTP到HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    # SSL证书配置
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    
    # 反向代理配置
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# 启用站点
ln -s /etc/nginx/sites-available/semantic-router /etc/nginx/sites-enabled/

# 测试Nginx配置
nginx -t

# 重启Nginx
systemctl restart nginx
```

## 6. 监控与维护

### 6.1 配置日志轮换

```bash
# 创建日志轮换配置
cat > /etc/logrotate.d/docker-containers << EOF
/var/lib/docker/containers/*/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    copytruncate
}
EOF
```

### 6.2 设置监控告警

使用Prometheus和Grafana监控Docker容器:

```bash
# 参考Prometheus和Grafana官方文档进行安装和配置
# https://prometheus.io/docs/introduction/overview/
# https://grafana.com/docs/grafana/latest/setup-grafana/
```

## 7. 故障排除

### 7.1 容器无法启动
```bash
# 查看容器日志
docker logs semantic-router

# 检查容器状态
docker inspect semantic-router
```

### 7.2 端口被占用
```bash
# 检查端口占用情况
netstat -tuln | grep 8000

# 或使用lsof
lsof -i :8000

# 停止占用端口的进程
kill -9 <pid>
```

### 7.3 内存不足
```bash
# 查看系统内存使用情况
free -h

# 查看容器内存使用情况
docker stats semantic-router

# 增加容器内存限制
docker run -d \
  -p 8000:8000 \
  --name semantic-router \
  --memory="4g" \
  semantic-router
```

## 8. 性能优化

### 8.1 调整工作进程数
```bash
docker run -d \
  -p 8000:8000 \
  --name semantic-router \
  semantic-router \
  uvicorn semantic_router.api.main:app --host 0.0.0.0 --port 8000 --workers 8
```

### 8.2 使用更高效的模型
```bash
docker run -d \
  -p 8000:8000 \
  --name semantic-router \
  -e MODEL_NAME="BAAI/bge-small-en-v1.5" \
  semantic-router
```

## 9. 安全最佳实践

### 9.1 使用非root用户运行容器

修改Dockerfile，添加非root用户:
```dockerfile
# 添加非root用户
RUN useradd -m -u 1000 semantic-router
USER semantic-router
```

### 9.2 限制容器权限
```bash
docker run -d \
  -p 8000:8000 \
  --name semantic-router \
  --cap-drop ALL \
  semantic-router
```

### 9.3 定期更新镜像
```bash
# 拉取最新代码
git pull

# 重新构建镜像
docker build -t semantic-router .

# 更新容器
docker stop semantic-router
docker rm semantic-router
docker run -d -p 8000:8000 --name semantic-router semantic-router
```

## 10. 总结

通过本指南，您可以在云服务器上成功部署Semantic Router服务。主要步骤包括:

1. 安装Docker和Docker Compose
2. 构建Semantic Router Docker镜像
3. 运行Docker容器
4. 验证服务正常运行
5. 配置高级功能和监控

如果您在部署过程中遇到任何问题，请参考故障排除部分或查看容器日志获取详细信息。
