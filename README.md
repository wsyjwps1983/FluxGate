# Semantic Router

[![GitHub](https://img.shields.io/github/license/aurelio-labs/semantic-router)](https://github.com/aurelio-labs/semantic-router/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/semantic-router.svg)](https://badge.fury.io/py/semantic-router)
[![CI](https://github.com/aurelio-labs/semantic-router/actions/workflows/test.yml/badge.svg)](https://github.com/aurelio-labs/semantic-router/actions/workflows/test.yml)
[![Python Versions](https://img.shields.io/pypi/pyversions/semantic-router.svg)](https://pypi.org/project/semantic-router/)

一个强大的路由系统，使用语义搜索和机器学习来路由用户查询，支持多种编码模型和索引后端。

## 特性

- **灵活的路由系统**：基于语义相似度和可选阈值路由查询
- **支持多种编码模型**：OpenAI、Cohere、HuggingFace、本地模型等
- **多种索引后端**：FAISS、Pinecone、Qdrant、PostgreSQL等
- **本地执行**：支持完全离线运行的本地编码模型
- **多模态支持**：处理文本、图像等多种输入类型
- **与主流LLM集成**：OpenAI、Mistral、Cohere等

## 架构概览

Semantic Router 由几个核心组件组成：

1. **编码器(Encoders)**：将文本或图像转换为向量表示
2. **索引(Indexes)**：存储和检索向量嵌入
3. **路由器(Routers)**：根据嵌入相似度路由查询
4. **路由(Routes)**：定义特定的路由规则和处理逻辑

## 快速启动

### 安装

```bash
pip install semantic-router
```

### 基本使用

```python
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder

# 定义路由
search_route = Route(
    name="search",
    utterances=[
        "我想搜索一个产品",
        "你能帮我找一下相关信息吗",
        "搜索最新的新闻"
    ],
    description="搜索相关信息的查询"
)

chat_route = Route(
    name="chat",
    utterances=[
        "你好，今天怎么样？",
        "我们来聊聊天吧",
        "有什么有趣的事情吗"
    ],
    description="闲聊性质的对话"
)

# 创建路由器
encoder = OpenAIEncoder()  # 需要设置 OPENAI_API_KEY 环境变量
router = SemanticRouter(encoder=encoder)
router.add([search_route, chat_route])

# 路由查询
result = router(
    "我想查找关于人工智能的最新研究"
)
print(f"路由到: {result.name}")
print(f"相似度分数: {result.score}")
```

## Dify 集成

Semantic Router 可以与 Dify 平台无缝集成，增强对话机器人的意图识别能力。

### 集成步骤

1. 在 Dify 中创建自定义工具
2. 将 Semantic Router 部署为 API 服务
3. 配置 Dify 调用该 API 进行意图识别

详细的集成指南可以在文档中找到。

## 路线图

- [x] 基础语义路由功能
- [x] 支持多种编码器
- [x] 支持多种索引后端
- [x] 本地执行支持
- [x] 多模态支持
- [ ] 更高级的路由策略
- [ ] 改进的性能优化
- [ ] 可视化管理界面

## 资源

- [文档](docs/)
- [API 参考](docs/API_USAGE_MANUAL.md)
- [贡献指南](docs/CONTRIBUTING.md)
- [部署指南](docs/DEPLOYMENT_GUIDE.md)

## 贡献

欢迎提交 Issue 和 Pull Request！请查看 [贡献指南](docs/CONTRIBUTING.md) 了解更多详情。

## 许可证

该项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。