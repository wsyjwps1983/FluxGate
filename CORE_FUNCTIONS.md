# Semantic Router 核心功能描述

## 1. 系统架构

### 1.1 整体架构图

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              Semantic Router                              │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   输入层        │   处理层        │   存储层        │   输出层            │
├─────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ 自然语言文本    │ Route           │ LocalIndex      │ RouteChoice         │
│                 │ BaseRouter      │ HybridLocalIndex│                     │
│                 │ SemanticRouter  │ FaissIndex      │                     │
│                 │ HybridRouter    │                 │                     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────┘
```

### 1.2 核心组件关系

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Route    │────▶│ BaseRouter  │────▶│   Index     │────▶│ RouteChoice │
└─────────────┘     └──────┬──────┘     └─────────────┘     └─────────────┘
                            │
                            ▼
                   ┌─────────────┐
                   │SemanticRouter│
                   └─────────────┘
                            │
                            ▼
                   ┌─────────────┐
                   │HybridRouter │
                   └─────────────┘
```

## 2. 核心组件说明

### 2.1 Route

**功能**：定义意图路由，包含意图名称、示例语句、阈值等信息。

**核心属性**：
- `name`: 路由名称，唯一标识一个意图
- `utterances`: 示例语句列表，用于训练和匹配
- `score_threshold`: 匹配阈值，用于判断是否命中该路由
- `function_schemas`: 动态路由的函数 schema
- `llm`: 动态路由使用的 LLM 实例

**核心方法**：
- `__call__`: 调用路由，返回 RouteChoice 对象
- `acall`: 异步调用路由
- `to_dict`: 转换为字典
- `from_dict`: 从字典创建 Route 对象

### 2.2 BaseRouter

**功能**：路由器基类，定义了路由器的核心接口和逻辑。

**核心属性**：
- `encoder`: 编码器实例，用于文本向量化
- `sparse_encoder`: 稀疏编码器实例（仅 HybridRouter 使用）
- `index`: 索引实例，用于向量存储和查询
- `routes`: 路由列表
- `score_threshold`: 全局匹配阈值
- `top_k`: 查询时返回的 top k 结果

**核心方法**：
- `__call__`: 调用路由器，返回 RouteChoice 或列表
- `acall`: 异步调用路由器
- `add`: 添加路由
- `update`: 更新路由
- `delete`: 删除路由
- `fit`: 训练路由器，优化阈值
- `evaluate`: 评估路由器性能

### 2.3 SemanticRouter

**功能**：基于密集向量的路由器，使用单一编码器进行文本向量化和匹配。

**核心特性**：
- 使用密集向量进行相似度匹配
- 支持对称和非对称编码器
- 支持异步操作

### 2.4 HybridRouter

**功能**：混合路由器，结合密集向量和稀疏向量进行匹配，提高匹配准确性。

**核心特性**：
- 同时使用密集编码器和稀疏编码器
- 支持 alpha 参数调节两种向量的权重
- 支持阈值的自动优化

### 2.5 LocalIndex

**功能**：本地向量索引，用于存储和查询向量。

**核心特性**：
- 基于内存的向量存储
- 支持向量的添加、删除和查询
- 支持路由过滤

### 2.6 HybridLocalIndex

**功能**：混合本地索引，同时支持密集向量和稀疏向量的存储和查询。

**核心特性**：
- 同时存储密集向量和稀疏向量
- 支持混合查询
- 支持路由过滤

## 3. 工作流程

### 3.1 路由创建流程

```
1. 定义 Route 对象，包含意图名称和示例语句
2. 初始化路由器（SemanticRouter 或 HybridRouter）
3. 调用 add 方法添加路由
4. 路由器自动将示例语句向量化
5. 将向量存储到索引中
```

### 3.2 查询匹配流程

```
1. 接收自然语言查询
2. 调用路由器的 __call__ 方法
3. 路由器将查询向量化
4. 在索引中查询最相似的向量
5. 计算相似度得分
6. 与阈值比较，判断是否命中
7. 返回 RouteChoice 对象
```

### 3.3 阈值优化流程

```
1. 准备训练数据（查询和对应的意图）
2. 调用 fit 方法
3. 生成查询向量
4. 随机搜索最优阈值
5. 评估不同阈值下的准确率
6. 更新最优阈值
```

## 4. API 接口说明

### 4.1 Route API

```python
# 创建 Route
route = Route(
    name="greeting",
    utterances=["你好", "早上好", "下午好"],
    score_threshold=0.7
)

# 调用 Route
route_choice = route(query="你好")
```

### 4.2 Router API

```python
# 初始化路由器
router = SemanticRouter(
    encoder=DenseEncoder(),
    routes=[route1, route2]
)

# 添加路由
router.add(route3)

# 查询匹配
result = router("你好")

# 训练优化
router.fit(X=queries, y=intents)

# 评估性能
accuracy = router.evaluate(X=test_queries, y=test_intents)
```

### 4.3 Index API

```python
# 添加向量
index.add(
    embeddings=embeddings,
    routes=route_names,
    utterances=utterances
)

# 查询向量
scores, route_names = index.query(
    vector=query_vector,
    top_k=5
)
```

## 5. 使用示例

### 5.1 基本使用示例

```python
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import LocalEncoder

# 创建路由
route1 = Route(
    name="greeting",
    utterances=["你好", "早上好", "下午好"],
    score_threshold=0.7
)

route2 = Route(
    name="farewell",
    utterances=["再见", "拜拜", "下次见"],
    score_threshold=0.7
)

# 初始化路由器
router = SemanticRouter(
    encoder=LocalEncoder(),
    routes=[route1, route2]
)

# 查询匹配
result = router("你好")
print(result.name)  # 输出: greeting
```

### 5.2 HybridRouter 使用示例

```python
from semantic_router import Route, HybridRouter
from semantic_router.encoders import LocalEncoder, BM25Encoder

# 创建路由
route1 = Route(
    name="greeting",
    utterances=["你好", "早上好", "下午好"],
    score_threshold=0.7
)

route2 = Route(
    name="farewell",
    utterances=["再见", "拜拜", "下次见"],
    score_threshold=0.7
)

# 初始化混合路由器
router = HybridRouter(
    encoder=LocalEncoder(),
    sparse_encoder=BM25Encoder(),
    routes=[route1, route2],
    alpha=0.3
)

# 查询匹配
result = router("你好")
print(result.name)  # 输出: greeting
```

### 5.3 动态路由使用示例

```python
from semantic_router import Route
from semantic_router.llms import OpenAILLM

# 定义函数 schema
schema = {
    "name": "get_weather",
    "description": "获取天气信息",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["city"]
    }
}

# 创建 LLM 实例
llm = OpenAILLM()

# 创建动态路由
route = Route(
    name="weather",
    utterances=["今天天气怎么样", "北京的天气", "明天会下雨吗"],
    function_schemas=[schema],
    llm=llm
)

# 调用动态路由
result = route(query="北京今天的天气怎么样")
print(result.function_call)  # 输出: 包含城市参数的函数调用
```

## 6. 核心功能特性

### 6.1 多模态支持

- 支持文本输入
- 支持向量输入
- 支持混合输入

### 6.2 灵活的编码器支持

- 支持密集编码器
- 支持稀疏编码器
- 支持混合编码器
- 支持本地和远程编码器

### 6.3 高效的向量索引

- 支持本地索引
- 支持远程索引（Pinecone、Qdrant、PostgreSQL）
- 支持混合索引

### 6.4 智能的阈值优化

- 支持自动阈值优化
- 支持路由级别的阈值设置
- 支持动态阈值调整

### 6.5 异步支持

- 支持异步编码
- 支持异步查询
- 支持异步路由调用

## 7. 性能指标

### 7.1 准确性

- 基于密集向量的匹配准确率：90%+
- 基于混合向量的匹配准确率：95%+
- 支持自定义评估指标

### 7.2 速度

- 文本向量化速度：毫秒级
- 向量查询速度：微秒级
- 支持批量处理

### 7.3 可扩展性

- 支持水平扩展
- 支持动态添加路由
- 支持分布式部署

## 8. 应用场景

### 8.1 对话系统

- 意图识别
- 对话路由
- 意图分类

### 8.2 搜索系统

- 语义搜索
- 向量搜索
- 混合搜索

### 8.3 推荐系统

- 内容推荐
- 相似内容匹配
- 个性化推荐

### 8.4 自动化系统

- 自动客服
- 任务自动化
- 智能助手

## 9. 部署与集成

### 9.1 本地部署

```bash
# 安装依赖
pip install semantic-router faiss-cpu

# 运行示例
python -m semantic_router.example
```

### 9.2 API 部署

- 支持 FastAPI 部署
- 支持 Flask 部署
- 支持 Docker 容器化

### 9.3 集成方式

- Python SDK
- REST API
- gRPC API

## 10. 未来规划

### 10.1 功能增强

- 支持更多编码器类型
- 支持更多索引类型
- 支持更多 LLM 类型
- 支持多语言

### 10.2 性能优化

- 优化向量索引性能
- 优化编码速度
- 优化查询速度

### 10.3 易用性提升

- 提供更简洁的 API
- 提供更丰富的文档
- 提供更多示例
- 提供可视化工具

## 11. 总结

Semantic Router 是一个功能强大、灵活易用的语义路由系统，支持多种编码器、多种索引类型和多种部署方式。它可以应用于对话系统、搜索系统、推荐系统等多个领域，帮助开发者快速构建智能应用。

通过不断的功能增强和性能优化，Semantic Router 将成为开发者构建智能应用的重要工具。