# Semantic Router API 使用手册

## 1. 概述

Semantic Router API 是一个基于语义的意图识别服务，能够根据用户输入的文本内容，自动识别其意图类型，并返回相关的相似度得分和匹配结果。该服务支持多种意图类型，包括研究报告、工作工具、创意案例、总结和其他类型。

## 2. 快速开始

### 2.1 服务地址

```
http://your-server-ip:8000
```

### 2.2 健康检查

```bash
curl http://your-server-ip:8000/health
```

预期响应：
```json
{
  "status": "ok",
  "message": "Semantic Router API is running",
  "version": "1.0.0"
}
```

## 3. API 端点

### 3.1 意图预测

#### 3.1.1 端点信息

- **URL**: `/predict`
- **方法**: `POST`
- **描述**: 预测输入文本的意图类型
- **Content-Type**: `application/json`

#### 3.1.2 请求参数

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
|--------|------|------|--------|------|
| query | string | 是 | - | 要预测的文本内容 |
| top_k | integer | 否 | 1 | 返回的匹配结果数量 |

#### 3.1.3 请求示例

```bash
curl -X POST -H "Content-Type: application/json" -d '{"query":"帮我总结附件中的内容，输出该行业的总结","top_k":1}' http://your-server-ip:8000/predict
```

#### 3.1.4 响应示例

```json
{
  "query": "帮我总结附件中的内容，输出该行业的总结",
  "intent": "summary",
  "matched": true,
  "similarity_score": 0.9987794160842896,
  "top_k_results": [
    {
      "intent": "summary",
      "similarity_score": 0.9987794160842896,
      "function_call": null
    }
  ]
}
```

### 3.2 获取所有路由

#### 3.2.1 端点信息

- **URL**: `/routes`
- **方法**: `GET`
- **描述**: 获取所有已配置的意图路由

#### 3.2.2 请求示例

```bash
curl http://your-server-ip:8000/routes
```

#### 3.2.3 响应示例

```json
{
  "routes": [
    {
      "name": "research_report",
      "utterances": [
        "我们要做一个连锁餐饮行业的消费者洞察报告...",
        "想要通过人群洞察，来指导奈雪2026年的产品策略..."
      ],
      "score_threshold": 0.7,
      "description": "research_report意图路由",
      "metadata": {}
    },
    // 其他路由...
  ],
  "count": 5
}
```

### 3.3 添加新路由

#### 3.3.1 端点信息

- **URL**: `/routes`
- **方法**: `POST`
- **描述**: 添加新的意图路由
- **Content-Type**: `application/json`

#### 3.3.2 请求参数

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
|--------|------|------|--------|------|
| name | string | 是 | - | 路由名称 |
| utterances | array | 是 | - | 该意图的示例文本列表 |
| score_threshold | number | 否 | 0.7 | 匹配阈值 |
| description | string | 否 | - | 路由描述 |
| metadata | object | 否 | {} | 附加元数据 |

#### 3.3.3 请求示例

```bash
curl -X POST -H "Content-Type: application/json" -d '{"name":"new_intent","utterances":["这是一个新意图的示例","另一个新意图的示例"],"score_threshold":0.75,"description":"新意图路由","metadata":{"category":"custom"}}' http://your-server-ip:8000/routes
```

#### 3.3.4 响应示例

```json
{
  "status": "success",
  "message": "路由 new_intent 添加成功",
  "route": {
    "name": "new_intent",
    "utterances": ["这是一个新意图的示例", "另一个新意图的示例"],
    "score_threshold": 0.75,
    "description": "新意图路由",
    "metadata": {"category": "custom"}
  }
}
```

### 3.4 更新路由

#### 3.4.1 端点信息

- **URL**: `/routes/{name}`
- **方法**: `PUT`
- **描述**: 更新指定名称的路由
- **Content-Type**: `application/json`

#### 3.4.2 请求参数

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
|--------|------|------|--------|------|
| name | string | 是 | - | 路由名称（在URL中） |
| utterances | array | 否 | - | 该意图的示例文本列表 |
| score_threshold | number | 否 | - | 匹配阈值 |
| description | string | 否 | - | 路由描述 |
| metadata | object | 否 | - | 附加元数据 |

#### 3.4.3 请求示例

```bash
curl -X PUT -H "Content-Type: application/json" -d '{"utterances":["更新后的示例1","更新后的示例2"],"score_threshold":0.8}' http://your-server-ip:8000/routes/new_intent
```

#### 3.4.4 响应示例

```json
{
  "status": "success",
  "message": "路由 new_intent 更新成功"
}
```

### 3.5 删除路由

#### 3.5.1 端点信息

- **URL**: `/routes/{name}`
- **方法**: `DELETE`
- **描述**: 删除指定名称的路由

#### 3.5.2 请求示例

```bash
curl -X DELETE http://your-server-ip:8000/routes/new_intent
```

#### 3.5.3 响应示例

```json
{
  "status": "success",
  "message": "路由 new_intent 删除成功"
}
```

### 3.6 获取统计信息

#### 3.6.1 端点信息

- **URL**: `/stats`
- **方法**: `GET`
- **描述**: 获取服务统计信息

#### 3.6.2 请求示例

```bash
curl http://your-server-ip:8000/stats
```

#### 3.6.3 响应示例

```json
{
  "total_routes": 5,
  "total_utterances": 45,
  "index_type": "faiss",
  "encoder_type": "LocalEncoder",
  "sparse_encoder_type": "BM25Encoder",
  "alpha": 0.3,
  "score_threshold": 0.7
}
```

## 4. 错误处理

### 4.1 常见错误码

| 错误码 | 描述 |
|--------|------|
| 400 | 请求参数错误 |
| 404 | 路由不存在 |
| 500 | 服务器内部错误 |

### 4.2 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

## 5. 代码示例

### 5.1 Python 示例

```python
import requests
import json

# 服务地址
url = "http://your-server-ip:8000/predict"

# 请求数据
payload = {
    "query": "帮我总结附件中的内容，输出该行业的总结",
    "top_k": 1
}

# 发送请求
response = requests.post(url, json=payload)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print(f"意图类型: {result['intent']}")
    print(f"匹配状态: {result['matched']}")
    print(f"相似度得分: {result['similarity_score']}")
    print(f"Top-k结果: {json.dumps(result['top_k_results'], ensure_ascii=False, indent=2)}")
else:
    print(f"请求失败: {response.status_code} {response.text}")
```

### 5.2 JavaScript 示例

```javascript
// 使用 fetch API
const url = 'http://your-server-ip:8000/predict';
const data = {
    query: '帮我总结附件中的内容，输出该行业的总结',
    top_k: 1
};

fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log('意图类型:', result.intent);
    console.log('匹配状态:', result.matched);
    console.log('相似度得分:', result.similarity_score);
    console.log('Top-k结果:', result.top_k_results);
})
.catch(error => {
    console.error('请求失败:', error);
});
```

### 5.3 Java 示例

```java
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public class SemanticRouterExample {
    public static void main(String[] args) {
        try {
            // 服务地址
            URL url = new URL("http://your-server-ip:8000/predict");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            
            // 设置请求方法和头信息
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            
            // 请求数据
            String jsonInputString = "{\"query\": \"帮我总结附件中的内容，输出该行业的总结\", \"top_k\": 1}";
            
            // 发送请求
            try(OutputStream os = conn.getOutputStream()) {
                byte[] input = jsonInputString.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }
            
            // 处理响应
            int responseCode = conn.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                String inputLine;
                StringBuilder response = new StringBuilder();
                
                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();
                
                System.out.println("响应结果: " + response.toString());
            } else {
                System.out.println("请求失败，响应码: " + responseCode);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## 6. 最佳实践

### 6.1 请求频率

- 建议控制请求频率，避免对服务器造成过大压力
- 对于批量请求，建议使用异步方式处理

### 6.2 输入文本长度

- 建议输入文本长度控制在 1000 字符以内
- 过长的文本可能会影响预测性能和准确性

### 6.3 意图设计

- 确保不同意图之间有明显的语义区分
- 为每个意图提供足够的示例文本（建议至少 5 条）
- 定期更新和优化意图示例

### 6.4 阈值调整

- 根据实际业务需求调整匹配阈值
- 较高的阈值可以提高准确率，但可能降低召回率
- 较低的阈值可以提高召回率，但可能降低准确率

## 7. 意图类型说明

### 7.1 research_report

- **描述**: 研究报告相关意图
- **示例**: "帮我做一个行业洞察报告", "分析市场趋势"

### 7.2 work_tools

- **描述**: 工作工具和方法论相关意图
- **示例**: "什么是麦肯锡金字塔模型", "如何进行SWOT分析"

### 7.3 creative_case

- **描述**: 创意案例相关意图
- **示例**: "如何策划一个成功的营销活动", "餐饮品牌开业活动方案"

### 7.4 summary

- **描述**: 总结相关意图
- **示例**: "帮我总结附件内容", "提炼核心观点"

### 7.5 other

- **描述**: 其他类型意图
- **示例**: "今天天气怎么样", "推荐一本好书"

## 8. 版本更新日志

### v1.0.0 (2025-11-30)

- 初始版本发布
- 支持意图预测功能
- 支持路由管理功能
- 支持统计信息查询

## 9. 联系方式

如有问题或建议，请联系技术支持团队。

---

**文档更新时间**: 2025-11-30
**版本**: v1.0.0
