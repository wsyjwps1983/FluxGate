# JSON格式模型使用指南

## 概述

我们已将模型保存格式从pickle更改为JSON，以解决序列化问题和提高兼容性。

## 主要变化

1. **模型保存格式**
   - 旧版本：使用pickle格式 (.pkl)
   - 新版本：使用JSON格式 (.json)，同时保存pickle副本用于快速加载

2. **文件扩展名**
   - 主要模型文件：`.json`
   - 辅助pickle文件：`.pkl`

3. **模型版本**
   - 旧版本：1.0 (仅pickle)
   - 新版本：2.0 (JSON为主，pickle为辅)

## 使用方法

### 训练模型

```bash
# 默认使用JSON格式保存
python -m app.main train --data-path intent_train.xlsx

# 模型将保存到 app/models/intent_router_YYYYMMDD_HHMMSS.json
```

### 列出模型

```bash
python -m app.main list

# 输出示例：
# 📂 可用模型列表
# ==================================================
# 📊 找到 3 个模型:
#   1. intent_router_20251125_171935.json (JSON)
#      路由数: 5
#      语句数: 35
#      版本: 2.0
#      训练时间: 2025-11-25T17:19:37.894156
```

### 加载和使用模型

```bash
# 使用JSON模型进行预测
python -m app.main predict --model app/models/intent_router_20251125_171935.json --query "你的查询"

# 交互式模式
python -m app.main predict --model app/models/intent_router_20251125_171935.json --interactive

# 批量预测
python -m app.main predict --model app/models/intent_router_20251125_171935.json --batch-file queries.txt
```

## 编程接口

### 训练和保存模型

```python
from app import IntentRouterTrainer

trainer = IntentRouterTrainer()
success = trainer.train_and_save(
    data_path="intent_train.xlsx",
    save_path="my_model.json"  # 将保存为JSON格式
)
```

### 加载模型

```python
from app import IntentRouterPredictor

# 自动检测JSON格式
predictor = IntentRouterPredictor(model_path="my_model.json")
predictor.load_model()

# 预测
result = predictor.predict("什么是机器学习？", return_details=True)
print(f"意图: {result['intent']}")
print(f"分数: {result['score']}")
```

## 文件结构

JSON模型文件包含以下字段：

```json
{
  "encoder_name": "BAAI/bge-large-zh-v1.5",
  "score_threshold": 0.3,
  "routes": [
    {
      "name": "intent_name",
      "utterances": ["示例语句1", "示例语句2"]
    }
  ],
  "training_data": {
    "intent_name": ["示例语句1", "示例语句2"]
  },
  "timestamp": "2025-11-25T17:19:37.894156",
  "version": "2.0",
  "compatibility": "semantic-router-v1",
  "metadata": {
    "num_routes": 5,
    "total_utterances": 35
  }
}
```

## 兼容性

- 系统同时支持JSON和pickle格式模型
- JSON模型加载时会自动重建路由器对象
- Pickle模型加载方式与之前相同

## 故障排除

1. **模型加载后预测不准确**
   - JSON模型加载后会重建路由器，可能需要重新预热
   - 尝试使用API密钥确保编码器正常工作

2. **找不到模型文件**
   - 确保使用正确的文件扩展名 (.json)
   - 检查文件路径是否正确

3. **模型保存失败**
   - 确保有足够的磁盘空间
   - 检查目录权限

## 优势

1. **避免序列化问题**：JSON格式不受pickle序列化限制
2. **更好的兼容性**：JSON是通用格式，易于跨语言使用
3. **可读性**：可以直接查看和编辑模型配置
4. **版本控制**：明确的版本号和兼容性信息

## 注意事项

1. JSON模型文件通常比pickle文件大
2. 加载JSON模型需要重建路由器对象，可能稍慢
3. 保存时同时创建JSON和Pickle两个文件以提供灵活性