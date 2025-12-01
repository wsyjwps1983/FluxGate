# 意图验证使用指南

## 概述

本指南介绍如何使用提供的脚本进行意图验证，包括：
1. 启动FastAPI服务
2. 使用初始意图数据初始化路由
3. 使用验证数据进行预测和评估

## 准备工作

### 1. 安装依赖

确保已安装所需依赖：

```bash
pip install fastapi uvicorn requests pandas openpyxl
```

### 2. 数据文件

确保以下文件存在：
- **初始意图数据**: `/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/final_training_data.json`
- **验证数据**: `/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/intent_train.xlsx`

## 使用步骤

### 1. 启动FastAPI服务

首先启动FastAPI服务：

```bash
python -m semantic_router.api.main
```

服务将在 `http://localhost:8000` 运行。

### 2. 初始化意图路由

使用 `init_intents.py` 脚本将初始意图数据添加到路由中：

```bash
python validate_intent.py
```

该脚本将：
- 读取 `final_training_data.json` 中的初始意图数据
- 按意图名称分组，收集每个意图的所有问题
- 为每个意图创建路由
- 调用API添加路由

### 3. 运行验证脚本

使用 `validate_intent.py` 脚本读取验证数据并逐条调用API进行预测：

```bash
python validate_intent.py
```

该脚本将：
- 读取 `intent_train.xlsx` 中的验证数据
- 逐条调用API进行预测
- 计算预测准确率
- 将结果保存到 `validation_results.json`

## 脚本说明

### init_intents.py

**功能**: 使用初始意图数据初始化路由

**主要步骤**:
1. 读取初始意图数据
2. 按意图名称分组
3. 创建路由数据
4. 调用API添加路由
5. 验证路由添加结果

### validate_intent.py

**功能**: 逐条读取验证数据调用API进行预测

**主要步骤**:
1. 读取初始意图数据（用于参考）
2. 读取验证数据
3. 逐条调用API预测
4. 计算准确率
5. 保存结果到文件

## API接口说明

### POST /predict

**功能**: 预测文本的意图

**请求体**:
```json
{
  "query": "文本内容",
  "top_k": 1
}
```

**响应体**:
```json
{
  "query": "文本内容",
  "intent": "预测的意图名称",
  "matched": true/false,
  "similarity_score": 0.85,
  "top_k_results": [
    {
      "intent": "意图名称",
      "similarity_score": 0.85,
      "function_call": null
    }
  ]
}
```

### POST /routes

**功能**: 添加新路由

**请求体**:
```json
{
  "name": "意图名称",
  "utterances": ["样本1", "样本2"],
  "score_threshold": 0.7,
  "description": "意图描述",
  "metadata": {}
}
```

### GET /routes

**功能**: 获取所有路由

**响应体**:
```json
{
  "routes": [
    {
      "name": "意图名称",
      "utterances": ["样本1", "样本2"],
      "score_threshold": 0.7,
      "description": "意图描述",
      "metadata": {}
    }
  ],
  "count": 1
}
```

## 结果分析

验证结果将保存到 `validation_results.json`，包含：

```json
{
  "timestamp": "2025-11-28 14:30:00",
  "total_count": 100,
  "correct_count": 85,
  "accuracy": 0.85,
  "results": [
    {
      "index": 1,
      "question": "问题内容",
      "true_intent": "真实意图",
      "predicted_intent": "预测意图",
      "is_correct": true,
      "similarity_score": 0.85,
      "matched": true,
      "top_k_results": [...],
      "error": null
    }
  ]
}
```

## 常见问题

### 1. API连接失败

**解决方法**:
- 确保FastAPI服务正在运行
- 检查API_URL配置是否正确
- 检查网络连接

### 2. Excel文件读取失败

**解决方法**:
- 确保Excel文件存在且路径正确
- 确保Excel文件包含"question"和"intent"列
- 确保已安装openpyxl库

### 3. 预测准确率低

**解决方法**:
- 调整路由的score_threshold参数
- 添加更多的样本数据
- 检查意图分类是否合理

## 示例输出

### 初始化脚本输出

```
=== 意图初始化脚本 ===

1. 读取初始意图数据: /path/to/final_training_data.json
   共 182 条初始意图数据

2. 按意图分组
   共 4 个意图类型
   - research_report: 9 条样本
   - work_tools: 14 条样本
   - creative_case: 7 条样本
   - summary: 7 条样本
   - other: 7 条样本

3. 创建路由数据
   共创建 5 个路由

4. 调用API添加路由
   添加路由成功 research_report
   添加路由成功 work_tools
   添加路由成功 creative_case
   添加路由成功 summary
   添加路由成功 other

5. 初始化完成
   成功添加 5/5 个路由

6. 验证路由添加结果
   当前路由总数: 5
   路由列表: ['research_report', 'work_tools', 'creative_case', 'summary', 'other']
```

### 验证脚本输出

```
=== 意图验证脚本 ===

1. 读取初始意图数据: /path/to/final_training_data.json
   共 182 条初始意图

2. 读取验证数据: /path/to/intent_train.xlsx
   成功读取验证数据，共 100 条记录

3. 开始预测，共 100 条数据
   预测第 1/100: 我们要做一个连锁餐饮行业的消费者洞察报告...
   预测第 2/100: 想要通过人群洞察，来指导奈雪2026年的产品策略...
   ...

4. 预测完成
   总条数: 100
   正确条数: 85
   准确率: 85.00%

5. 保存结果到: /path/to/validation_results.json
   结果保存成功
```

## 扩展建议

1. **批量预测**: 可以修改脚本支持批量预测，提高效率
2. **实时监控**: 添加实时监控功能，监控预测准确率
3. **自动调优**: 根据验证结果自动调整阈值
4. **可视化**: 添加可视化功能，展示预测结果
5. **多模型对比**: 支持对比不同模型的预测结果

## 联系方式

如有问题，请联系相关技术人员。