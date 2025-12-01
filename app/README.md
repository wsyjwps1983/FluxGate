# SiliconFlow 意图识别系统

基于 SiliconFlow 的语义路由系统，支持从 Excel 文件加载训练数据，训练意图识别模型，并进行预测。

## 功能特点

- 📊 支持从 Excel 文件加载训练数据
- 🤖 集成 SiliconFlow 大语言模型和嵌入模型
- 💾 支持模型保存和加载
- 🎯 高准确率的意图识别
- 📈 完整的模型评估功能
- 🔧 命令行和交互式两种使用方式

## 安装依赖

```bash
pip install pandas openpyxl
```

## 设置 API 密钥

在使用前，需要设置 SiliconFlow API 密钥：

```bash
export SILICONFLOW_API_KEY="your_api_key_here"
```

或者在运行时通过 `--api-key` 参数提供。

## 数据格式

训练数据应该是 Excel 文件（.xlsx），包含两列：
- `intent`: 意图名称
- `utterance`: 用户语句

示例：

| intent   | utterance              |
|----------|------------------------|
| 技术问题 | 什么是机器学习？       |
| 技术问题 | 深度学习有什么应用？   |
| 产品问题 | 如何申请API密钥？      |
| 其他问题 | 你好                   |

## 快速开始

### 1. 使用快速开始脚本

```bash
python app/quick_start.py
```

这个脚本会：
- 加载 `intent_train.xlsx` 文件中的训练数据
- 训练一个意图识别模型
- 保存模型到 `app/models/` 目录
- 启动交互式预测模式

### 2. 使用命令行工具

#### 训练模型

```bash
python -m app.main train \
    --data-path intent_train.xlsx \
    --encoder-name BAAI/bge-large-zh-v1.5 \
    --threshold 0.3
```

#### 预测单个查询

```bash
python -m app.main predict \
    --model app/models/intent_router_20241125_143022.pkl \
    --query "什么是机器学习？"
```

#### 交互式预测

```bash
python -m app.main predict \
    --model app/models/intent_router_20241125_143022.pkl \
    --interactive
```

#### 批量预测

```bash
# 创建查询文件
echo -e "什么是机器学习？\n如何申请API密钥？\n推荐学习资料" > queries.txt

# 批量预测
python -m app.main predict \
    --model app/models/intent_router_20241125_143022.pkl \
    --batch-file queries.txt
```

#### 评估模型

```bash
python -m app.main evaluate \
    --model app/models/intent_router_20241125_143022.pkl \
    --test-data intent_train.xlsx
```

#### 列出可用模型

```bash
python -m app.main list
```

## 编程接口

### 训练模型

```python
from app import IntentRouterTrainer

# 创建训练器
trainer = IntentRouterTrainer(
    encoder_name="BAAI/bge-large-zh-v1.5",
    score_threshold=0.3
)

# 训练并保存模型
trainer.train_and_save(
    data_path="intent_train.xlsx",
    save_path="my_model.pkl"
)
```

### 加载模型并预测

```python
from app import IntentRouterPredictor

# 创建预测器
predictor = IntentRouterPredictor(model_path="my_model.pkl")

# 加载模型
predictor.load_model()

# 单个预测
intent = predictor.predict("什么是机器学习？")

# 批量预测
queries = ["查询1", "查询2", "查询3"]
results = predictor.predict_batch(queries)
```

### 数据加载和清理

```python
from app import load_intent_data, clean_data, validate_data

# 加载数据
raw_data = load_intent_data("intent_train.xlsx")

# 验证数据
is_valid, issues = validate_data(raw_data)

# 清理数据
clean_data = clean_data(raw_data)
```

## 文件结构

```
app/
├── __init__.py          # 包初始化文件
├── data_loader.py       # 数据加载模块
├── model_trainer.py     # 模型训练模块
├── model_predictor.py   # 模型预测模块
├── main.py             # 命令行主程序
├── quick_start.py      # 快速开始脚本
├── example_usage.py    # 使用示例
├── README.md           # 本文档
├── models/             # 模型保存目录
│   └── *.pkl           # 训练好的模型文件
└── results/            # 结果输出目录
    ├── predictions_*.csv  # 预测结果
    └── evaluation_*.json  # 评估结果
```

## 参数说明

### 训练参数

- `encoder_name`: 编码器模型名称，默认为 "BAAI/bge-large-zh-v1.5"
- `score_threshold`: 匹配阈值，默认为 0.3
- `api_key`: SiliconFlow API 密钥

### 预测参数

- `model`: 模型文件路径
- `query`: 单个查询文本
- `batch_file`: 批量查询文件路径
- `interactive`: 启用交互式模式

## 性能优化建议

1. **调整阈值**：根据准确率和召回率需求调整 `score_threshold`
2. **增加训练数据**：更多样的训练数据能提高模型泛化能力
3. **批量处理**：使用批量预测可以提高处理效率
4. **模型缓存**：加载一次模型后可以重复使用

## 常见问题

### 1. API 密钥问题

确保设置了正确的 SiliconFlow API 密钥：
- 通过环境变量设置
- 或通过 `--api-key` 参数提供

### 2. 数据格式问题

确保 Excel 文件包含正确的列名：
- 系统会尝试自动识别列名
- 如果识别失败，可以修改 `data_loader.py` 中的列名设置

### 3. 内存使用

大量训练数据可能会消耗较多内存，建议：
- 使用数据清理去除重复和无效数据
- 分批处理大量数据

## 更新日志

### v1.0.0
- 初始版本发布
- 支持从 Excel 加载训练数据
- 支持模型训练和保存
- 支持模型加载和预测
- 提供命令行和编程接口

## 许可证

本项目遵循与主项目相同的许可证。