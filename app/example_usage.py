#!/usr/bin/env python3
"""
使用示例脚本
展示如何使用意图识别系统的各个组件
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app import load_intent_data, clean_data, IntentRouterTrainer, IntentRouterPredictor

def example_data_loading():
    """示例：数据加载"""
    print("📊 示例：数据加载")
    print("-" * 40)
    
    # 设置数据路径
    data_path = os.path.join(project_root, "intent_train.xlsx")
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    # 加载原始数据
    raw_data = load_intent_data(data_path)
    print(f"📂 加载了 {len(raw_data)} 个意图类别")
    
    # 清理数据
    cleaned_data = clean_data(raw_data)
    print(f"🧹 清理后剩余 {len(cleaned_data)} 个意图类别")
    
    # 显示数据统计
    for intent, utterances in cleaned_data.items():
        print(f"  📂 {intent}: {len(utterances)} 条语句")
    
    return cleaned_data

def example_training(api_key):
    """示例：模型训练"""
    print("\n🚀 示例：模型训练")
    print("-" * 40)
    
    # 创建训练器
    trainer = IntentRouterTrainer(api_key=api_key)
    
    # 加载数据
    data_path = os.path.join(project_root, "intent_train.xlsx")
    if not trainer.load_data(data_path):
        print("❌ 数据加载失败")
        return None
    
    # 创建路由
    routes = trainer.create_routes()
    
    # 初始化编码器
    trainer.initialize_encoder()
    
    # 预热编码器
    trainer.preheat_encoder()
    
    # 训练路由器
    router = trainer.train_router()
    
    # 评估性能
    accuracy = trainer.evaluate_router()
    
    # 保存模型
    models_dir = os.path.join(project_root, "app", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "example_model.pkl")
    if trainer.save_model(model_path):
        print(f"✅ 模型已保存到: {model_path}")
        return model_path
    else:
        return None

def example_prediction(model_path, api_key):
    """示例：模型预测"""
    print("\n🤖 示例：模型预测")
    print("-" * 40)
    
    # 创建预测器
    predictor = IntentRouterPredictor(model_path=model_path, api_key=api_key)
    
    # 加载模型
    if not predictor.load_model():
        print("❌ 模型加载失败")
        return
    
    # 单个预测
    query = "什么是机器学习？"
    result = predictor.predict(query, return_details=True)
    intent = result['intent']
    score = result.get('score', None)
    
    print(f"🔍 单个预测:")
    print(f"  查询: {query}")
    print(f"  意图: {intent}")
    print(f"  分数: {score}")
    
    # 批量预测
    queries = [
        "如何申请API密钥？",
        "推荐学习资料",
        "天气怎么样",
        "深度学习有什么应用"
    ]
    
    results = predictor.predict_batch(queries, return_details=True)
    
    print(f"\n📊 批量预测:")
    for query, result in zip(queries, results):
        intent = result['intent']
        print(f"  '{query}' -> {intent or 'None'}")
    
    # 评估模型
    print(f"\n📈 评估模型:")
    
    # 创建测试数据
    test_data = [
        ("机器学习是什么", "技术问题"),
        ("如何使用API", "产品问题"),
        ("AI最新研究", "学术问题"),
        ("再见", "其他问题")
    ]
    
    evaluation = predictor.evaluate_with_test_data(test_data)
    print(f"  总体准确率: {evaluation['accuracy']:.1f}%")

def main():
    """主函数"""
    print("🎯 SiliconFlow 意图识别系统 - 使用示例")
    print("=" * 60)
    
    # 检查API密钥
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("⚠️  未检测到 API 密钥")
        api_key = input("请输入 SiliconFlow API 密钥: ").strip()
        if not api_key:
            print("❌ API 密钥不能为空")
            return
        os.environ["SILICONFLOW_API_KEY"] = api_key
    
    # 1. 数据加载示例
    training_data = example_data_loading()
    
    if not training_data:
        print("❌ 无法继续示例，因为数据加载失败")
        return
    
    # 2. 模型训练示例
    model_path = example_training(api_key)
    
    if not model_path:
        print("❌ 无法继续示例，因为模型训练失败")
        return
    
    # 3. 模型预测示例
    example_prediction(model_path, api_key)
    
    print("\n🎉 示例演示完成！")
    print("\n💡 接下来你可以：")
    print("  • 使用 quick_start.py 体验完整流程")
    print("  • 使用 main.py 进行命令行操作")
    print("  • 在自己的代码中导入 app 模块使用")

if __name__ == "__main__":
    main()