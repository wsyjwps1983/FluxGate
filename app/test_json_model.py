#!/usr/bin/env python3
"""
测试JSON格式模型的保存和加载
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app import IntentRouterTrainer, IntentRouterPredictor

def test_json_model():
    """测试JSON模型功能"""
    print("🧪 测试JSON格式模型保存和加载")
    print("=" * 50)
    
    # 1. 训练并保存模型
    print("1️⃣ 训练并保存模型")
    data_path = os.path.join(project_root, "intent_train.xlsx")
    models_dir = os.path.join(project_root, "app", "models")
    model_path = os.path.join(models_dir, "test_json_model.json")
    
    trainer = IntentRouterTrainer()
    success = trainer.train_with_threshold_optimization(data_path, model_path)  
    if not success:
        print("❌ 模型训练失败")
        return False
    
    # 2. 加载JSON模型
    print("\n2️⃣ 加载JSON模型")
    predictor = IntentRouterPredictor(model_path=model_path)
    
    if not predictor.load_model():
        print("❌ 模型加载失败")
        return False
    
    # 3. 测试预测
    print("\n3️⃣ 测试预测")
    test_queries = [
        "什么是PEST？",
        "有哪些营销工具可以给我做下推荐？",
        "推荐一些营销类的学习资料",
        "想要通过人群洞察，来指导奈雪2026年的产品策略"
    ]
    
    for query in test_queries:
        result = predictor.predict(query, return_details=True)
        intent = result['intent']
        score = result.get('score', None)
        
        if intent:
            print(f"✅ '{query}' -> {intent} (分数: {score if score is not None else 0:.4f})")
        else:
            print(f"❌ '{query}' -> 未识别")
    
    print("\n✅ JSON格式模型测试完成")
    return True

if __name__ == "__main__":
    success = test_json_model()
    sys.exit(0 if success else 1)