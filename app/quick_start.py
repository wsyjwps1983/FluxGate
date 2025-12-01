#!/usr/bin/env python3
"""
快速开始脚本
演示如何使用意图识别系统
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app import IntentRouterTrainer, IntentRouterPredictor

def main():
    """快速开始演示"""
    print("🚀 SiliconFlow 意图识别系统 - 快速开始")
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
    
    # 设置路径
    data_path = os.path.join(project_root, "intent_train.xlsx")
    
    # 检查训练数据文件
    if not os.path.exists(data_path):
        print(f"❌ 训练数据文件不存在: {data_path}")
        print("请确保 intent_train.xlsx 文件在项目根目录下")
        return
    
    # 创建目录
    models_dir = os.path.join(project_root, "app", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"📂 训练数据: {data_path}")
    print(f"💾 模型目录: {models_dir}")
    
    try:
        # 1. 训练模型
        print("\n1️⃣ 训练意图识别模型")
        print("-" * 40)
        
        trainer = IntentRouterTrainer(api_key=api_key)
        
        # 训练并保存模型
        model_path = os.path.join(models_dir, "quick_start_model.pkl")
        
        if trainer.train_and_save(data_path, model_path):
            print("✅ 模型训练成功！")
        else:
            print("❌ 模型训练失败")
            return
        
        # 2. 加载模型并测试
        print("\n2️⃣ 测试模型功能")
        print("-" * 40)
        
        predictor = IntentRouterPredictor(model_path=model_path)
        predictor.load_model()
        
        # 测试查询
        test_queries = [
            "什么是人工智能？",
            "如何申请API密钥？",
            "推荐一些学习资料",
            "你好"
        ]
        
        print("🔍 测试查询结果:")
        for query in test_queries:
            result = predictor.predict(query, return_details=True)
            intent = result['intent']
            score = result.get('score', None)
            
            if intent:
                print(f"  '{query}' -> {intent} (分数: {score:.4f})")
            else:
                print(f"  '{query}' -> 未识别")
        
        # 3. 交互式模式
        print("\n3️⃣ 进入交互式预测模式")
        print("-" * 40)
        print("输入查询进行意图识别，输入 'quit' 或 'exit' 退出")
        
        while True:
            try:
                query = input("\n🔍 请输入查询: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    print("👋 退出快速开始演示")
                    break
                
                if not query:
                    continue
                
                # 预测
                result = predictor.predict(query, return_details=True)
                intent = result['intent']
                score = result.get('score', None)
                
                if intent:
                    print(f"🎯 识别意图: {intent}")
                    if score:
                        print(f"📊 匹配分数: {score:.4f}")
                else:
                    print("❓ 未识别到意图")
                
            except KeyboardInterrupt:
                print("\n👋 退出快速开始演示")
                break
            except Exception as e:
                print(f"❌ 预测出错: {e}")
        
        print("\n🎉 快速开始演示完成！")
        print("\n💡 更多用法:")
        print("  • 训练模型: python -m app.main train --data-path intent_train.xlsx")
        print("  • 预测查询: python -m app.main predict --model app/models/quick_start_model.pkl --query '你的查询'")
        print("  • 交互模式: python -m app.main predict --model app/models/quick_start_model.pkl --interactive")
        print("  • 查看帮助: python -m app.main --help")
        
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()