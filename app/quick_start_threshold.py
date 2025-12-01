#!/usr/bin/env python3
"""
阈值优化快速启动脚本
展示如何使用model_trainer.py中的阈值优化功能
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.model_trainer import IntentRouterTrainer

def quick_start_example():
    """快速启动示例"""
    print("🚀 语义路由器阈值优化快速启动示例")
    print("="*50)
    
    # 检查API密钥
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("❌ 未找到SILICONFLOW_API_KEY环境变量")
        print("请设置API密钥: export SILICONFLOW_API_KEY=your_api_key")
        return False
    
    # 检查训练数据
    data_path = os.path.join(project_root, "intent_train.xlsx")
    if not os.path.exists(data_path):
        print(f"❌ 未找到训练数据文件: {data_path}")
        print("请确保训练数据文件存在于项目根目录")
        return False
    
    # 创建模型保存目录
    models_dir = os.path.join(project_root, "app", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 示例1: 基本训练
    print("\n📝 示例1: 基本训练（不进行阈值优化）")
    print("-" * 30)
    trainer = IntentRouterTrainer(
        encoder_name="BAAI/bge-large-zh-v1.5",
        score_threshold=0.3
    )
    
    success = trainer.train_and_save(
        data_path=data_path,
        save_path=os.path.join(models_dir, "basic_model.json")
    )
    
    if success:
        print("✅ 基本训练完成")
    else:
        print("❌ 基本训练失败")
        return False
    
    # 示例2: 带自动阈值优化的训练
    print("\n📝 示例2: 自动阈值优化训练")
    print("-" * 30)
    auto_trainer = IntentRouterTrainer(
        encoder_name="BAAI/bge-large-zh-v1.5",
        score_threshold=0.3
    )
    
    auto_results = auto_trainer.train_with_threshold_optimization(
        data_path=data_path,
        save_path=os.path.join(models_dir, "auto_optimized_model.json"),
        optimization_method="automatic"
    )
    
    if auto_results['success']:
        print("✅ 自动阈值优化训练完成")
        opt_results = auto_results.get('optimization_results', {})
        print(f"📊 准确率提升: {opt_results.get('improvement', 0):+.1f}%")
    else:
        print("❌ 自动阈值优化训练失败")
        return False
    
    # 示例3: 带手动阈值优化的训练
    print("\n📝 示例3: 手动阈值优化训练")
    print("-" * 30)
    manual_trainer = IntentRouterTrainer(
        encoder_name="BAAI/bge-large-zh-v1.5",
        score_threshold=0.3
    )
    
    manual_results = manual_trainer.train_with_threshold_optimization(
        data_path=data_path,
        save_path=os.path.join(models_dir, "manual_optimized_model.json"),
        optimization_method="manual"
    )
    
    if manual_results['success']:
        print("✅ 手动阈值优化训练完成")
        opt_results = manual_results.get('optimization_results', {})
        print(f"📊 准确率提升: {opt_results.get('improvement', 0):+.1f}%")
    else:
        print("❌ 手动阈值优化训练失败")
        return False
    
    # 总结
    print("\n🎉 所有示例执行完成！")
    print("\n📂 模型文件位置:")
    print(f"  基本模型: {os.path.join(models_dir, 'basic_model.json')}")
    print(f"  自动优化模型: {os.path.join(models_dir, 'auto_optimized_model.json')}")
    print(f"  手动优化模型: {os.path.join(models_dir, 'manual_optimized_model.json')}")
    
    print("\n💡 使用提示:")
    print("  1. 使用以下命令进行预测:")
    print(f"     python -m app.main predict --model {os.path.join(models_dir, 'auto_optimized_model.json')}")
    print("  2. 使用以下命令评估模型:")
    print(f"     python -m app.main evaluate --model {os.path.join(models_dir, 'auto_optimized_model.json')} --test-data {data_path}")
    
    return True

def main():
    """主函数"""
    try:
        success = quick_start_example()
        if success:
            print("\n✅ 快速启动示例执行成功！")
        else:
            print("\n❌ 快速启动示例执行失败")
        return success
    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)