#!/usr/bin/env python3
"""
阈值优化测试脚本
测试model_trainer.py中的阈值优化功能
"""

import os
import sys
import pandas as pd
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.model_trainer import IntentRouterTrainer

def create_test_data():
    """创建测试数据"""
    # 创建测试数据目录
    test_data_dir = os.path.join(project_root, "app", "test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    
    # 准备测试数据
    test_data = {
        'intent': [
            'greeting', 'greeting', 'greeting', 'greeting', 'greeting',
            'goodbye', 'goodbye', 'goodbye', 'goodbye', 'goodbye',
            'question', 'question', 'question', 'question', 'question',
            'compliment', 'compliment', 'compliment', 'compliment', 'compliment'
        ],
        'utterance': [
            '你好', '早上好', '晚上好', '您好', '嗨',
            '再见', '拜拜', '下次见', '回头见', '一会见',
            '今天天气怎么样', '明天会下雨吗', '几点了', '今天星期几', '会议几点开始',
            '你真棒', '做得好', '太厉害了', '真聪明', '非常好'
        ]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(test_data)
    
    # 保存为Excel文件
    test_file_path = os.path.join(test_data_dir, "threshold_test_data.xlsx")
    df.to_excel(test_file_path, index=False)
    
    print(f"📊 测试数据已创建: {test_file_path}")
    return test_file_path

def test_basic_training():
    """测试基本训练功能"""
    print("\n" + "="*50)
    print("🧪 测试1: 基本训练功能")
    print("="*50)
    
    # 创建测试数据
    test_file_path = create_test_data()
    
    # 创建模型保存目录
    models_dir = os.path.join(project_root, "app", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 基本训练测试
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    basic_model_path = os.path.join(models_dir, f"basic_test_{timestamp}.json")
    
    trainer = IntentRouterTrainer(score_threshold=0.3)
    success = trainer.train_and_save(
        data_path=test_file_path,
        save_path=basic_model_path
    )
    
    if success:
        print("✅ 基本训练测试通过")
    else:
        print("❌ 基本训练测试失败")
    
    return success, basic_model_path

def test_threshold_optimization():
    """测试阈值优化功能"""
    print("\n" + "="*50)
    print("🧪 测试2: 阈值优化功能")
    print("="*50)
    
    # 使用相同的测试数据
    test_file_path = os.path.join(project_root, "app", "test_data", "threshold_test_data.xlsx")
    
    # 创建模型保存目录
    models_dir = os.path.join(project_root, "app", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 阈值优化测试 - 自动方法
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    auto_opt_model_path = os.path.join(models_dir, f"auto_opt_test_{timestamp}.json")
    
    trainer_auto = IntentRouterTrainer(score_threshold=0.3)
    results_auto = trainer_auto.train_with_threshold_optimization(
        data_path=test_file_path,
        save_path=auto_opt_model_path,
        optimization_method="automatic"
    )
    
    if results_auto['success']:
        opt_results = results_auto.get('optimization_results', {})
        print("✅ 自动阈值优化测试通过")
        print(f"📊 优化前准确率: {opt_results.get('initial_accuracy', 0):.1f}%")
        print(f"📊 优化后准确率: {opt_results.get('optimized_accuracy', 0):.1f}%")
        print(f"📈 准确率提升: {opt_results.get('improvement', 0):+.1f}%")
    else:
        print("❌ 自动阈值优化测试失败")
    
    # 阈值优化测试 - 手动方法
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    manual_opt_model_path = os.path.join(models_dir, f"manual_opt_test_{timestamp}.json")
    
    trainer_manual = IntentRouterTrainer(score_threshold=0.3)
    results_manual = trainer_manual.train_with_threshold_optimization(
        data_path=test_file_path,
        save_path=manual_opt_model_path,
        optimization_method="manual"
    )
    
    if results_manual['success']:
        opt_results = results_manual.get('optimization_results', {})
        print("✅ 手动阈值优化测试通过")
        print(f"📊 优化前准确率: {opt_results.get('initial_accuracy', 0):.1f}%")
        print(f"📊 优化后准确率: {opt_results.get('optimized_accuracy', 0):.1f}%")
        print(f"📈 准确率提升: {opt_results.get('improvement', 0):+.1f}%")
    else:
        print("❌ 手动阈值优化测试失败")
    
    return results_auto['success'], results_manual['success']

def test_manual_threshold_optimization():
    """测试手动阈值优化细节"""
    print("\n" + "="*50)
    print("🧪 测试3: 手动阈值优化细节")
    print("="*50)
    
    # 使用相同的测试数据
    test_file_path = os.path.join(project_root, "app", "test_data", "threshold_test_data.xlsx")
    
    # 创建训练器
    trainer = IntentRouterTrainer(score_threshold=0.3)
    
    # 加载数据
    if not trainer.load_data(test_file_path):
        print("❌ 数据加载失败")
        return False
    
    # 创建路由
    trainer.create_routes()
    
    # 初始化编码器
    trainer.initialize_encoder()
    
    # 预热编码器
    trainer.preheat_encoder()
    
    # 训练路由器
    trainer.train_router()
    
    # 准备测试数据
    _, test_data = trainer.prepare_test_data(test_size=0.5)
    
    # 运行阈值优化
    print("\n🔄 运行手动阈值优化...")
    opt_results = trainer.optimize_thresholds(
        test_data=test_data,
        optimization_method="manual"
    )
    
    # 打印优化结果
    print("\n📊 优化结果详情:")
    print(f"优化方法: {opt_results.get('method', 'unknown')}")
    print(f"优化前准确率: {opt_results.get('initial_accuracy', 0):.1f}%")
    print(f"优化后准确率: {opt_results.get('optimized_accuracy', 0):.1f}%")
    print(f"准确率提升: {opt_results.get('improvement', 0):+.1f}%")
    
    print("\n📋 阈值对比:")
    initial_thresholds = opt_results.get('initial_thresholds', {})
    optimized_thresholds = opt_results.get('optimized_thresholds', {})
    
    for route_name in initial_thresholds:
        init_val = initial_thresholds.get(route_name, 0)
        opt_val = optimized_thresholds.get(route_name, 0)
        print(f"  {route_name}: {init_val:.3f} -> {opt_val:.3f}")
    
    return True

def main():
    """主函数"""
    print("🧪 阈值优化功能测试")
    print("="*50)
    
    # 检查API密钥
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("❌ 未找到SILICONFLOW_API_KEY环境变量，请先设置API密钥")
        return False
    
    try:
        # 运行测试
        basic_success, _ = test_basic_training()
        auto_success, manual_success = test_threshold_optimization()
        detail_success = test_manual_threshold_optimization()
        
        # 汇总结果
        print("\n" + "="*50)
        print("📊 测试结果汇总")
        print("="*50)
        print(f"✅ 基本训练功能: {'通过' if basic_success else '失败'}")
        print(f"✅ 自动阈值优化: {'通过' if auto_success else '失败'}")
        print(f"✅ 手动阈值优化: {'通过' if manual_success else '失败'}")
        print(f"✅ 优化细节测试: {'通过' if detail_success else '失败'}")
        
        # 总体结果
        all_passed = all([basic_success, auto_success, manual_success, detail_success])
        print(f"\n{'🎉 所有测试通过！' if all_passed else '❌ 部分测试失败'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)