#!/usr/bin/env python3
"""
测试脚本
验证修复后的代码功能
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app import IntentRouterTrainer, IntentRouterPredictor, load_intent_data, validate_data, clean_data

def test_data_loading():
    """测试数据加载"""
    print("🧪 测试数据加载")
    print("-" * 40)
    
    try:
        # 加载原始数据
        data_path = os.path.join(project_root, "intent_train.xlsx")
        raw_data = load_intent_data(data_path)
        
        # 验证数据
        is_valid, issues = validate_data(raw_data)
        print(f"数据验证结果: {'通过' if is_valid else '失败'}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")
        
        # 清理数据
        cleaned_data = clean_data(raw_data)
        print(f"清理后的意图数量: {len(cleaned_data)}")
        
        print("✅ 数据加载测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def test_trainer_initialization():
    """测试训练器初始化"""
    print("\n🧪 测试训练器初始化")
    print("-" * 40)
    
    try:
        # 创建训练器
        trainer = IntentRouterTrainer(
            encoder_name="BAAI/bge-large-zh-v1.5",
            score_threshold=0.3,
            api_key="test_key"  # 测试用，不会实际调用API
        )
        
        print("✅ 训练器初始化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 训练器初始化测试失败: {e}")
        return False

def test_predictor_initialization():
    """测试预测器初始化"""
    print("\n🧪 测试预测器初始化")
    print("-" * 40)
    
    try:
        # 创建预测器
        predictor = IntentRouterPredictor(
            model_path="test_model.pkl",
            api_key="test_key"  # 测试用
        )
        
        print("✅ 预测器初始化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 预测器初始化测试失败: {e}")
        return False

def test_command_line_interface():
    """测试命令行接口"""
    print("\n🧪 测试命令行接口")
    print("-" * 40)
    
    try:
        # 测试帮助信息
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "app.main", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            print("✅ 命令行接口测试通过")
            return True
        else:
            print(f"❌ 命令行接口测试失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 命令行接口测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 代码修复验证测试")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_trainer_initialization,
        test_predictor_initialization,
        test_command_line_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！代码修复成功。")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)