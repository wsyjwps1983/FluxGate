"""
主程序
整合训练、预测和模型管理功能
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.model_trainer import IntentRouterTrainer
from app.model_predictor import IntentRouterPredictor

def setup_directories():
    """创建必要的目录结构"""
    # 创建模型保存目录
    models_dir = os.path.join(project_root, "app", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 创建结果输出目录
    results_dir = os.path.join(project_root, "app", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    return models_dir, results_dir

def train_model(args):
    """训练模型命令"""
    print("🚀 开始训练意图识别模型")
    print("=" * 50)
    
    # 设置目录
    models_dir, _ = setup_directories()
    
    # 参数检查
    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        return False
    
    # 生成模型文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"intent_router_{timestamp}.json"
    model_path = os.path.join(models_dir, model_filename)
    
    print(f"📂 训练数据: {args.data_path}")
    print(f"💾 模型将保存到: {model_path}")
    
    # 创建训练器
    trainer = IntentRouterTrainer(
        encoder_name=args.encoder_name,
        score_threshold=args.threshold,
        api_key=args.api_key
    )
    
    # 根据参数选择是否进行阈值优化
    if args.optimize_thresholds:
        # 带阈值优化的训练
        results = trainer.train_with_threshold_optimization(
            data_path=args.data_path,
            save_path=model_path,
            optimization_method=args.optimization_method
        )
        success = results['success']
        if success:
            print("\n🎉 阈值优化训练完成！")
            opt_results = results.get('optimization_results', {})
            print(f"📊 优化前准确率: {opt_results.get('initial_accuracy', 0):.1f}%")
            print(f"📊 优化后准确率: {opt_results.get('optimized_accuracy', 0):.1f}%")
            print(f"📈 准确率提升: {opt_results.get('improvement', 0):+.1f}%")
            
            # 打印优化后的阈值
            optimized_thresholds = opt_results.get('optimized_thresholds')
            if optimized_thresholds:
                print("\n📋 优化后的路由阈值:")
                for route_name, threshold in optimized_thresholds.items():
                    print(f"    - {route_name}: {threshold:.3f}")
    else:
        # 普通训练
        success = trainer.train_and_save(
            data_path=args.data_path,
            save_path=model_path
        )
    
    if success:
        print(f"\n✅ 训练完成！")
        print(f"💾 模型文件: {model_path}")
        print(f"💡 使用以下命令进行预测:")
        print(f"   python -m app.main predict --model {model_path}")
    
    return success

def predict_model(args):
    """预测命令"""
    print("🤖 意图识别预测")
    print("=" * 50)
    
    # 参数检查
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return False
    
    # 创建预测器
    predictor = IntentRouterPredictor(
        model_path=args.model,
        api_key=args.api_key
    )
    
    # 加载模型
    if not predictor.load_model():
        return False
    
    # 根据不同模式执行预测
    if args.interactive:
        # 交互式模式
        predictor.interactive_mode()
    
    elif args.query:
        # 单个查询
        result = predictor.predict(args.query, return_details=True)
        intent = result['intent']
        score = result.get('score', None)
        
        if intent:
            print(f"🎯 识别意图: {intent}")
            if score:
                print(f"📊 匹配分数: {score:.4f}")
        else:
            print("❓ 未识别到意图")
    
    elif args.batch_file:
        # 批量预测
        if not os.path.exists(args.batch_file):
            print(f"❌ 批量查询文件不存在: {args.batch_file}")
            return False
        
        # 读取查询
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"📊 正在处理 {len(queries)} 条查询...")
        
        # 批量预测
        results = predictor.predict_batch(queries, return_details=True)
        
        # 设置结果目录
        _, results_dir = setup_directories()
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"predictions_{timestamp}.csv"
        output_path = os.path.join(results_dir, output_filename)
        
        # 导出结果
        predictor.export_predictions(queries, output_path)
        
        # 打印部分结果
        print("\n预测结果预览:")
        for i, (query, result) in enumerate(zip(queries[:10], results[:10])):
            intent = result['intent']
            print(f"  {i+1}. {query[:50]}{'...' if len(query) > 50 else ''} -> {intent or 'None'}")
        
        if len(queries) > 10:
            print(f"  ... 还有 {len(queries) - 10} 条结果")
        
        print(f"\n📊 完整结果已保存到: {output_path}")
    
    else:
        print("❌ 请提供查询参数: --query, --batch-file 或 --interactive")
        return False
    
    return True

def evaluate_model(args):
    """评估模型命令"""
    print("📊 评估模型性能")
    print("=" * 50)
    
    # 参数检查
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return False
    
    if not os.path.exists(args.test_data):
        print(f"❌ 测试数据文件不存在: {args.test_data}")
        return False
    
    # 创建预测器并加载模型
    predictor = IntentRouterPredictor(
        model_path=args.model,
        api_key=args.api_key
    )
    
    if not predictor.load_model():
        return False
    
    # 加载测试数据
    from app.data_loader import load_intent_data
    test_data_dict = load_intent_data(args.test_data)
    
    # 转换为测试格式
    test_queries = []
    for intent, utterances in test_data_dict.items():
        for utterance in utterances[:3]:  # 每个意图取3条作为测试
            test_queries.append((utterance, intent))
    
    print(f"📊 使用 {len(test_queries)} 条测试数据进行评估")
    
    # 评估
    evaluation = predictor.evaluate_with_test_data(test_queries)
    
    # 设置结果目录
    _, results_dir = setup_directories()
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_filename = f"evaluation_{timestamp}.json"
    eval_path = os.path.join(results_dir, eval_filename)
    
    # 保存评估结果
    import json
    with open(eval_path, 'w', encoding='utf-8') as f:
        # 序列化评估结果，移除不可序列化的对象
        serializable_eval = {
            'total_samples': evaluation['total_samples'],
            'correct_predictions': evaluation['correct_predictions'],
            'accuracy': evaluation['accuracy'],
            'intent_stats': evaluation['intent_stats'],
            'model_info': predictor.model_info,
            'timestamp': timestamp
        }
        json.dump(serializable_eval, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 评估结果已保存到: {eval_path}")
    
    return True

def list_models(args):
    """列出可用模型"""
    print("📂 可用模型列表")
    print("=" * 50)
    
    models_dir, _ = setup_directories()
    
    if not os.path.exists(models_dir):
        print("❌ 模型目录不存在")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.json'))]
    
    if not model_files:
        print("📭 没有找到训练好的模型")
        return False
    
    print(f"📊 找到 {len(model_files)} 个模型:")
    for i, model_file in enumerate(sorted(model_files), 1):
        model_path = os.path.join(models_dir, model_file)
        
        # 获取模型基本信息
        try:
            if model_file.endswith('.json'):
                # 读取JSON格式模型
                import json
                with open(model_path, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
                file_type = "JSON"
            else:
                # 读取Pickle格式模型
                import pickle
                with open(model_path, 'rb') as f:
                    save_data = pickle.load(f)
                file_type = "Pickle"
            
            metadata = save_data.get('metadata', {})
            timestamp = save_data.get('timestamp', 'Unknown')
            version = save_data.get('version', 'Unknown')
            
            print(f"  {i}. {model_file} ({file_type})")
            print(f"     路由数: {metadata.get('num_routes', 'Unknown')}")
            print(f"     语句数: {metadata.get('total_utterances', 'Unknown')}")
            print(f"     版本: {version}")
            print(f"     训练时间: {timestamp}")
            print()
            
        except Exception as e:
            print(f"  {i}. {model_file} (读取信息失败: {e})")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="意图识别系统 - SiliconFlow集成")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data-path', required=True, help='训练数据文件路径 (Excel)')
    train_parser.add_argument('--api-key', help='SiliconFlow API密钥 (或设置SILICONFLOW_API_KEY环境变量)')
    train_parser.add_argument('--encoder-name', default='BAAI/bge-large-zh-v1.5', help='编码器模型名称')
    train_parser.add_argument('--threshold', type=float, default=0.3, help='匹配阈值')
    train_parser.add_argument('--optimize-thresholds', action='store_true', help='启用阈值优化')
    train_parser.add_argument('--optimization-method', choices=['automatic', 'manual'], default='automatic', help='阈值优化方法')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='使用模型进行预测')
    predict_parser.add_argument('--model', required=True, help='模型文件路径')
    predict_parser.add_argument('--api-key', help='SiliconFlow API密钥')
    predict_parser.add_argument('--query', help='单个查询')
    predict_parser.add_argument('--batch-file', help='批量查询文件路径 (每行一个查询)')
    predict_parser.add_argument('--interactive', action='store_true', help='交互式预测模式')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型性能')
    eval_parser.add_argument('--model', required=True, help='模型文件路径')
    eval_parser.add_argument('--test-data', required=True, help='测试数据文件路径 (Excel)')
    eval_parser.add_argument('--api-key', help='SiliconFlow API密钥')
    
    # 列出模型命令
    list_parser = subparsers.add_parser('list', help='列出可用模型')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 设置API密钥环境变量
    if hasattr(args, 'api_key') and args.api_key:
        os.environ["SILICONFLOW_API_KEY"] = args.api_key
    
    # 执行命令
    try:
        if args.command == 'train':
            success = train_model(args)
        elif args.command == 'predict':
            success = predict_model(args)
        elif args.command == 'evaluate':
            success = evaluate_model(args)
        elif args.command == 'list':
            success = list_models(args)
        else:
            print(f"❌ 未知命令: {args.command}")
            success = False
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()