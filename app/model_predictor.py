"""
模型预测模块
用于加载训练好的模型进行预测
"""

import os
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from semantic_router.routers import SemanticRouter
from semantic_router.schema import RouteChoice
from semantic_router import Route

class IntentRouterPredictor:
    """意图路由器预测类"""
    
    def __init__(self, model_path: Optional[str] = None, api_key: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            api_key: SiliconFlow API密钥
        """
        self.model_path = model_path
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.router = None
        self.model_info = None
        
        # 如果提供了API密钥，设置环境变量
        if self.api_key:
            os.environ["SILICONFLOW_API_KEY"] = self.api_key
    
    def load_model(self, model_path: str = None) -> bool:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径，如果未提供则使用初始化时的路径
            
        Returns:
            bool: 是否加载成功
        """
        path = model_path or self.model_path
        
        if not path:
            raise ValueError("未提供模型路径")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        try:
            print(f"📂 正在加载模型: {path}")
            
            # 检查文件类型
            if path.endswith('.json'):
                # 加载JSON格式模型
                with open(path, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
                
                # 从JSON数据重建Route对象
                routes = []
                for route_data in save_data['routes']:
                    route = Route(
                        name=route_data['name'],
                        utterances=route_data['utterances']
                    )
                    routes.append(route)
                
                # 重建路由器
                from semantic_router.encoders import SiliconFlowEncoder
                encoder = SiliconFlowEncoder(
                    name=save_data['encoder_name'],
                    score_threshold=save_data['score_threshold']
                )
                
                self.router = SemanticRouter(
                    encoder=encoder,
                    routes=routes,
                    auto_sync="local"
                )
                
                self.model_info = {
                    'encoder_name': save_data['encoder_name'],
                    'score_threshold': save_data['score_threshold'],
                    'routes': routes,
                    'training_data': save_data['training_data'],
                    'timestamp': save_data['timestamp'],
                    'metadata': save_data['metadata']
                }
                
                # 检查模型版本兼容性
                model_version = save_data.get('version', 'unknown')
                if model_version not in ["1.0", "2.0"]:
                    print(f"⚠️  模型版本 {model_version} 可能与当前版本不兼容")
                
                print("✅ JSON模型加载完成")
                
            else:
                # 加载pickle格式模型（兼容旧版本）
                with open(path, 'rb') as f:
                    save_data = pickle.load(f)
                
                self.router = save_data['router']
                self.model_info = {
                    'encoder_name': save_data['encoder_name'],
                    'score_threshold': save_data['score_threshold'],
                    'routes': save_data['routes'],
                    'training_data': save_data['training_data'],
                    'timestamp': save_data['timestamp'],
                    'metadata': save_data['metadata'],
                    'version': save_data.get('version', 'unknown')
                }
                
                # 检查模型版本兼容性
                model_version = save_data.get('version', 'unknown')
                if model_version not in ["1.0", "2.0"]:
                    print(f"⚠️  模型版本 {model_version} 可能与当前版本不兼容")
                
                print("✅ Pickle模型加载完成")
            print(f"📋 模型信息:")
            print(f"    - 路由数量: {self.model_info['metadata']['num_routes']}")
            print(f"    - 总语句数: {self.model_info['metadata']['total_utterances']}")
            print(f"    - 编码器: {self.model_info['encoder_name']}")
            print(f"    - 阈值: {self.model_info['score_threshold']}")
            print(f"    - 训练时间: {self.model_info['timestamp']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def predict(self, query: str, return_details: bool = False) -> Any:
        """
        预测单个查询的意图
        
        Args:
            query: 输入查询
            return_details: 是否返回详细信息
            
        Returns:
            如果return_details为False，返回意图名称或None
            如果return_details为True，返回详细结果字典
        """
        if not self.router:
            raise ValueError("请先加载模型")
        
        try:
            result = self.router(query)
            
            if return_details:
                if result:
                    return {
                        'intent': result.name,
                        'score': getattr(result, 'score', None),
                        'route': result
                    }
                else:
                    return {
                        'intent': None,
                        'score': None,
                        'route': None
                    }
            else:
                return result.name if result else None
                
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            if return_details:
                return {'intent': None, 'score': None, 'error': str(e)}
            else:
                return None
    
    def predict_batch(self, queries: List[str], return_details: bool = False) -> List[Any]:
        """
        批量预测查询的意图
        
        Args:
            queries: 输入查询列表
            return_details: 是否返回详细信息
            
        Returns:
            预测结果列表
        """
        if not self.router:
            raise ValueError("请先加载模型")
        
        results = []
        for query in queries:
            result = self.predict(query, return_details)
            results.append(result)
        
        return results
    
    def get_route_info(self) -> Dict[str, List[str]]:
        """
        获取路由信息
        
        Returns:
            Dict[str, List[str]]: 意图名称到语句列表的映射
        """
        if not self.model_info:
            return {}
        
        return self.model_info['training_data']
    
    def evaluate_with_test_data(self, test_data: List[tuple]) -> Dict[str, Any]:
        """
        使用测试数据评估模型性能
        
        Args:
            test_data: 测试数据列表，格式为[(query, expected_intent), ...]
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        if not self.router:
            raise ValueError("请先加载模型")
        
        print(f"🔍 使用 {len(test_data)} 条测试数据评估模型...")
        
        results = []
        correct = 0
        total = len(test_data)
        
        for query, expected_intent in test_data:
            result = self.predict(query, return_details=True)
            predicted_intent = result['intent']
            
            is_correct = predicted_intent == expected_intent
            if is_correct:
                correct += 1
            
            results.append({
                'query': query,
                'expected': expected_intent,
                'predicted': predicted_intent,
                'correct': is_correct,
                'score': result.get('score', None)
            })
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # 按意图分组统计
        intent_stats = {}
        for item in results:
            intent = item['expected']
            if intent not in intent_stats:
                intent_stats[intent] = {'total': 0, 'correct': 0}
            intent_stats[intent]['total'] += 1
            if item['correct']:
                intent_stats[intent]['correct'] += 1
        
        # 计算每个意图的准确率
        for intent, stats in intent_stats.items():
            stats['accuracy'] = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        
        evaluation = {
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'intent_stats': intent_stats,
            'detailed_results': results
        }
        
        print(f"📊 评估结果:")
        print(f"    - 总样本: {total}")
        print(f"    - 正确预测: {correct}")
        print(f"    - 总体准确率: {accuracy:.1f}%")
        
        print("📋 各意图准确率:")
        for intent, stats in intent_stats.items():
            print(f"    - {intent}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1f}%")
        
        return evaluation
    
    def export_predictions(self, queries: List[str], output_path: str) -> bool:
        """
        导出预测结果到文件
        
        Args:
            queries: 查询列表
            output_path: 输出文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 获取预测结果
            results = self.predict_batch(queries, return_details=True)
            
            # 准备导出数据
            export_data = []
            for i, (query, result) in enumerate(zip(queries, results)):
                export_data.append({
                    'id': i + 1,
                    'query': query,
                    'predicted_intent': result['intent'],
                    'score': result.get('score', ''),
                    'timestamp': datetime.now().isoformat()
                })
            
            # 写入文件
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                if export_data:
                    fieldnames = export_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(export_data)
            
            print(f"📊 预测结果已导出到: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False
    
    def interactive_mode(self):
        """
        启动交互式预测模式
        """
        if not self.router:
            raise ValueError("请先加载模型")
        
        print("\n🤖 交互式意图识别模式")
        print("=" * 50)
        print("输入查询进行意图识别，输入 'quit' 或 'exit' 退出")
        print("输入 'info' 查看模型信息")
        print("输入 'routes' 查看所有可用路由")
        print("=" * 50)
        
        while True:
            try:
                query = input("\n🔍 请输入查询: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    print("👋 退出交互模式")
                    break
                
                if query.lower() == 'info':
                    print(f"\n📋 模型信息:")
                    print(f"    - 编码器: {self.model_info['encoder_name']}")
                    print(f"    - 阈值: {self.model_info['score_threshold']}")
                    print(f"    - 路由数量: {self.model_info['metadata']['num_routes']}")
                    print(f"    - 训练时间: {self.model_info['timestamp']}")
                    continue
                
                if query.lower() == 'routes':
                    routes = self.get_route_info()
                    print(f"\n📂 可用路由 ({len(routes)}):")
                    for i, (intent, utterances) in enumerate(routes.items(), 1):
                        print(f"    {i}. {intent} ({len(utterances)} 条语句)")
                    continue
                
                if not query:
                    continue
                
                # 预测
                result = self.predict(query, return_details=True)
                intent = result['intent']
                score = result.get('score', None)
                
                if intent:
                    print(f"🎯 识别意图: {intent}")
                    if score:
                        print(f"📊 匹配分数: {score:.4f}")
                else:
                    print("❓ 未识别到意图")
                
            except KeyboardInterrupt:
                print("\n👋 退出交互模式")
                break
            except Exception as e:
                print(f"❌ 预测出错: {e}")