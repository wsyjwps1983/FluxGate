"""
模型训练模块
用于训练和保存语义路由模型
"""

import os
import time
import pickle
import json
from typing import List, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np

from semantic_router import Route
from semantic_router.encoders import SiliconFlowEncoder
from semantic_router.routers import SemanticRouter

from .data_loader import load_intent_data, validate_data, clean_data
# 设置API密钥
os.environ["SILICONFLOW_API_KEY"] = "sk-lzjbutyzaadepnpqbrvwekvclwwwbvezvppwpzwtleucfkuc"
class IntentRouterTrainer:
    """意图路由器训练类"""
    
    def __init__(self, 
                 encoder_name: str = "BAAI/bge-large-zh-v1.5",
                 score_threshold: float = 0.3,
                 api_key: Optional[str] = None):
        """
        初始化训练器
        
        Args:
            encoder_name: 编码器模型名称
            score_threshold: 匹配阈值
            api_key: SiliconFlow API密钥
        """
        self.encoder_name = encoder_name
        self.score_threshold = score_threshold
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        
        if not self.api_key:
            raise ValueError("未提供API密钥，请设置SILICONFLOW_API_KEY环境变量或传入api_key参数")
        
        # 设置环境变量
        os.environ["SILICONFLOW_API_KEY"] = self.api_key
        
        self.encoder = None
        self.router = None
        self.training_data = None
        self.routes = None
        
    def load_data(self, file_path: str) -> bool:
        """
        加载训练数据
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            print(f"📂 正在加载训练数据: {file_path}")
            
            # 加载原始数据
            raw_data = load_intent_data(file_path)
            print(f"📊 加载了 {len(raw_data)} 个意图类别的原始数据")
            
            # 验证数据
            is_valid, issues = validate_data(raw_data)
            if not is_valid:
                print("⚠️  数据质量检查发现问题:")
                for issue in issues:
                    print(f"    - {issue}")
                print("🧹 将清理数据...")
            
            # 清理数据
            self.training_data = clean_data(raw_data)
            
            print(f"✅ 数据加载完成，清理后共 {len(self.training_data)} 个意图类别:")
            for intent, utterances in self.training_data.items():
                print(f"    📂 {intent}: {len(utterances)} 条语句")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def create_routes(self) -> List[Route]:
        """
        从训练数据创建路由
        
        Returns:
            List[Route]: 路由列表
        """
        if not self.training_data:
            raise ValueError("请先加载训练数据")
        
        routes = []
        for intent_name, utterances in self.training_data.items():
            route = Route(
                name=str(intent_name),
                utterances=utterances
            )
            routes.append(route)
        
        self.routes = routes
        print(f"🔧 创建了 {len(routes)} 个路由")
        return routes
    
    def initialize_encoder(self):
        """初始化编码器"""
        try:
            print(f"🔧 初始化编码器: {self.encoder_name}")
            self.encoder = SiliconFlowEncoder(
                name=self.encoder_name,
                score_threshold=self.score_threshold
            )
            print("✅ 编码器初始化完成")
        except ImportError as e:
            print(f"❌ 编码器模块导入失败: {e}")
            raise
        except ConnectionError as e:
            print(f"❌ 无法连接到API: {e}")
            raise
        except Exception as e:
            print(f"❌ 编码器初始化失败: {e}")
            raise
    
    def preheat_encoder(self):
        """预热编码器以处理批量数据"""
        if not self.encoder or not self.routes:
            raise ValueError("请先初始化编码器和创建路由")
        
        print("🔥 预热编码器...")
        
        # 收集所有语句
        all_utterances = []
        for route in self.routes:
            all_utterances.extend(route.utterances)
        
        print(f"📊 预热 {len(all_utterances)} 条语句")
        
        # 分批处理，避免API限制
        batch_size = 30  # SiliconFlow API限制
        total_batches = (len(all_utterances) + batch_size - 1) // batch_size
        
        for i in range(0, len(all_utterances), batch_size):
            batch = all_utterances[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    print(f"🔄 处理批次 {batch_num}/{total_batches} ({len(batch)} 条)" + 
                          (f" (重试 {retry_count+1}/{max_retries})" if retry_count > 0 else ""))
                    _ = self.encoder(batch)  # 预热
                    success = True
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"❌ 批次 {batch_num} 处理失败，已达最大重试次数: {e}")
                        raise
                    print(f"⚠️  批次 {batch_num} 处理失败，将重试: {e}")
                    time.sleep(2 ** retry_count)  # 指数退避
                
            # 避免API限制
            if batch_num < total_batches:
                time.sleep(1)
        
        print("✅ 编码器预热完成")
    
    def train_router(self, auto_sync: str = "local") -> SemanticRouter:
        """
        训练语义路由器
        
        Args:
            auto_sync: 同步模式
            
        Returns:
            SemanticRouter: 训练好的路由器
        """
        if not self.encoder or not self.routes:
            raise ValueError("请先初始化编码器并创建路由")
        
        try:
            print("🚀 开始训练语义路由器...")
            
            # 创建路由器
            self.router = SemanticRouter(
                encoder=self.encoder,
                routes=self.routes,
                auto_sync=auto_sync
            )
            
            print("✅ 语义路由器训练完成")
            return self.router
            
        except Exception as e:
            print(f"❌ 路由器训练失败: {e}")
            raise
    
    def evaluate_router(self, test_queries: List[tuple] = None, return_details: bool = False):
        """
        评估路由器性能
        
        Args:
            test_queries: 测试查询列表，格式为[(query, expected_intent), ...]
            return_details: 是否返回详细评估结果
            
        Returns:
            float: 准确率 (如果return_details=True，返回(准确率, 详细结果))
        """
        if not self.router:
            raise ValueError("请先训练路由器")
        
        if not test_queries:
            # 如果没有提供测试数据，从训练数据中抽样
            test_queries = []
            for intent_name, utterances in self.training_data.items():
                # 每个意图取1-2条作为测试
                for i in range(min(2, len(utterances))):
                    test_queries.append((utterances[i], intent_name))
        
        print(f"🔍 使用 {len(test_queries)} 条测试数据评估路由器...")
        
        correct = 0
        detailed_results = []
        
        for query, expected_intent in test_queries:
            try:
                route = self.router(query)
                is_correct = route and route.name == expected_intent
                if is_correct:
                    correct += 1
                
                # 收集详细结果用于分析
                detailed_results.append({
                    'query': query,
                    'expected': expected_intent,
                    'predicted': route.name if route else None,
                    'score': route.score if route else None,
                    'correct': is_correct
                })
                
                # 可以打印详细结果用于调试
                # else:
                #     print(f"❓ '{query}' -> {route.name if route else 'None'} (期望: {expected_intent})")
            except Exception as e:
                print(f"❌ 查询失败: '{query}' -> {e}")
                detailed_results.append({
                    'query': query,
                    'expected': expected_intent,
                    'predicted': None,
                    'score': None,
                    'correct': False,
                    'error': str(e)
                })
        
        accuracy = correct / len(test_queries) * 100 if test_queries else 0
        print(f"📊 路由器准确率: {correct}/{len(test_queries)} = {accuracy:.1f}%")
        
        if return_details:
            return accuracy, detailed_results
        else:
            return accuracy
    
    def prepare_test_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        从训练数据中准备测试数据集
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            Tuple: (训练数据, 测试数据)，格式为[(utterance, intent), ...]
        """
        if not self.training_data:
            raise ValueError("请先加载训练数据")
        
        # 将训练数据转换为(utterance, intent)格式的列表
        all_data = []
        for intent_name, utterances in self.training_data.items():
            for utterance in utterances:
                all_data.append((utterance, intent_name))
        
        # 分割训练和测试数据
        train_data, test_data = train_test_split(
            all_data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=[intent for _, intent in all_data]  # 按意图分层抽样
        )
        
        print(f"📊 准备数据集完成:")
        print(f"    - 训练数据: {len(train_data)} 条")
        print(f"    - 测试数据: {len(test_data)} 条")
        
        return train_data, test_data
    
    def optimize_thresholds(self, 
                          test_data: List[tuple] = None, 
                          max_iterations: int = 100,
                          optimization_method: str = "automatic"):
        """
        优化路由阈值
        
        Args:
            test_data: 测试数据，格式为[(utterance, intent), ...]
            max_iterations: 最大迭代次数
            optimization_method: 优化方法，可选"automatic"或"manual"
            
        Returns:
            dict: 优化结果，包含新的阈值和性能指标
        """
        if not self.router:
            raise ValueError("请先训练路由器")
        
        # 如果没有提供测试数据，准备默认测试数据
        if not test_data:
            _, test_data = self.prepare_test_data()
        
        print("🔧 开始阈值优化...")
        print(f"📊 使用 {len(test_data)} 条测试数据进行优化")
        print(f"⚙️  优化方法: {optimization_method}")
        
        # 获取当前阈值
        current_thresholds = self.router.get_thresholds()
        print(f"📋 当前路由阈值:")
        for route_name, threshold in current_thresholds.items():
            print(f"    - {route_name}: {threshold:.3f}")
        
        # 准备训练数据用于fit方法
        X_test, y_test = zip(*test_data)
        
        # 获取优化前的性能
        initial_accuracy = self.evaluate_router(test_data)
        # 确保accuracy是float类型
        if isinstance(initial_accuracy, tuple):
            initial_accuracy = initial_accuracy[0]
        print(f"📊 优化前准确率: {initial_accuracy:.1f}%")
        
        optimization_results = {
            'initial_thresholds': current_thresholds.copy(),
            'initial_accuracy': float(initial_accuracy),
            'optimized_thresholds': None,
            'optimized_accuracy': None,
            'improvement': 0,
            'method': optimization_method
        }
        
        if optimization_method == "automatic":
            # 使用文档中提到的fit方法进行自动优化
            print("\n🔄 使用自动优化方法...")
            try:
                self.router.fit(X=list(X_test), y=list(y_test))
                optimized_accuracy = self.evaluate_router(test_data)
                # 确保accuracy是float类型
                if isinstance(optimized_accuracy, tuple):
                    optimized_accuracy = optimized_accuracy[0]
                optimized_thresholds = self.router.get_thresholds()
                
                print(f"✅ 自动优化完成!")
                improvement = float(optimized_accuracy) - float(initial_accuracy)
                
                optimization_results['optimized_thresholds'] = optimized_thresholds
                optimization_results['optimized_accuracy'] = optimized_accuracy
                optimization_results['improvement'] = improvement
                
                print(f"📊 优化后准确率: {optimized_accuracy:.1f}%")
                print(f"📈 准确率提升: {improvement:+.1f}%")
                print(f"📋 优化后阈值:")
                for route_name, threshold in optimized_thresholds.items():
                    print(f"    - {route_name}: {threshold:.3f}")
                
            except Exception as e:
                print(f"❌ 自动优化失败: {e}")
                print("🔄 尝试手动优化...")
                optimization_method = "manual"
        
        if optimization_method == "manual":
            # 手动优化方法：调整每个路由的阈值
            print("\n🔄 使用手动优化方法...")
            
            # 评估每个路由的当前阈值范围
            route_scores = self._analyze_route_scores(test_data)
            
            # 优化每个路由的阈值
            optimized_thresholds = {}
            
            for route_name, scores in route_scores.items():
                current_threshold = current_thresholds.get(route_name, self.score_threshold)
                
                # 找到最佳阈值
                best_threshold, best_accuracy = self._find_best_threshold(
                    route_name, scores, test_data, current_threshold
                )
                
                optimized_thresholds[route_name] = best_threshold
                
                print(f"📊 {route_name}:")
                print(f"    - 当前阈值: {current_threshold:.3f}")
                print(f"    - 优化阈值: {best_threshold:.3f}")
                print(f"    - 准确率变化: {best_accuracy:.1f}%")
            
            # 应用优化后的阈值 - 由于直接设置阈值方法可能不存在，我们采用重建路由器的方法
            optimized_routes = []
            for route in self.routes:
                # 获取优化后的阈值
                new_threshold = optimized_thresholds.get(route.name, route.score_threshold)
                
                # 创建带有新阈值的路由
                optimized_route = Route(
                    name=route.name,
                    utterances=route.utterances
                )
                optimized_route.score_threshold = new_threshold
                optimized_routes.append(optimized_route)
            
            # 重建路由器
            self.router = SemanticRouter(
                encoder=self.encoder,
                routes=optimized_routes,
                auto_sync="local"
            )
            
            # 评估优化后的性能
            optimized_accuracy = self.evaluate_router(test_data)
            # 确保accuracy是float类型
            if isinstance(optimized_accuracy, tuple):
                optimized_accuracy = optimized_accuracy[0]
            improvement = float(optimized_accuracy) - float(initial_accuracy)
            
            optimization_results['optimized_thresholds'] = optimized_thresholds
            optimization_results['optimized_accuracy'] = optimized_accuracy
            optimization_results['improvement'] = improvement
            
            print(f"\n📊 优化后准确率: {optimized_accuracy:.1f}%")
            print(f"📈 准确率提升: {improvement:+.1f}%")
        
        return optimization_results
    
    def _analyze_route_scores(self, test_data: List[tuple]) -> dict:
        """
        分析每个路由的分数分布
        
        Args:
            test_data: 测试数据
            
        Returns:
            dict: 每个路由的分数列表
        """
        route_scores = {}
        
        for query, expected_intent in test_data:
            try:
                route = self.router(query)
                score = route.score if route else 0
                
                # 按预期意图分组分数
                if expected_intent not in route_scores:
                    route_scores[expected_intent] = []
                route_scores[expected_intent].append(score)
                
            except Exception as e:
                print(f"⚠️  无法分析查询: '{query}' -> {e}")
        
        return route_scores
    
    def _find_best_threshold(self, 
                           route_name: str, 
                           scores: List[float],
                           test_data: List[tuple],
                           current_threshold: float):
        """
        为特定路由找到最佳阈值
        
        Args:
            route_name: 路由名称
            scores: 该路由的分数列表
            test_data: 测试数据
            current_threshold: 当前阈值
            
        Returns:
            Tuple: (最佳阈值, 对应准确率)
        """
        if not scores:
            return current_threshold, 0
        
        # 确定搜索范围
        min_score, max_score = min(scores), max(scores)
        search_range = np.linspace(min_score * 0.8, max_score * 1.2, 20)
        
        best_threshold = current_threshold
        best_accuracy = 0
        
        # 临时存储原始阈值
        original_thresholds = self.router.get_thresholds()
        
        # 搜索最佳阈值
        for threshold in search_range:
            # 设置临时阈值
            temp_thresholds = original_thresholds.copy()
            temp_thresholds[route_name] = threshold
            
            try:
                # 应用临时阈值 - 由于set_thresholds方法可能不存在，我们使用fit方法
                # 先准备训练数据
                X_fit = [item[0] for item in test_data if item[1] == route_name]
                y_fit = [item[1] for item in test_data if item[1] == route_name]
                
                # 只有当有足够的数据时才进行拟合
                if len(X_fit) > 0:
                    # 创建临时路由
                    temp_route = Route(name=route_name, utterances=X_fit)
                    # 临时修改阈值
                    temp_route.score_threshold = threshold
                    
                    # 临时替换路由器中的路由
                    original_routes = self.router.routes.copy()
                    updated_routes = []
                    for r in original_routes:
                        if r.name == route_name:
                            updated_routes.append(temp_route)
                        else:
                            updated_routes.append(r)
                    
                    # 创建临时路由器
                    temp_router = SemanticRouter(
                        encoder=self.encoder,
                        routes=updated_routes,
                        auto_sync="local"
                    )
                    
                    # 评估性能
                    accuracy = self._evaluate_router_with_temp(temp_router, test_data)
                else:
                    accuracy = 0
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                    
            except Exception as e:
                print(f"⚠️  阈值 {threshold:.3f} 评估失败: {e}")
        
        return best_threshold, best_accuracy
    
    def _evaluate_router_with_temp(self, temp_router, test_data: List[tuple]) -> float:
        """
        使用临时路由器评估性能
        
        Args:
            temp_router: 临时路由器
            test_data: 测试数据
            
        Returns:
            float: 准确率
        """
        correct = 0
        for query, expected_intent in test_data:
            try:
                route = temp_router(query)
                if route and route.name == expected_intent:
                    correct += 1
            except Exception as e:
                print(f"⚠️  临时评估查询失败: '{query}' -> {e}")
        
        accuracy = correct / len(test_data) * 100 if test_data else 0
        return float(accuracy)
    
    def train_with_threshold_optimization(self, 
                                       data_path: str, 
                                       save_path: str, 
                                       test_data: List[tuple] = None,
                                       optimization_method: str = "automatic"):
        """
        带阈值优化的完整训练流程
        
        Args:
            data_path: 训练数据路径
            save_path: 模型保存路径
            test_data: 测试数据
            optimization_method: 优化方法
            
        Returns:
            dict: 训练和优化结果
        """
        try:
            # 1. 加载数据
            if not self.load_data(data_path):
                return {'success': False, 'error': '数据加载失败'}
            
            # 2. 准备测试数据
            if not test_data:
                _, test_data = self.prepare_test_data()
            
            # 3. 创建路由
            self.create_routes()
            
            # 4. 初始化编码器
            self.initialize_encoder()
            
            # 5. 预热编码器
            self.preheat_encoder()
            
            # 6. 训练路由器
            self.train_router()
            
            # 7. 优化阈值
            optimization_results = self.optimize_thresholds(test_data, optimization_method=optimization_method)
            
            # 8. 保存模型
            success = self.save_model(save_path)
            
            if success:
                print(f"\n🎉 训练、优化和保存完成！")
                print(f"📊 最终准确率: {optimization_results.get('optimized_accuracy', 0):.1f}%")
                print(f"📈 准确率提升: {optimization_results.get('improvement', 0):+.1f}%")
                print(f"💾 模型保存路径: {save_path}")
            
            return {
                'success': success,
                'optimization_results': optimization_results,
                'model_path': save_path if success else None
            }
            
        except Exception as e:
            print(f"❌ 训练优化流程失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def save_model(self, save_path: str) -> bool:
        """
        保存训练好的模型
        
        Args:
            save_path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        if not self.router:
            raise ValueError("请先训练路由器")
        
        # 确保路径以.json结尾
        if not save_path.endswith('.json'):
            save_path = save_path.replace('.pkl', '.json')
            if not save_path.endswith('.json'):
                save_path += '.json'
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 准备可序列化的路由数据
            routes_data = []
            for route in self.routes:
                routes_data.append({
                    'name': route.name,
                    'utterances': route.utterances
                })
            
            # 准备保存的数据
            save_data = {
                'encoder_name': self.encoder_name,
                'score_threshold': self.score_threshold,
                'routes': routes_data,
                'training_data': self.training_data,
                'timestamp': datetime.now().isoformat(),
                'version': "2.0",  # 新版本使用JSON格式
                'compatibility': "semantic-router-v1",
                'metadata': {
                    'num_routes': len(self.routes) if self.routes else 0,
                    'total_utterances': sum(len(utterances) for utterances in self.training_data.values()) if self.training_data else 0
                }
            }
            
            # 保存到JSON文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            # 同时保存一个pickle文件用于快速加载（可选）
            pickle_path = save_path.replace('.json', '.pkl')
            try:
                with open(pickle_path, 'wb') as f:
                    # 只保存可pickle的部分，不包含router对象
                    pickle_data = {
                        'encoder_name': self.encoder_name,
                        'score_threshold': self.score_threshold,
                        'routes': self.routes,
                        'training_data': self.training_data,
                        'timestamp': save_data['timestamp'],
                        'version': "2.0",
                        'compatibility': "semantic-router-v1",
                        'metadata': save_data['metadata']
                    }
                    pickle.dump(pickle_data, f)
                print(f"📄 附加保存pickle文件: {pickle_path}")
            except Exception as e:
                print(f"⚠️  无法保存pickle文件: {e}")
            
            print(f"💾 模型已保存到: {save_path}")
            print(f"📋 模型信息:")
            print(f"    - 路由数量: {len(self.routes) if self.routes else 0}")
            print(f"    - 总语句数: {sum(len(utterances) for utterances in self.training_data.values()) if self.training_data else 0}")
            print(f"    - 编码器: {save_data['encoder_name']}")
            print(f"    - 阈值: {save_data['score_threshold']}")
            print(f"    - 保存时间: {save_data['timestamp']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型保存失败: {e}")
            return False
    
    def train_and_save(self, data_path: str, save_path: str, test_queries: List[tuple] = None) -> bool:
        """
        完整的训练和保存流程
        
        Args:
            data_path: 训练数据路径
            save_path: 模型保存路径
            test_queries: 测试查询
            
        Returns:
            bool: 是否成功
        """
        try:
            # 1. 加载数据
            if not self.load_data(data_path):
                return False
            
            # 2. 创建路由
            self.create_routes()
            
            # 3. 初始化编码器
            self.initialize_encoder()
            
            # 4. 预热编码器
            self.preheat_encoder()
            
            # 5. 训练路由器
            self.train_router()
            
            # 6. 评估性能
            accuracy = self.evaluate_router(test_queries)
            
            # 7. 保存模型
            success = self.save_model(save_path)
            
            if success:
                print(f"\n🎉 训练和保存完成！")
                print(f"📊 最终准确率: {accuracy:.1f}%")
                print(f"💾 模型保存路径: {save_path}")
            
            return success
            
        except Exception as e:
            print(f"❌ 训练流程失败: {e}")
            return False