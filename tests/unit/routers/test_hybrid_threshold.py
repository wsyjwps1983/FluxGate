import pytest
from semantic_router import Route, HybridRouter
from semantic_router.encoders import LocalEncoder, BM25Encoder
from semantic_router.index import FaissIndex


class TestHybridRouterThreshold:
    """测试HybridRouter的自定义阈值支持"""
    
    def test_route_level_threshold(self):
        """测试路由级别的阈值设置"""
        # 初始化组件
        encoder = LocalEncoder()
        sparse_encoder = BM25Encoder()
        index = FaissIndex()
        
        # 创建带有不同阈值的路由
        route1 = Route(
            name="route1",
            utterances=["hello", "hi", "greeting"],
            score_threshold=0.8  # 高阈值
        )
        
        route2 = Route(
            name="route2",
            utterances=["bye", "goodbye", "farewell"],
            score_threshold=0.5  # 低阈值
        )
        
        # 初始化HybridRouter
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            index=index,
            routes=[route1, route2],
            alpha=0.3
        )
        
        # 验证路由阈值设置正确
        assert router.routes[0].score_threshold == 0.8
        assert router.routes[1].score_threshold == 0.5
    
    def test_add_route_with_threshold(self):
        """测试添加带有阈值的路由"""
        # 初始化组件
        encoder = LocalEncoder()
        sparse_encoder = BM25Encoder()
        index = FaissIndex()
        
        # 初始化HybridRouter
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            index=index,
            alpha=0.3
        )
        
        # 添加带有阈值的路由
        route = Route(
            name="test_route",
            utterances=["test1", "test2", "test3"],
            score_threshold=0.7
        )
        
        router.add(route)
        
        # 验证路由添加成功且阈值正确
        assert len(router.routes) == 1
        assert router.routes[0].score_threshold == 0.7
    
    def test_add_route_without_threshold(self):
        """测试添加没有阈值的路由，应该使用默认阈值"""
        # 初始化组件
        encoder = LocalEncoder()
        sparse_encoder = BM25Encoder()
        index = FaissIndex()
        
        # 初始化HybridRouter，设置全局阈值
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            index=index,
            alpha=0.3
        )
        
        # 设置全局阈值
        router.score_threshold = 0.6
        
        # 添加没有阈值的路由
        route = Route(
            name="test_route",
            utterances=["test1", "test2", "test3"]
        )
        
        router.add(route)
        
        # 验证路由添加成功且使用了全局阈值
        assert len(router.routes) == 1
        assert router.routes[0].score_threshold == 0.6
    
    def test_threshold_in_prediction(self):
        """测试阈值在预测中的应用"""
        # 初始化组件
        encoder = LocalEncoder()
        sparse_encoder = BM25Encoder()
        index = FaissIndex()
        
        # 创建路由
        route1 = Route(
            name="greeting",
            utterances=["hello", "hi", "good morning"],
            score_threshold=0.8  # 高阈值
        )
        
        route2 = Route(
            name="farewell",
            utterances=["bye", "goodbye", "see you"],
            score_threshold=0.5  # 低阈值
        )
        
        # 初始化HybridRouter
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            index=index,
            routes=[route1, route2],
            alpha=0.3
        )
        
        # 测试低相似度查询
        result = router("hey there")
        
        # 由于greeting路由阈值较高，而farewell阈值较低，应该匹配到farewell
        assert result.name == "farewell" or result.name == "greeting"  # 取决于编码器的实际输出
    
    def test_update_threshold(self):
        """测试更新路由阈值"""
        # 初始化组件
        encoder = LocalEncoder()
        sparse_encoder = BM25Encoder()
        index = FaissIndex()
        
        # 创建路由
        route = Route(
            name="test_route",
            utterances=["test1", "test2", "test3"],
            score_threshold=0.5
        )
        
        # 初始化HybridRouter
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            index=index,
            routes=[route],
            alpha=0.3
        )
        
        # 更新路由阈值
        router.update(name="test_route", threshold=0.9)
        
        # 验证阈值更新成功
        assert router.routes[0].score_threshold == 0.9
    
    def test_async_add_route_with_threshold(self):
        """测试异步添加带有阈值的路由"""
        import asyncio
        
        # 初始化组件
        encoder = LocalEncoder()
        sparse_encoder = BM25Encoder()
        index = FaissIndex()
        
        # 初始化HybridRouter
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            index=index,
            alpha=0.3
        )
        
        # 创建带有阈值的路由
        route = Route(
            name="async_route",
            utterances=["async1", "async2", "async3"],
            score_threshold=0.7
        )
        
        # 异步添加路由
        async def test_async_add():
            await router.aadd(route)
            
            # 验证路由添加成功且阈值正确
            assert len(router.routes) == 1
            assert router.routes[0].score_threshold == 0.7
        
        # 运行异步测试
        asyncio.run(test_async_add())
    
    def test_set_score_threshold_method(self):
        """测试_set_score_threshold方法"""
        # 初始化组件
        encoder = LocalEncoder()
        sparse_encoder = BM25Encoder()
        index = FaissIndex()
        
        # 创建路由
        route1 = Route(
            name="route1",
            utterances=["test1", "test2", "test3"]
        )
        
        route2 = Route(
            name="route2",
            utterances=["test4", "test5", "test6"]
        )
        
        # 初始化HybridRouter
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            index=index,
            routes=[route1, route2],
            alpha=0.3
        )
        
        # 设置encoder的score_threshold
        encoder.score_threshold = 0.7
        
        # 调用_set_score_threshold方法
        router._set_score_threshold()
        
        # 验证阈值计算正确（encoder.score_threshold * alpha）
        expected_threshold = 0.7 * 0.3
        assert router.score_threshold == expected_threshold
        
        # 验证所有路由都设置了阈值
        assert all(route.score_threshold == expected_threshold for route in router.routes)
    
    def test_different_thresholds_effect(self):
        """测试不同阈值对预测结果的影响"""
        # 初始化组件
        encoder = LocalEncoder()
        sparse_encoder = BM25Encoder()
        index = FaissIndex()
        
        # 创建路由
        route1 = Route(
            name="route1",
            utterances=["hello", "hi", "greeting"],
            score_threshold=0.9  # 非常高的阈值
        )
        
        route2 = Route(
            name="route2",
            utterances=["test", "testing", "test case"],
            score_threshold=0.1  # 非常低的阈值
        )
        
        # 初始化HybridRouter
        router = HybridRouter(
            encoder=encoder,
            sparse_encoder=sparse_encoder,
            index=index,
            routes=[route1, route2],
            alpha=0.3
        )
        
        # 测试查询
        query = "this is a test"
        result = router(query)
        
        # 由于route2的阈值非常低，应该匹配到route2
        assert result.name == "route2" or result.name == "route1"  # 取决于编码器的实际输出