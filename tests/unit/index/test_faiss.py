import numpy as np
import pytest
from semantic_router.index import FaissIndex


class TestFaissIndex:
    """测试FaissIndex类的功能"""
    
    def test_init(self):
        """测试初始化"""
        index = FaissIndex()
        assert index is not None
        assert index.type == "faiss"
        assert len(index) == 0
    
    def test_add_and_query(self):
        """测试添加向量和查询"""
        index = FaissIndex()
        
        # 添加测试数据
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        routes = ["route1", "route2", "route3"]
        utterances = ["hello", "world", "test"]
        
        index.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances
        )
        
        # 验证添加成功
        assert len(index) == 3
        
        # 查询测试
        query_vector = np.array([0.1, 0.2, 0.3])
        scores, route_names = index.query(query_vector, top_k=2)
        
        # 验证查询结果
        assert len(scores) == 2
        assert len(route_names) == 2
        assert route_names[0] == "route1"  # 最相似的应该是第一个向量
        assert scores[0] > scores[1]  # 分数应该递减
    
    def test_get_vector_by_utterance(self):
        """测试通过utterance获取向量"""
        index = FaissIndex()
        
        # 添加测试数据
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        routes = ["route1", "route2", "route3"]
        utterances = ["hello", "world", "test"]
        
        index.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances
        )
        
        # 测试获取存在的utterance的向量
        vector = index.get_vector_by_utterance("hello")
        assert vector is not None
        assert isinstance(vector, np.ndarray)
        assert np.allclose(vector, np.array([0.1, 0.2, 0.3]))
        
        # 测试获取不存在的utterance的向量
        vector = index.get_vector_by_utterance("nonexistent")
        assert vector is None
    
    def test_get_utterance_by_vector(self):
        """测试通过向量获取utterance"""
        index = FaissIndex()
        
        # 添加测试数据
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        routes = ["route1", "route2", "route3"]
        utterances = ["hello", "world", "test"]
        
        index.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances
        )
        
        # 测试获取最相似的utterance
        vector = np.array([0.1, 0.2, 0.3])
        results = index.get_utterance_by_vector(vector, top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == "hello"  # 最相似的应该是第一个utterance
        assert results[0][1] > results[1][1]  # 分数应该递减
    
    def test_route_filter(self):
        """测试路由过滤"""
        index = FaissIndex()
        
        # 添加测试数据
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        routes = ["route1", "route2", "route1"]
        utterances = ["hello", "world", "test"]
        
        index.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances
        )
        
        # 使用路由过滤查询
        query_vector = np.array([0.1, 0.2, 0.3])
        scores, route_names = index.query(query_vector, top_k=2, route_filter=["route1"])
        
        # 验证结果只包含route1
        assert len(scores) == 2
        assert len(route_names) == 2
        assert all(route == "route1" for route in route_names)
    
    def test_delete(self):
        """测试删除路由"""
        index = FaissIndex()
        
        # 添加测试数据
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        routes = ["route1", "route2", "route3"]
        utterances = ["hello", "world", "test"]
        
        index.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances
        )
        
        # 删除route2
        index.delete("route2")
        
        # 验证删除成功
        assert len(index) == 2
        
        # 查询验证
        query_vector = np.array([0.4, 0.5, 0.6])
        scores, route_names = index.query(query_vector, top_k=3)
        
        # 结果中不应该包含route2
        assert "route2" not in route_names
    
    def test_dimension_mismatch(self):
        """测试维度不匹配的情况"""
        index = FaissIndex()
        
        # 添加3维向量
        embeddings_3d = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        routes = ["route1", "route2"]
        utterances = ["hello", "world"]
        
        index.add(
            embeddings=embeddings_3d,
            routes=routes,
            utterances=utterances
        )
        
        # 尝试添加4维向量，应该抛出异常
        embeddings_4d = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        routes = ["route3", "route4"]
        utterances = ["test1", "test2"]
        
        with pytest.raises(ValueError):
            index.add(
                embeddings=embeddings_4d,
                routes=routes,
                utterances=utterances
            )
    
    def test_get_utterances(self):
        """测试获取所有utterances"""
        index = FaissIndex()
        
        # 添加测试数据
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        routes = ["route1", "route2"]
        utterances = ["hello", "world"]
        metadata_list = [{"key1": "value1"}, {"key2": "value2"}]
        
        index.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances,
            metadata_list=metadata_list
        )
        
        # 获取utterances
        utterance_objects = index.get_utterances(include_metadata=True)
        
        assert len(utterance_objects) == 2
        assert utterance_objects[0].route == "route1"
        assert utterance_objects[0].utterance == "hello"
        assert utterance_objects[0].metadata == {"key1": "value1"}
        assert utterance_objects[1].route == "route2"
        assert utterance_objects[1].utterance == "world"
        assert utterance_objects[1].metadata == {"key2": "value2"}
    
    def test_describe(self):
        """测试describe方法"""
        index = FaissIndex()
        
        # 添加测试数据
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        routes = ["route1", "route2"]
        utterances = ["hello", "world"]
        
        index.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances
        )
        
        # 测试describe
        config = index.describe()
        assert config.type == "faiss"
        assert config.dimensions == 3
        assert config.vectors == 2
    
    def test_is_ready(self):
        """测试is_ready方法"""
        index = FaissIndex()
        
        # 初始状态应该不ready
        assert index.is_ready() is False
        
        # 添加数据后应该ready
        embeddings = [[0.1, 0.2, 0.3]]
        routes = ["route1"]
        utterances = ["hello"]
        
        index.add(
            embeddings=embeddings,
            routes=routes,
            utterances=utterances
        )
        
        assert index.is_ready() is True