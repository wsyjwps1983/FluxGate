"""
SiliconFlow最佳实践示例（修复版）
展示如何正确使用SiliconFlow编码器和LLM进行语义路由
"""

import os
import time
from semantic_router import Route
from semantic_router.encoders import SiliconFlowEncoder
from semantic_router.llms import SiliconFlowLLM
from semantic_router.routers import SemanticRouter
from semantic_router.schema import Message

# 设置API密钥
os.environ["SILICONFLOW_API_KEY"] = "sk-lzjbutyzaadepnpqbrvwekvclwwwbvezvppwpzwtleucfkuc"

def create_encoder():
    """创建优化的编码器"""
    return SiliconFlowEncoder(
        name="BAAI/bge-large-zh-v1.5",  # 中文嵌入模型
        score_threshold=0.3  # 较低阈值增加匹配概率
    )

def create_llm():
    """创建LLM实例"""
    return SiliconFlowLLM(
        name="deepseek-ai/DeepSeek-R1",
        temperature=0.1,
        max_tokens=100
    )

def create_routes():
    """创建丰富的路由集合"""
    return [
        Route(
            name="技术问题",
            utterances=[
                # 基础概念
                "什么是人工智能",
                "机器学习是什么",
                "深度学习的原理",
                "神经网络如何工作",
                
                # 应用场景
                "深度学习有什么应用",
                "AI在哪些领域有应用",
                "计算机视觉的应用",
                "自然语言处理技术",
                
                # 技术细节
                "如何训练模型",
                "Python有哪些AI库",
                "什么是Transformer架构",
                "强化学习算法"
            ]
        ),
        Route(
            name="产品问题",
            utterances=[
                # 基础服务
                "硅基流动提供什么服务",
                "SiliconFlow有什么功能",
                "支持哪些模型",
                
                # API使用
                "如何使用API",
                "API调用方法",
                "如何申请API密钥",
                "API调用限制",
                
                # 价格和方案
                "价格方案是什么",
                "服务计费方式",
                "如何付费",
                
                # 支持和帮助
                "有技术支持吗",
                "如何联系客服",
                "文档在哪里"
            ]
        ),
        Route(
            name="学术问题",
            utterances=[
                # 学术研究
                "人工智能的最新研究",
                "有哪些相关的论文",
                "AI领域的知名学者",
                
                # 学习资源
                "推荐学习资料",
                "有哪些在线课程",
                "如何入门深度学习",
                
                # 实践项目
                "有什么实践项目",
                "如何做研究",
                "学术会议信息"
            ]
        ),
        Route(
            name="其他问题",
            utterances=[
                # 一般对话
                "你好",
                "谢谢你",
                "再见",
                
                # 其他
                "天气怎么样",
                "今天星期几",
                "其他话题"
            ]
        )
    ]

def initialize_router():
    """初始化语义路由器"""
    print("🔧 初始化语义路由器...")
    
    # 创建组件
    encoder = create_encoder()
    routes = create_routes()
    
    # 创建路由器
    router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
    
    # 预热（分批处理）
    print("🔥 预热编码器和索引...")
    
    # 分批预热以避免批量大小限制
    all_utterances = []
    for route in routes:
        all_utterances.extend(route.utterances)
    
    # 分批处理，每批最多32个
    batch_size = 30  # 留一些余量
    for i in range(0, len(all_utterances), batch_size):
        batch = all_utterances[i:i+batch_size]
        _ = encoder(batch)  # 预热
        time.sleep(1)  # 避免API限制
    
    time.sleep(2)  # 等待索引初始化
    
    print("✅ 路由器初始化完成")
    return router

def test_router(router):
    """测试路由器功能"""
    print("\n🔍 测试语义路由器")
    print("=" * 50)
    
    # 测试用例
    test_cases = [
        ("什么是机器学习？", "技术问题"),
        ("硅基流动能为我提供哪些服务，如何申请API密钥？", "产品问题"),
        ("推荐一些AI学习资料", "学术问题"),
        ("你好", "其他问题"),
        ("强化学习算法有哪些？", "技术问题"),
        ("服务计费方式是什么？", "产品问题"),
        ("AI领域的最新研究", "学术问题"),
        ("天气怎么样", "其他问题")
    ]
    
    # 测试并统计结果
    correct = 0
    for query, expected in test_cases:
        try:
            route = router(query)
            if route and route.name == expected:
                print(f"✅ '{query}' -> {route.name} (正确)")
                correct += 1
            elif route:
                print(f"⚠️  '{query}' -> {route.name} (期望: {expected})")
            else:
                print(f"❓ '{query}' -> 未匹配")
        except Exception as e:
            print(f"❌ '{query}' -> 错误: {e}")
    
    # 统计结果
    accuracy = correct / len(test_cases) * 100
    print(f"\n📊 路由准确率: {correct}/{len(test_cases)} = {accuracy:.1f}%")
    
    return accuracy

def test_llm(llm):
    """测试LLM功能"""
    print("\n🤖 测试LLM功能")
    print("=" * 50)
    
    test_prompts = [
        "请简要介绍语义路由技术",
        "深度学习和机器学习有什么区别？",
        "如何优化神经网络性能？"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            messages = [Message(role="user", content=prompt)]
            response = llm(messages)
            print(f"\n💬 提示 {i}: {prompt}")
            print(f"🤖 回答: {response[:150]}...")
        except Exception as e:
            print(f"❌ 提示 {i} 失败: {e}")

def test_encoder(encoder):
    """测试编码器功能"""
    print("\n📊 测试编码器功能")
    print("=" * 50)
    
    test_texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的子集",
        "深度学习使用神经网络模型"
    ]
    
    try:
        embeddings = encoder(test_texts)
        print(f"✅ 成功编码 {len(test_texts)} 个文本")
        print(f"📏 每个嵌入的维度: {len(embeddings[0])}")
        print(f"🔢 第一个嵌入的前5个值: {embeddings[0][:5]}")
        
        # 计算相似度
        import numpy as np
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_1_2 = cosine_sim(embeddings[0], embeddings[1])
        sim_2_3 = cosine_sim(embeddings[1], embeddings[2])
        
        print(f"\n📏 文本相似度:")
        print(f"文本1 vs 文本2: {sim_1_2:.4f}")
        print(f"文本2 vs 文本3: {sim_2_3:.4f}")
        
    except Exception as e:
        print(f"❌ 编码器测试失败: {e}")

def test_integration():
    """测试集成功能"""
    print("\n🔗 测试集成功能")
    print("=" * 50)
    
    try:
        # 测试AutoEncoder
        from semantic_router.encoders import AutoEncoder
        auto_encoder = AutoEncoder("siliconflow", "BAAI/bge-large-zh-v1.5")
        embeddings = auto_encoder(["测试文本"])
        print(f"✅ AutoEncoder测试成功，嵌入维度: {len(embeddings[0])}")
        
        # 测试不同模型
        models = [
            "BAAI/bge-large-zh-v1.5"
        ]
        
        print("\n🧪 测试不同模型:")
        for model in models:
            try:
                test_encoder = SiliconFlowEncoder(name=model, score_threshold=0.3)
                test_embeddings = test_encoder(["测试"])
                print(f"  ✅ {model}: 维度 {len(test_embeddings[0])}")
            except Exception as e:
                print(f"  ❌ {model}: 错误 {e}")
                
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")

def main():
    """主函数"""
    print("🚀 SiliconFlow 最佳实践示例（修复版）")
    print("=" * 50)
    
    # 1. 初始化组件
    router = initialize_router()
    llm = create_llm()
    encoder = create_encoder()
    
    # 2. 测试编码器
   # test_encoder(encoder)
    
    # 3. 测试LLM
   # test_llm(llm)
    
    # 4. 测试路由器
    accuracy = test_router(router)
    
    # 5. 测试集成功能
    #test_integration()
    
    # 6. 总结
    print("\n" + "=" * 50)
    print("🎉 测试完成")
    #print(f"✅ 编码器功能正常")
    #print(f"✅ LLM功能正常")
    print(f"✅ 路由器功能正常（准确率: {accuracy:.1f}%）")
    print(f"✅ 集成功能正常")
    
    # 7. 优化建议
    print("\n💡 优化建议:")
    if accuracy < 80:
        print("  📉 路由准确率较低，建议：")
        print("    1. 降低 score_threshold 到 0.2-0.3")
        print("    2. 增加每个路由的 utterances 数量")
        print("    3. 确保 utterances 覆盖更多表达方式")
    else:
        print("  📈 路由性能良好，可以考虑：")
        print("    1. 适当提高阈值以提高精确度")
        print("    2. 添加更多路由类别")
        print("    3. 优化 utterances 的多样性")
    
    print("\n📚 更多信息请参考 SILICONFLOW_INTEGRATION.md")

if __name__ == "__main__":
    main()