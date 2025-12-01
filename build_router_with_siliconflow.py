import pandas as pd
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from semantic_router import SemanticRouter, Route
from semantic_router.encoders.siliconflow import SiliconFlowEncoder

# 加载环境变量
load_dotenv()

def load_excel_data(excel_path: str) -> pd.DataFrame:
    """加载Excel数据"""
    print(f"正在从 {excel_path} 加载训练数据...")
    df = pd.read_excel(excel_path)
    print(f"成功加载 {len(df)} 条数据")
    print(f"数据列: {list(df.columns)}")
    
    # 显示数据的前5行
    print("\n数据前5行:")
    print(df.head())
    
    # 统计一级意图分类
    intent_counts = df['一级意图分类'].value_counts()
    print("\n一级意图分类统计:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count} 条")
    
    return df

def build_router(df: pd.DataFrame, encoder: SiliconFlowEncoder) -> SemanticRouter:
    """构建路由"""
    print("\n开始构建路由...")
    
    routes = []
    # 按一级意图分类分组
    grouped = df.groupby('一级意图分类')
    
    for intent, group in grouped:
        # 获取该意图下的所有提问内容作为示例
        examples = group['提问内容'].tolist()
        # 过滤空值
        examples = [ex for ex in examples if pd.notna(ex) and isinstance(ex, str) and ex.strip()]
        
        if examples:
            route = Route(
                name=intent,
                utterances=examples,
                description=f"Intent for {intent}"
            )
            routes.append(route)
            print(f"创建路由: {intent} (包含 {len(examples)} 个示例)")
        else:
            print(f"跳过路由 {intent}: 没有有效的示例")
    
    # 创建路由器
    router = SemanticRouter(routes=routes, encoder=encoder)
    print(f"\n路由构建完成，共创建 {len(router.routes)} 个路由")
    
    return router



def main():
    # 加载环境变量
    load_dotenv()
    
    # 配置参数
    excel_path = "/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/intent_train.xlsx"
    router_output_path = "intent_router.json"
    model_name = "BAAI/bge-large-zh-v1.5"  # 使用适合中文的模型
    score_threshold = 0.7  # 相似度阈值
    
    # 获取API密钥
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("请设置 SILICONFLOW_API_KEY 环境变量")
    
    print(f"使用模型: {model_name}, 阈值: {score_threshold}")
    print("正在使用真实的API密钥进行embedding计算...")
    
    # 创建编码器
    encoder = SiliconFlowEncoder(
        name=model_name,
        siliconflow_api_key=api_key,
        score_threshold=score_threshold
    )
    print(f"SiliconFlow编码器初始化成功")
    
    # 加载数据
    df = load_excel_data(excel_path)
    
    # 构建路由
    router = build_router(df, encoder)
    
    # 保存路由配置
    import json
    router_config = {
        "routes": [
            {
                "name": route.name,
                "utterances": route.utterances,
                "description": route.description
            }
            for route in router.routes
        ],
        "encoder": {
            "type": "siliconflow",
            "name": model_name,
            "score_threshold": score_threshold
        }
    }
    
    with open(router_output_path, 'w', encoding='utf-8') as f:
        json.dump(router_config, f, ensure_ascii=False, indent=2)
    
    print(f"\n路由配置已保存到: {router_output_path}")
    
    print("\n✅ 路由构建完成！")
    print("可以使用以下命令进行验证:")
    print(f"python verify_router.py --router {router_output_path} --test_data /Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/final_training_data.json")


if __name__ == "__main__":
    main()
