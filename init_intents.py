#!/usr/bin/env python3
"""
意图初始化脚本
使用final_training_data.json中的数据初始化意图路由
"""

import json
import requests
import time
from typing import List, Dict, Any

# 配置
API_URL = "http://localhost:8000"
INITIAL_INTENTS_FILE = "/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/final_training_data.json"

# 读取初始意图数据
def load_initial_intents(file_path: str) -> List[Dict[str, str]]:
    """读取初始意图数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 按意图分组
def group_intents_by_name(intents: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """按意图名称分组，收集每个意图的所有问题"""
    grouped = {}
    for intent in intents:
        name = intent["intent"]
        question = intent["question"]
        if name not in grouped:
            grouped[name] = []
        grouped[name].append(question)
    return grouped

# 创建路由
def create_route(intent_name: str, utterances: List[str]) -> Dict[str, Any]:
    """创建路由数据"""
    return {
        "name": intent_name,
        "utterances": utterances,
        "score_threshold": 0.7,  # 默认阈值
        "description": f"{intent_name}意图路由",
        "metadata": {}
    }

# 调用API添加路由
def add_route(route_data: Dict[str, Any]) -> bool:
    """调用API添加路由"""
    try:
        url = f"{API_URL}/routes"
        print(f"  调用API: {url}")
        print(f"  路由数据: {route_data['name']}")
        response = requests.post(url, json=route_data, timeout=30)
        print(f"  响应状态码: {response.status_code}")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"添加路由失败 {route_data['name']}: {e}")
        return False

# 主函数
def main():
    """主函数"""
    print("=== 意图初始化脚本 ===")
    
    # 1. 读取初始意图数据
    print(f"\n1. 读取初始意图数据: {INITIAL_INTENTS_FILE}")
    initial_intents = load_initial_intents(INITIAL_INTENTS_FILE)
    print(f"   共 {len(initial_intents)} 条初始意图数据")
    
    # 2. 按意图分组
    print("\n2. 按意图分组")
    grouped_intents = group_intents_by_name(initial_intents)
    print(f"   共 {len(grouped_intents)} 个意图类型")
    for intent_name, utterances in grouped_intents.items():
        print(f"   - {intent_name}: {len(utterances)} 条样本")
    
    # 3. 创建路由数据
    print("\n3. 创建路由数据")
    routes = []
    for intent_name, utterances in grouped_intents.items():
        route = create_route(intent_name, utterances)
        routes.append(route)
    print(f"   共创建 {len(routes)} 个路由")
    
    # 4. 调用API添加路由
    print("\n4. 调用API添加路由")
    success_count = 0
    for route in routes:
        if add_route(route):
            success_count += 1
        time.sleep(0.1)  # 避免请求过快
    
    print(f"\n5. 初始化完成")
    print(f"   成功添加 {success_count}/{len(routes)} 个路由")
    
    # 6. 验证路由是否添加成功
    print("\n6. 验证路由添加结果")
    try:
        response = requests.get(f"{API_URL}/routes", timeout=30)
        response.raise_for_status()
        result = response.json()
        print(f"   当前路由总数: {result['count']}")
        print(f"   路由列表: {[route['name'] for route in result['routes']]}")
    except requests.exceptions.RequestException as e:
        print(f"   验证失败: {e}")

if __name__ == "__main__":
    main()