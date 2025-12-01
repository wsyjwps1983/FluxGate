#!/usr/bin/env python3
"""
意图验证脚本
使用final_training_data.json作为初始意图，intent_train.xlsx作为验证数据
逐条读取验证数据调用FastAPI接口进行预测
"""

import json
import http.client
import pandas as pd
import time
import os
from typing import List, Dict, Any

# 配置
API_URL = "http://localhost:8000"
INITIAL_INTENTS_FILE = "/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/final_training_data.json"
VALIDATION_DATA_FILE = "/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/intent_train.xlsx"
OUTPUT_FILE = "/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/validation_results.json"

# 读取初始意图数据
def load_initial_intents(file_path: str) -> List[Dict[str, str]]:
    """读取初始意图数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 读取验证数据
def load_validation_data(file_path: str) -> pd.DataFrame:
    """读取Excel格式的验证数据"""
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"成功读取验证数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"读取验证数据失败: {e}")
        return pd.DataFrame()

# 调用预测接口
def predict_intent(text: str) -> Dict[str, Any]:
    """调用FastAPI接口进行意图预测"""
    try:
        # 解析API URL
        host = "localhost"
        port = 8000
        path = "/predict"
        
        # 准备请求数据
        payload = {
            "query": text,
            "top_k": 1
        }
        json_payload = json.dumps(payload, ensure_ascii=False)
        encoded_payload = json_payload.encode('utf-8')
        
        # 创建连接
        conn = http.client.HTTPConnection(host, port, timeout=30)
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(encoded_payload))
        }
        
        # 发送请求
        conn.request("POST", path, body=encoded_payload, headers=headers)
        
        # 获取响应
        response = conn.getresponse()
        
        # 读取响应内容
        response_data = response.read().decode('utf-8')
        
        # 关闭连接
        conn.close()
        
        if response.status == 200:
            return json.loads(response_data)
        else:
            raise Exception(f"HTTP Error: {response.status} {response.reason}")
    except Exception as e:
        print(f"调用接口失败: {e}")
        return {
            "query": text,
            "intent": None,
            "matched": False,
            "similarity_score": None,
            "top_k_results": [],
            "error": str(e)
        }

# 主函数
def main():
    """主函数"""
    print("=== 意图验证脚本 ===")
    
    # 1. 读取初始意图数据
    print(f"\n1. 读取初始意图数据: {INITIAL_INTENTS_FILE}")
    initial_intents = load_initial_intents(INITIAL_INTENTS_FILE)
    print(f"   共 {len(initial_intents)} 条初始意图")
    
    # 2. 读取验证数据
    print(f"\n2. 读取验证数据: {VALIDATION_DATA_FILE}")
    validation_df = load_validation_data(VALIDATION_DATA_FILE)
    
    if validation_df.empty:
        print("   验证数据为空，退出脚本")
        return
    
    # 3. 准备结果列表
    results = []
    correct_count = 0
    total_count = len(validation_df)
    
    # 4. 逐条调用接口预测
    print(f"\n3. 开始预测，共 {total_count} 条数据")
    for idx, row in validation_df.iterrows():
        # 使用Excel文件中的实际列名
        if "提问内容" not in validation_df.columns or "一级意图分类" not in validation_df.columns:
            print("   Excel文件缺少必要的列(提问内容或一级意图分类)")
            return
        
        question = row["提问内容"]
        true_intent = row["一级意图分类"]
        
        print(f"   预测第 {idx+1}/{total_count}: {question[:50]}...")
        
        # 调用接口预测
        prediction = predict_intent(question)
        
        # 计算是否正确
        predicted_intent = prediction.get("intent")
        is_correct = predicted_intent == true_intent
        if is_correct:
            correct_count += 1
        
        # 保存结果
        result = {
            "index": idx + 1,
            "question": question,
            "true_intent": true_intent,
            "predicted_intent": predicted_intent,
            "is_correct": is_correct,
            "similarity_score": prediction.get("similarity_score"),
            "matched": prediction.get("matched"),
            "top_k_results": prediction.get("top_k_results", []),
            "error": prediction.get("error")
        }
        results.append(result)
        
        # 避免请求过快
        time.sleep(0.1)
    
    # 5. 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n4. 预测完成")
    print(f"   总条数: {total_count}")
    print(f"   正确条数: {correct_count}")
    print(f"   准确率: {accuracy:.2%}")
    
    # 6. 保存结果到文件
    print(f"\n5. 保存结果到: {OUTPUT_FILE}")
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": total_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "results": results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"   结果保存成功")
    
    # 7. 打印错误统计
    error_count = sum(1 for r in results if r.get("error"))
    if error_count > 0:
        print(f"\n6. 错误统计")
        print(f"   错误条数: {error_count}")
    
    print("\n=== 脚本执行完成 ===")

if __name__ == "__main__":
    main()