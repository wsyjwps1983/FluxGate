#!/usr/bin/env python3
"""
简单测试脚本，用于调试/predict端点
"""

import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置
API_URL = "http://localhost:8000/predict"

# 测试数据
test_query = "帮我总结附件中的内容，输出该行业的总结"

# 测试函数
def test_predict():
    """测试/predict端点"""
    try:
        print(f"测试URL: {API_URL}")
        print(f"测试查询: {test_query}")
        
        payload = {
            "query": test_query,
            "top_k": 1
        }
        
        print(f"请求数据: {json.dumps(payload, ensure_ascii=False)}")
        
        # 创建会话，禁用连接池
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        session.mount("http://", adapter)
        
        # 添加调试信息
        print("\n请求头:")
        headers = {
            "Content-Type": "application/json"
        }
        for key, value in headers.items():
            print(f"  {key}: {value}")
        
        # 发送请求，明确禁用代理
        response = session.post(
            API_URL, 
            json=payload, 
            headers=headers,
            timeout=30,
            verify=False,  # 禁用SSL验证
            proxies={}
        )
        
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应头:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        print(f"响应内容: {response.text}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\n请求失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=== 测试/predict端点 ===")
    result = test_predict()
    if result:
        print("\n测试成功!")
    else:
        print("\n测试失败!")
