#!/usr/bin/env python3
"""
使用http.client测试/predict端点，避免代理问题
"""

import http.client
import json

# 配置
HOST = "localhost"
PORT = 8000
PATH = "/predict"

# 测试数据
test_query = "帮我总结附件中的内容，输出该行业的总结"

# 测试函数
def test_predict():
    """使用http.client测试/predict端点"""
    try:
        print(f"测试地址: {HOST}:{PORT}{PATH}")
        print(f"测试查询: {test_query}")
        
        # 创建连接
        conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
        
        # 准备请求数据
        payload = {
            "query": test_query,
            "top_k": 1
        }
        json_payload = json.dumps(payload, ensure_ascii=False)
        
        print(f"请求数据: {json_payload}")
        
        # 编码为UTF-8
        encoded_payload = json_payload.encode('utf-8')
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(encoded_payload))
        }
        
        print(f"请求头: {headers}")
        
        # 发送请求
        conn.request("POST", PATH, body=encoded_payload, headers=headers)
        
        # 获取响应
        response = conn.getresponse()
        
        print(f"\n响应状态码: {response.status}")
        print(f"响应头: {response.getheaders()}")
        
        # 读取响应内容
        response_data = response.read().decode('utf-8')
        print(f"响应内容: {response_data}")
        
        # 关闭连接
        conn.close()
        
        if response.status == 200:
            return json.loads(response_data)
        else:
            return None
    except Exception as e:
        print(f"\n请求失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=== 使用http.client测试/predict端点 ===")
    result = test_predict()
    if result:
        print("\n测试成功!")
    else:
        print("\n测试失败!")
