import json

# 读取JSON文件
file_path = '/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/final_training_data.json'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("JSON文件读取成功!")
    print(f"\n数据条目数量: {len(data)}")
    
    if data:
        # 显示第一条数据的结构
        print("\n数据结构示例:")
        print(json.dumps(data[0], ensure_ascii=False, indent=2))
        
        # 统计不同的intent类型
        intents = {}  # 使用一级意图分类
        for item in data:
            intent = item.get('intent', 'unknown')
            intents[intent] = intents.get(intent, 0) + 1
        
        print("\n意图类型统计:")
        for intent, count in intents.items():
            print(f"{intent}: {count}条")
        
        # 显示前5条数据作为示例
        print("\n前5条数据示例:")
        for i, item in enumerate(data[:5]):
            print(f"\n条目{i+1}:")
            print(f"问题: {item.get('question', 'N/A')}")
            print(f"意图: {item.get('intent', 'N/A')}")
            
        # 检查是否有其他字段
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        print("\n所有字段名:", list(all_keys))
        
    else:
        print("JSON文件为空")
        
except Exception as e:
    print(f"读取JSON文件时出错: {e}")
