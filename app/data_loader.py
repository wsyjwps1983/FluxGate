"""
数据加载模块
用于从Excel文件中加载训练数据
"""

import pandas as pd
from typing import Dict, List, Tuple
import os

def load_intent_data(file_path: str) -> Dict[str, List[str]]:
    """
    从Excel文件中加载意图训练数据
    
    Args:
        file_path: Excel文件路径
        
    Returns:
        Dict[str, List[str]]: 意图名称到语句列表的映射
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"📊 成功加载数据文件，共 {len(df)} 条记录")
        print(f"📋 列名: {list(df.columns)}")
        print(f"🔍 数据预览:")
        print(df.head())
        
        # 假设Excel文件有两列: 'intent' 和 'utterance'
        # 根据实际列名调整
        intent_col = 'intent'
        utterance_col = 'utterance'
        
        # 如果列名不同，尝试自动识别
        if intent_col not in df.columns or utterance_col not in df.columns:
            print("⚠️  标准列名未找到，尝试自动识别...")
            for col in df.columns:
                col_lower = col.lower()
                # 支持英文和中文列名识别
                if ('intent' in col_lower or '意图' in col) and intent_col == 'intent':  # 检查原始值
                    intent_col = col
                elif (('utterance' in col_lower or 'text' in col_lower) or '内容' in col or '提问' in col) and utterance_col == 'utterance':  # 检查原始值
                    utterance_col = col
        
        if intent_col not in df.columns or utterance_col not in df.columns:
            raise ValueError(f"无法找到意图和语句列，可用列: {list(df.columns)}")
        
        print(f"📌 使用列: 意图='{intent_col}', 语句='{utterance_col}'")
        
        # 按意图分组
        intent_data = {}
        for intent in df[intent_col].unique():
            intent_utterances = df[df[intent_col] == intent][utterance_col].tolist()
            intent_data[str(intent)] = intent_utterances
            print(f"  📂 {intent}: {len(intent_utterances)} 条语句")
        
        return intent_data
        
    except Exception as e:
        raise Exception(f"加载数据文件失败: {str(e)}")

def validate_data(intent_data: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    """
    验证训练数据的质量
    
    Args:
        intent_data: 意图数据字典
        
    Returns:
        Tuple[bool, List[str]]: (是否有效, 问题列表)
    """
    issues = []
    
    if not intent_data:
        issues.append("数据为空")
        return False, issues
    
    # 检查每个意图的语句数量
    for intent, utterances in intent_data.items():
        if len(utterances) < 3:
            issues.append(f"意图 '{intent}' 的语句数量过少 ({len(utterances)} < 3)")
        
        if len(utterances) > 1000:
            issues.append(f"意图 '{intent}' 的语句数量过多 ({len(utterances)} > 1000)")
        
        # 检查空语句
        empty_count = sum(1 for u in utterances if not str(u).strip())
        if empty_count > 0:
            issues.append(f"意图 '{intent}' 包含 {empty_count} 条空语句")
    
    # 检查重复语句
    all_utterances = []
    for utterances in intent_data.values():
        all_utterances.extend(utterances)
    
    unique_utterances = set(all_utterances)
    duplicate_count = len(all_utterances) - len(unique_utterances)
    if duplicate_count > 0:
        issues.append(f"发现 {duplicate_count} 条重复语句")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def clean_data(intent_data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    清理训练数据
    
    Args:
        intent_data: 原始意图数据
        
    Returns:
        Dict[str, List[str]]: 清理后的意图数据
    """
    cleaned_data = {}
    
    for intent, utterances in intent_data.items():
        # 清理每个语句
        cleaned_utterances = []
        for utterance in utterances:
            if str(utterance).strip():  # 非空
                # 基本清理：去除首尾空格
                cleaned = str(utterance).strip()
                if cleaned:
                    cleaned_utterances.append(cleaned)
        
        # 去重
        cleaned_utterances = list(set(cleaned_utterances))
        
        if cleaned_utterances:  # 只保留非空的意图
            cleaned_data[intent] = cleaned_utterances
    
    return cleaned_data