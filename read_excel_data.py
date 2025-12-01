import pandas as pd

# 读取Excel文件
file_path = '/Users/freecisco_yan/Documents/SoinAITech/soinai_projects/semantic-router-main/intent_train.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Excel文件读取成功!")
    print("\n文件内容:")
    print(df)
    print("\n文件信息:")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())
    print("\n数据类型:")
    print(df.dtypes)
except Exception as e:
    print(f"读取Excel文件时出错: {e}")
    # 尝试以不同方式读取
    try:
        # 尝试读取所有sheet
        xls = pd.ExcelFile(file_path)
        print(f"发现的工作表: {xls.sheet_names}")
        # 读取每个工作表
        for sheet_name in xls.sheet_names:
            print(f"\n工作表 '{sheet_name}' 的内容:")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(df)
    except Exception as e2:
        print(f"尝试读取所有工作表时出错: {e2}")
