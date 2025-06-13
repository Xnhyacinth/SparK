import os
import json
import pandas as pd
import ast 

def process_res_json_to_dataframe(directory):
    """
    读取目录下每个文件夹中的 res.json 文件，将其转换为 Pandas DataFrame。

    Args:
        directory (str): 主目录路径。

    Returns:
        pd.DataFrame: 合并的 DataFrame，其中列名为文件夹名，行索引为 res.json 中的 key，值为 res.json 中的 value。
    """
    data = {}  # 用于存储每个文件夹的 {文件夹名: res.json 数据}

    # 遍历主目录下的每个子文件夹
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        # 检查是否是文件夹
        if os.path.isdir(folder_path):
            res_json_path = os.path.join(folder_path, 'res.json')

            # 检查 res.json 是否存在
            if os.path.exists(res_json_path):
                # 读取 res.json 文件
                try:
                    with open(res_json_path, 'r', encoding='utf-8') as f:
                      # with open('input.json', 'r') as file:
                        lines = f.readlines()

                        # 解析每行的 json 对象
                        json_objects = [json.loads(line) for line in lines]

                        # 将 json 对象转换为 json 字符串
                        res_data = json.dumps(json_objects)
                        # res_data = [json.load(line) for line in f.read_lines]  # 加载为字典
                        # res_data = json.load(f)
                        data[folder_name] = res_data  # 使用文件夹名作为列名
                except Exception as e:
                    print(f"Error reading {res_json_path}: {e}")
            else:
                print(f"'res.json' not found in folder: {folder_path}")

    # 转换为 Pandas DataFrame
    # df = pd.DataFrame(data)
    final_data = {}

    # 遍历字典中的每个键值对
    for dataset_name, values_str in data.items():
        # 解析字符串列表为 Python 列表
        values_list = ast.literal_eval(values_str)  # 将字符串形式的列表转换为真正的列表
        temp_dict = {}

        # 提取键值对
        for item in values_list:
            key, value = item.split(": ")  # 按 `: ` 分割键和值
            temp_dict[key.strip()] = float(value)  # 去除空格并将值转换为浮点数

        # 添加到最终数据字典
        final_data[dataset_name] = temp_dict

    # 将字典转换为 DataFrame
    df = pd.DataFrame(final_data)
    df["avg"] = df.mean(axis=1)

    return df

def reorder_columns(df, column_order):
    """
    按指定顺序排序 DataFrame 的列。

    Args:
        df (pd.DataFrame): 输入的 Pandas DataFrame。
        column_order (list): 需要按照的列顺序。

    Returns:
        pd.DataFrame: 重新排序后的 DataFrame。
    """
    # 创建一个新的列顺序列表，将不存在的列剔除
    valid_columns = [col for col in column_order if col in df.columns]
    
    # 按顺序重新排列列
    reordered_df = df[valid_columns + [col for col in df.columns if col not in valid_columns]]

    return reordered_df

column_order = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", 
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", 
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
]

# 按指定顺序排序列


# 示例用法
directory_path = '/modelopsnas/modelops/468440/kvpress/output/results/Meta-Llama-3-8B-Instruct/comress_questions/1024/longbench'  # 主目录路径
result_df = process_res_json_to_dataframe(directory_path)

result_df = reorder_columns(result_df, column_order)

# 打印结果
print(result_df)

# 如果需要保存为 CSV 文件
result_df.to_csv('output.csv', index=True)