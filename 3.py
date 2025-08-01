
# import json

# # 读取 jsonl 文件
# with open('input.json', 'r') as file:
#     lines = file.readlines()

# # 解析每行的 json 对象
# json_objects = [json.loads(line) for line in lines]

# # 将 json 对象转换为 json 字符串
# json_data = json.dumps(json_objects)

# # 写入 json 文件
# with open('output.json', 'w') as file:
#     file.write(json_data)

# from datasets import load_dataset


# for task in [
#     "narrativeqa",
#     "qasper",
#     "multifieldqa_en",
#     "multifieldqa_zh",
#     "hotpotqa",
#     "2wikimqa",
#     "musique",
#     "dureader",
#     "gov_report",
#     "qmsum",
#     "multi_news",
#     "vcsum",
#     "trec",
#     "triviaqa",
#     "samsum",
#     "lsht",
#     "passage_count",
#     "passage_retrieval_en",
#     "passage_retrieval_zh",
#     "lcc",
#     "repobench-p"
# ]:
#     dataset = load_dataset("Xnhyacinth/LongBench", task, split="test")


from huggingface_hub import upload_file, upload_folder, create_repo
import os
# 1. 创建数据集仓库
# create_repo("Xnhyacinth/adathink", repo_type="dataset")

# 2. 上传训练数据
# upload_file(
#     path_or_fileobj="/mnt/bn/bytenn-lq2/lhx/vlm_datasets/llava_next_779k.json",
#     path_in_repo="llava_next_779k.json",
#     repo_id="xnhyacinth/adathink",
#     repo_type="dataset"
# )

# 3. 上传图像（如果不太大）
# for name in os.listdir("./"):
#     if name.startswith("output") and os.path.isdir(name):
#         print(f"Uploading folder: {name}")
#         upload_folder(
#             folder_path=name,
#             repo_id="Xnhyacinth/adathink",
#             repo_type="dataset",
#             path_in_repo=name+"/"
#         )
# from huggingface_hub import HfApi

# api = HfApi()

# # 上传大文件夹
# for name in os.listdir("./"):
#     if name.startswith("output") and os.path.isdir(name):
#         print(f"Uploading folder: {name}")
#         # api.upload_large_folder(
#         #     folder_path=name,
#         #     repo_id="Xnhyacinth/adathink",
#         #     repo_type="dataset",
#         #     path_in_repo=name,  # 保留原目录结构
#         #     # 可选参数
#         #     ignore_patterns=[".git", "*.pyc", "__pycache__", "*.log"],
#         #     # delete_patterns=None,  # 删除远程仓库中匹配的文件
#         #     # commit_message="Upload large dataset",
#         #     # commit_description="Uploading",
#         #     # create_pr=False,  # 是否创建 Pull Request
#         #     revision="main",
#         #     allow_patterns=None,  # 只上传匹配的文件
#         #     # thread_count=5,  # 并发线程数，可以调整以优化上传速度
#         # )
#         api.upload_folder(
#             folder_path=name,
#             repo_id="Xnhyacinth/adathink",
#             repo_type="dataset",
#             path_in_repo=name,  # 这个参数在upload_folder中支持
#             ignore_patterns=[".git", "*.pyc", "__pycache__", "*.log"],
#             commit_message="Upload output folder",
#         )

# ===== 删除远程仓库文件和文件夹的方法 =====

# 方法1: 删除单个文件
# api.delete_file(
#     path_in_repo="要删除的文件路径.txt",
#     repo_id="Xnhyacinth/adathink",
#     repo_type="dataset",
#     commit_message="Delete specific file"
# )

# 方法2: 删除整个文件夹
# api.delete_folder(
#     path_in_repo="要删除的文件夹名",
#     repo_id="Xnhyacinth/adathink", 
#     repo_type="dataset",
#     commit_message="Delete folder"
# )

# 方法3: 批量删除多个文件/文件夹
# files_to_delete = ["file1.txt", "folder1", "folder2/subfolder"]
# for path in files_to_delete:
#     try:
#         if "." in path.split("/")[-1]:  # 判断是文件
#             api.delete_file(
#                 path_in_repo=path,
#                 repo_id="Xnhyacinth/adathink",
#                 repo_type="dataset",
#                 commit_message=f"Delete file {path}"
#             )
#         else:  # 判断是文件夹
#             api.delete_folder(
#                 path_in_repo=path,
#                 repo_id="Xnhyacinth/adathink",
#                 repo_type="dataset", 
#                 commit_message=f"Delete folder {path}"
#             )
#         print(f"Successfully deleted: {path}")
#     except Exception as e:
#         print(f"Failed to delete {path}: {e}")

# 方法4: 删除所有 output 开头的文件夹
# import requests

# # 首先获取仓库中的所有文件列表
# try:
#     repo_files = api.list_repo_files(
#         repo_id="Xnhyacinth/adathink",
#         repo_type="dataset"
#     )
    
#     # 找出所有 output 开头的文件夹
#     output_folders = set()
#     for file_path in repo_files:
#         if file_path.startswith("llama31_layer"):
#             folder_name = file_path.split("/")[0]
#             output_folders.add(folder_name)
    
#     # 删除这些文件夹
#     for folder in output_folders:
#         try:
#             api.delete_file(
#                 path_in_repo=folder,
#                 repo_id="Xnhyacinth/adathink",
#                 repo_type="dataset",
#                 commit_message=f"Delete output folder {folder}"
#             )
#             print(f"Successfully deleted folder: {folder}")
#         except Exception as e:
#             print(f"Failed to delete folder {folder}: {e}")

# except Exception as e:
#     print(f"Failed to list repo files: {e}")

# 方法5: 使用 upload_large_folder 的 delete_patterns 参数
# api.upload_large_folder(
#     folder_path="./empty_folder",  # 上传一个空文件夹
#     repo_id="Xnhyacinth/adathink", 
#     repo_type="dataset",
#     delete_patterns=["llama31_layer*"],  # 删除所有 output 开头的文件/文件夹
#     commit_message="Clean up output folders"
# )


# # 4. 上传模型检查点
# create_repo("your-username/llava-qwen-model", repo_type="model")
# upload_folder(
#     folder_path="./checkpoints/qwen_2_5/models/google_siglip2-base-patch16-384-Qwen_Qwen2.5-0.5B-Instruct-pt_558k_qwen_2_5_ft_779k_rope",
#     repo_id="your-username/llava-qwen-model",
#     repo_type="model"
# )

from datasets import load_dataset
import json
# Load the dataset
dataset = load_dataset('2wiki')

# Display the first few rows of the dataset
print(dataset['train'][0])

# Save the test dataset as a JSON Lines file
# breakpoint()
test_dataset = dataset['validation']

def parse_context(example):
    example['context'] = json.loads(example['context'])
    return example
test_dataset = test_dataset.map(parse_context)
# breakpoint()
test_dataset = test_dataset
test_dataset.rename_column_('context', 'label')
test_dataset.to_json('dev0.jsonl', orient='records', lines=True)
# [' '.join(test_dataset[0]['context'][0][1]) for rank in range(3)]

import json

def get_jsonl(f):
    # import json
    return [json.loads(x) for x in open(f).readlines()]

data = get_jsonl('dev0.jsonl')
for i,d in enumerate(data):
    data[i]['context'] = json.loads(d['context'])

with open('dev.jsonl', 'w') as f:
    for d in data:
        f.write(json.dumps(d)+'\n')

#     #   'Title: ' + data[0]['context'][0][0] + '. Context: ' + ' '.join(data[0]['context'][0][1])
# breakpoint()