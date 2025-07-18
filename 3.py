
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
create_repo("Xnhyacinth/adathink", repo_type="dataset")

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
from huggingface_hub import HfApi

api = HfApi()

# 上传大文件夹
for name in os.listdir("./"):
    if name.startswith("output") and os.path.isdir(name):
    api.upload_large_folder(
        folder_path=name,
        repo_id="Xnhyacinth/adathink",
        repo_type="dataset",
        # 可选参数
        ignore_patterns=[".git", "*.pyc", "__pycache__", "*.log"],
        delete_patterns=None,  # 删除远程仓库中匹配的文件
        commit_message="Upload large dataset",
        commit_description="Uploading",
        create_pr=False,  # 是否创建 Pull Request
        revision="main",
        allow_patterns=None,  # 只上传匹配的文件
        thread_count=5,  # 并发线程数，可以调整以优化上传速度
    )


# # 4. 上传模型检查点
# create_repo("your-username/llava-qwen-model", repo_type="model")
# upload_folder(
#     folder_path="./checkpoints/qwen_2_5/models/google_siglip2-base-patch16-384-Qwen_Qwen2.5-0.5B-Instruct-pt_558k_qwen_2_5_ft_779k_rope",
#     repo_id="your-username/llava-qwen-model",
#     repo_type="model"
# )