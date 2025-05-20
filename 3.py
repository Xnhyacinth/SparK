
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

from datasets import load_dataset


for task in [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p"
]:
    dataset = load_dataset("Xnhyacinth/LongBench", task, split="test")