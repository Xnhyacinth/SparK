
import json

# 读取 jsonl 文件
with open('input.json', 'r') as file:
    lines = file.readlines()

# 解析每行的 json 对象
json_objects = [json.loads(line) for line in lines]

# 将 json 对象转换为 json 字符串
json_data = json.dumps(json_objects)

# 写入 json 文件
with open('output.json', 'w') as file:
    file.write(json_data)