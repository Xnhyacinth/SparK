import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('output_norm1/results/Meta-Llama-3-8B-Instruct/comress_questions/0.0/128/results.csv')  # 替换为你的 CSV 文件路径
df['Average'] = df.iloc[:, 1:].mean(axis=1)
# 打印整个内容
print(df)
# print(df.iloc[:, :13])

# 如果想逐行打印：
for index, row in df.iterrows():
    print(f"Row {index}: {row.to_dict()}")
# import torch
# import torch.nn as nn

# # 输入，形状为 (10k, dim)
# input_tensor = torch.randn(3, 10000, 128)

# # 池化到目标大小 (16, dim)
# pool = nn.AdaptiveAvgPool2d((16, 128))
# output = pool(input_tensor)
# print(output.shape) # 输出 (16, 128)

# pool = nn.AdaptiveAvgPool2d((2, 128))

# # 分块逻辑
# chunks = torch.chunk(input_tensor, 8, dim=1)  # 分成 8 chunks
# chunk_outputs = [pool(chunk) for chunk in chunks]  # 对每个 chunk 池化
# output1 = torch.cat(chunk_outputs, dim=1)
# breakpoint()