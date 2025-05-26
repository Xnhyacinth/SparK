
import re

def extract_selected_channels(log_file_path):
    # 用于存储提取的数字
    selected_channels_list = []
    
    # 正则表达式模式，匹配 "Number of selected channels: " 后的数字
    pattern = r"Number of selected channels:\s*(\d+)"
    
    # 打开并读取文件
    with open(log_file_path, 'r') as file:
        for line in file:
            # 使用正则表达式查找匹配的内容
            match = re.search(pattern, line)
            if match:
                # 提取匹配到的数字，并转换为整数
                selected_channels = int(match.group(1))
                selected_channels_list.append(selected_channels)
    
    return selected_channels_list

for l in ['128', '512', '1024', '2048']:
  log_file_path = f'logs0/llama3-8b-inst/snap_adathink_{l}_1_channel0.5_t0.0_no0.0_0.99.log' 
  result = extract_selected_channels(log_file_path)
#   breakpoint()
  nums = 128 * int(l) * 8 * len(result)
  print(sum(result) / nums)