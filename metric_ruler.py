import os
import json
import argparse
import numpy as np
import csv
import ast

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--compress_ratio', type=str, default=None)
    parser.add_argument('--com_channel', type=str, default='0.5')
    parser.add_argument('--threshold_ratio', type=str, default=None)
    parser.add_argument('--tem', type=str, default='0.0')
    parser.add_argument('--compress_q', type=str, default=None)
    parser.add_argument('--pooling_ratio', type=str, default='0.0')
    return parser.parse_args(args)

def read_ruler_results(directory, method_key):
    """
    读取 res.json 文件中指定 method_key 的 ruler 结果
    格式如: "full_kv_0.0: {'cwe': {'string_match': 88.86}, ...}"
    """
    res_json_path = os.path.join(directory, "res.json")
    
    if not os.path.exists(res_json_path):
        print(f"File not found: {res_json_path}")
        return None
    
    try:
        with open(res_json_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            if ':' in line:
                method, result_str = line.split(':', 1)
                method = method.strip()
                result_str = result_str.strip()

                if method == method_key:
                    try:
                        # 使用 ast.literal_eval 安全地解析字典字符串
                        results_dict = ast.literal_eval(result_str)
                        return results_dict
                    except (ValueError, SyntaxError) as e:
                        print(f"Invalid result format: {result_str} for method {method}")
                        print(f"Error: {e}")
                        return None
        
        print(f"Method {method_key} not found in {res_json_path}")
        return None
        
    except Exception as e:
        print(f"Error reading {res_json_path}: {e}")
        return None

def extract_ruler_scores(results_dict):
    """
    从结果字典中提取各个任务的分数，按指定顺序排列
    顺序: Niah1, Niah2, Niah3, MKey1, MKey2, MKey3, MValue, MQuery, VT, CWE, FWE, QA1, QA2
    """
    if results_dict is None:
        return [-1] * 13  # 返回13个-1
    
    # 映射关系：显示名称 -> 实际key名称
    task_mapping = {
        'Niah1': 'niah_single_1',
        'Niah2': 'niah_single_2', 
        'Niah3': 'niah_single_3',
        'MKey1': 'niah_multikey_1',
        'MKey2': 'niah_multikey_2',
        'MKey3': 'niah_multikey_3',
        'MValue': 'niah_multivalue',
        'MQuery': 'niah_multiquery',
        'VT': 'vt',
        'CWE': 'cwe',
        'FWE': 'fwe',
        'QA1': 'qa_1',
        'QA2': 'qa_2'
    }
    
    scores = []
    for task_name in ['Niah1', 'Niah2', 'Niah3', 'MKey1', 'MKey2', 'MKey3', 'MValue', 'MQuery', 'VT', 'CWE', 'FWE', 'QA1', 'QA2']:
        actual_key = task_mapping[task_name]
        if actual_key in results_dict and 'string_match' in results_dict[actual_key]:
            score = results_dict[actual_key]['string_match']
            scores.append(score)
        else:
            print(f"Warning: {actual_key} not found in results or missing string_match")
            scores.append(-1)
    
    return scores

if __name__ == '__main__':
    args = parse_args()
    
    # 方法列表
    methods = [
        "full_kv", "streaming_llm", 'observed_attention', "expected_attention", 
        "adasnapkv", "criti_snapkv", "tova", "random", "snapkv", "snap_think", 
        'snap_adathink', "pyramidkv", "pyramid_think", "pyramid_adathink"
    ]
    
    # 结果列表，第一行是表头
    results_list = [
        ["Method", "Niah1", "Niah2", "Niah3", "MKey1", "MKey2", "MKey3", "MValue", "MQuery", "VT", "CWE", "FWE", "QA1", "QA2", "Average"]
    ]
    
    # 添加每个方法对应的行
    for method in methods:
        results_list.append([method])
    
    # 读取每个方法的结果
    for idx, method in enumerate(methods):
        try:
            args.method = method
            
            # 构建文件路径
            if args.compress_q is not None:
                if 'adathink' in method:
                    prefix = ""
                    if args.threshold_ratio is not None:
                        prefix += f"__threshold{args.threshold_ratio}"
                    if args.pooling_ratio != '0.0':
                        prefix += f"__no{args.pooling_ratio}"
                    
                    args.eval_file = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio, 'ruler')
                elif 'think' in method:
                    import re
                    pattern = r'output[^/]*'
                    args.eval_file = os.path.join(re.sub(pattern, 'output_norm', args.results_dir), 'compress_questions', args.tem, args.compress_ratio, 'ruler')
                elif 'full_kv' in method:
                    import re
                    pattern = r'output[^/]*'
                    args.eval_file = os.path.join(re.sub(pattern, 'output', args.results_dir), 'compress_questions', args.tem, '0', 'ruler')    
                else:
                    import re
                    pattern = r'output[^/]*'
                    args.eval_file = os.path.join(re.sub(pattern, 'output_norm', args.results_dir), 'compress_questions', args.tem, args.compress_ratio, 'ruler')
            else:
                if 'snap_adathink' in method:
                    prefix = ""
                    if args.threshold_ratio is not None:
                        prefix += f"__threshold{args.threshold_ratio}"
                    if args.pooling_ratio != '0.0':
                        prefix += f"__no{args.pooling_ratio}"
                    
                    args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'ruler')
                elif 'think' in method:
                    args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'ruler')
                elif 'full_kv' in method:
                    args.eval_file = os.path.join(args.results_dir, args.tem, '0', 'ruler')    
                else:
                    args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'ruler')
            
            # 读取结果
            if 'full_kv' in method:
                method_key = f"{method}_0.0"
            elif 'think' in method:
                method_key = f"{method}_{args.com_channel}"
            else:
                method_key = f"{method}_0.5"
            
            results_dict = read_ruler_results(args.eval_file, method_key)
            
            # 如果第一个key读取失败，尝试直接用方法名
            if results_dict is None:
                results_dict = read_ruler_results(args.eval_file, method)
            
            # 提取分数
            scores = extract_ruler_scores(results_dict)
            
            # 计算平均值（排除-1的值）
            valid_scores = [score for score in scores if score != -1]
            if valid_scores:
                average = round(sum(valid_scores) / len(valid_scores), 2)
            else:
                average = -1
            
            # 添加到结果列表
            results_list[idx + 1].extend(scores)
            results_list[idx + 1].append(average)
            
            print(f"Method {method}: scores {scores}, average {average}")
            
        except Exception as e:
            print(f"Error processing method {method}: {e}")
            # 添加-1填充
            results_list[idx + 1].extend([-1] * 14)  # 13个任务分数 + 1个平均值
    
    # 保存结果到CSV文件
    if args.compress_q is None:
        outpath = os.path.join(args.results_dir, args.tem, args.compress_ratio)
    else:
        outpath = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio)
    
    os.makedirs(outpath, exist_ok=True)
    
    with open(os.path.join(outpath, f"ruler_results_{args.com_channel}.csv"), 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)
    
    print(f"\nResults saved to: {os.path.join(outpath, f'ruler_results_{args.com_channel}.csv')}")
    
    # 打印结果表格
    print("\n" + "="*100)
    print("RULER Results Summary:")
    print("="*100)
    for row in results_list:
        print("\t".join([str(x) for x in row]))
