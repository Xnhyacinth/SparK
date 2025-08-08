import os
import json
import argparse
import numpy as np

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--compress_ratio', type=str, default=None)
    parser.add_argument('--value_compress_ratio', type=str, default=None)
    parser.add_argument('--com_channel', type=str, default='0.5')
    parser.add_argument('--threshold_ratio', type=str, default=None)
    parser.add_argument('--tem', type=str, default='0.0')
    parser.add_argument('--compress_q', type=str, default=None)
    parser.add_argument('--pooling_ratio', type=str, default='0.0')
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def read_res_json(directory, method_key):
    """Read results from res.json file in the specified directory."""
    res_json_path = os.path.join(directory, "res.json")
    
    if not os.path.exists(res_json_path):
        print(f"File not found: {res_json_path}")
        return -1
    
    try:
        with open(res_json_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            if ':' in line:
                method, score_str = line.split(':', 1)
                method = method.strip()
                score_str = score_str.strip()

                if method == method_key:
                    try:
                        return float(score_str)
                    except ValueError:
                        print(f"Invalid score format: {score_str} for method {method}")
                        return -1
        
        print(f"Method {method_key} not found in {res_json_path}")
        return -1
        
    except Exception as e:
        print(f"Error reading {res_json_path}: {e}")
        return -1


if __name__ == '__main__':
    args = parse_args()
    
    dataset_list = [
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "gov_report",
        "qmsum",
        "multi_news",
        "trec",
        "triviaqa",
        "samsum",
        "passage_count",
        "passage_retrieval_en",
        "lcc",
        "repobench-p"
    ]
    
    results_list = [
        ["dataset"],
        ['full_kv'],
        ['streaming_llm'],
        ['observed_attention'],
        ['expected_attention'],
        ['adasnapkv'],
        ['criti_snapkv'],
        ['tova'],
        ['random'],
        ['snapkv'],
        ['snap_think'],
        ['snap_spark'],
        ['pyramidkv'],
        ['pyramid_think'],
        ['pyramid_spark']
    ]
    
    model2maxlen = json.load(open("longbench/config/model2maxlen.json", "r"))
    max_context_length = model2maxlen[args.results_dir.split('/')[-1]]

    for dataset in dataset_list:
        
        results_list[0].append(dataset)
        
        for idx, method in enumerate(["full_kv", "streaming_llm", 'observed_attention', "expected_attention", "adasnapkv", "criti_snapkv", "tova", "random", "snapkv", "snap_think", 'snap_spark', "pyramidkv", "pyramid_think", "pyramid_spark"]):
            try:
                args.method = method
                args.dataset = dataset
                
                # Determine evaluation file path based on compression settings and method
                if args.compress_q is not None:
                    if 'spark' in method:
                        prefix = ""
                        if args.threshold_ratio is not None:
                            prefix += f"__threshold{args.threshold_ratio}"
                        if args.pooling_ratio != '0.0':
                            prefix += f"__no{args.pooling_ratio}"
                        
                        args.eval_file = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio, 'longbench', dataset)
                    elif 'think' in method:
                        import re
                        pattern = r'output[^/]*'
                        args.eval_file = os.path.join(re.sub(pattern, 'output_norm', args.results_dir), 'compress_questions', args.tem, args.compress_ratio, 'longbench', dataset)
                    elif 'full_kv' in method:
                        import re
                        pattern = r'output[^/]*'
                        args.eval_file = os.path.join(re.sub(pattern, 'output', args.results_dir), 'compress_questions', args.tem, '0', 'longbench', dataset)
                    else:
                        import re
                        pattern = r'output[^/]*'
                        args.eval_file = os.path.join(re.sub(pattern, 'output_norm', args.results_dir), 'compress_questions', args.tem, args.compress_ratio, 'longbench', dataset)
                else:
                    if 'snap_spark' in method:
                        prefix = ""
                        if args.threshold_ratio is not None:
                            prefix += f"__threshold{args.threshold_ratio}"
                        if args.pooling_ratio != '0.0':
                            prefix += f"__no{args.pooling_ratio}"
                        
                        args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'longbench', dataset)
                    elif 'think' in method:
                        args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'longbench', dataset)
                    elif 'full_kv' in method:
                        args.eval_file = os.path.join(args.results_dir, args.tem, '0', 'longbench', dataset)
                    else:
                        args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'longbench', dataset)
                
                # Read scores based on method type
                scores = dict()
                if 'full_kv' in method:
                    score = read_res_json(args.eval_file, f"{method}_0.0")
                elif 'spark' in method and args.value_compress_ratio is not None:
                    score = read_res_json(args.eval_file, f"{method}_{args.com_channel}_{args.value_compress_ratio}")
                    if score == -1:
                        score = read_res_json(args.eval_file, f"{method}_{args.com_channel}_{args.pooling_ratio}_{args.value_compress_ratio}")
                elif 'think' in method:
                    score = read_res_json(args.eval_file, f"{method}_{args.com_channel}")
                    if score == -1:
                        score = read_res_json(args.eval_file, f"{method}_{args.com_channel}_{args.pooling_ratio}")
                else:
                    score = read_res_json(args.eval_file, f"{method}_0.5")
                    if score == -1:
                        score = read_res_json(args.eval_file, f"{method}_0.5_{args.pooling_ratio}")
                    
                if score == -1:
                    score = read_res_json(args.eval_file, f"{method}")

                scores[args.dataset] = score
                output_dir = os.path.dirname(args.eval_file)
                results_list[idx+1].append(score)
                
                with open(os.path.join(output_dir, f"{args.method}_metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
            
                print(f"dataset {args.dataset} method {args.method} scores {scores}")
                
            except:
                results_list[idx+1].append(-1)
                print(f"dataset {args.dataset} method {args.method} scores {None}")

    # Calculate average for each method, excluding invalid scores
    for i in range(1, len(results_list)):
        row_values = [val for val in results_list[i][1:] if isinstance(val, (int, float)) and val != -1]
        if row_values:
            average = round(sum(row_values) / len(row_values), 2)
            results_list[i].append(average)
        else:
            results_list[i].append(0)  # Placeholder if all scores were -1 or no scores

    # Add "Average" header for the new column
    results_list[0].append("Average")
    
    import csv
    if args.compress_q is None:
        outpath = os.path.join(args.results_dir, args.tem, args.compress_ratio)
    else:
        outpath = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio)
    os.makedirs(outpath, exist_ok=True)
    
    if args.value_compress_ratio is not None:
        with open(os.path.join(outpath, f"results_{args.com_channel}_{args.value_compress_ratio}.csv"), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(results_list)
    else:
        with open(os.path.join(outpath, f"results_{args.com_channel}.csv"), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(results_list)
