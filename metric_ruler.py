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
    Read ruler results for specified method_key from res.json file
    Format like: "full_kv_0.0: {'cwe': {'string_match': 88.86}, ...}"
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
                        # Use ast.literal_eval to safely parse dictionary string
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
    Extract scores for each task from results dictionary, arranged in specified order
    Order: Niah1, Niah2, Niah3, MKey1, MKey2, MKey3, MValue, MQuery, VT, CWE, FWE, QA1, QA2
    """
    if results_dict is None:
        return [-1] * 13  # Return 13 -1s
    
    # Mapping: display name -> actual key name
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
    
    # Method list
    methods = [
        "full_kv", "streaming_llm", 'observed_attention', "expected_attention", 
        "adasnapkv", "criti_snapkv", "tova", "random", "snapkv", "snap_think", 
        'snap_quark', "pyramidkv", "pyramid_think", "pyramid_quark"
    ]
    datasets = ['16384', '8192', '4096']
    
    # Read results for each method
    for dataset in datasets:
        # Results list, first row is header
        results_list = [
            ["Method", "Niah1", "Niah2", "Niah3", "MKey1", "MKey2", "MKey3", "MValue", "MQuery", "VT", "CWE", "FWE", "QA1", "QA2", "Average"]
        ]
        
        # Add corresponding row for each method
        for method in methods:
            results_list.append([method])

        for idx, method in enumerate(methods):
            try:
                args.method = method
            
            
                # Construct file path
                if args.compress_q is not None:
                    if 'quark' in method:
                        prefix = ""
                        if args.threshold_ratio is not None:
                            prefix += f"__threshold{args.threshold_ratio}"
                        if args.pooling_ratio != '0.0':
                            prefix += f"__no{args.pooling_ratio}"
                        
                        args.eval_file = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio, 'ruler', dataset)
                    elif 'think' in method:
                        import re
                        pattern = r'output[^/]*'
                        args.eval_file = os.path.join(re.sub(pattern, 'output_norm', args.results_dir), 'compress_questions', args.tem, args.compress_ratio, 'ruler', dataset)
                    elif 'full_kv' in method:
                        import re
                        pattern = r'output[^/]*'
                        args.eval_file = os.path.join(re.sub(pattern, 'output', args.results_dir), 'compress_questions', args.tem, '0', 'ruler', dataset)    
                    else:
                        import re
                        pattern = r'output[^/]*'
                        args.eval_file = os.path.join(re.sub(pattern, 'output_norm', args.results_dir), 'compress_questions', args.tem, args.compress_ratio, 'ruler', dataset)
                else:
                    if 'snap_quark' in method:
                        prefix = ""
                        if args.threshold_ratio is not None:
                            prefix += f"__threshold{args.threshold_ratio}"
                        if args.pooling_ratio != '0.0':
                            prefix += f"__no{args.pooling_ratio}"
                        
                        args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'ruler', dataset)
                    elif 'think' in method:
                        args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'ruler', dataset)
                    elif 'full_kv' in method:
                        args.eval_file = os.path.join(args.results_dir, args.tem, '0', 'ruler', dataset)    
                    else:
                        args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'ruler', dataset)
                
                # Read results
                if 'full_kv' in method:
                    method_key = f"{method}_0.0"
                elif 'think' in method:
                    method_key = f"{method}_{args.com_channel}"
                else:
                    method_key = f"{method}_0.5"
                
                results_dict = read_ruler_results(args.eval_file, method_key)
                
                # If first key reading fails, try using method name directly
                if results_dict is None:
                    results_dict = read_ruler_results(args.eval_file, f"{method_key}_{args.pooling_ratio}")
                
                
                # Extract scores
                scores = extract_ruler_scores(results_dict)
                
                # Calculate average (excluding -1 values)
                valid_scores = [score for score in scores if score != -1]
                if valid_scores:
                    average = round(sum(valid_scores) / len(valid_scores), 2)
                else:
                    average = -1
                
                # Add to results list
                results_list[idx + 1].extend(scores)
                results_list[idx + 1].append(average)
                
                print(f"Method {method}: scores {scores}, average {average}")
            
            except Exception as e:
                print(f"Error processing method {method}: {e}")
                # Add -1 padding
                results_list[idx + 1].extend([-1] * 14)  # 13 task scores + 1 average
    
        # Save results to CSV file
        if args.compress_q is None:
            outpath = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'ruler', dataset)
        else:
            outpath = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio, 'ruler', dataset)
        
        os.makedirs(outpath, exist_ok=True)
        
        with open(os.path.join(outpath, f"ruler_results_{args.com_channel}.csv"), 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(results_list)
        
        print(f"\nResults saved to: {os.path.join(outpath, f'ruler_results_{args.com_channel}.csv')}")
        
        # Print results table
        print("\n" + "="*100)
        print("RULER Results Summary:")
        print("="*100)
        for row in results_list:
            print("\t".join([str(x) for x in row]))
