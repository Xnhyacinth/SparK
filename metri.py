import os
import json
import argparse
import numpy as np

# from longbench.metrics import (
#     qa_f1_score,
#     rouge_zh_score,
#     qa_f1_zh_score,
#     rouge_score,
#     classification_score,
#     retrieval_score,
#     retrieval_zh_score,
#     count_score,
#     code_sim_score,
# )

# dataset2metric = {
#     "narrativeqa": qa_f1_score,
#     "qasper": qa_f1_score,
#     "multifieldqa_en": qa_f1_score,
#     "multifieldqa_zh": qa_f1_zh_score,
#     "hotpotqa": qa_f1_score,
#     "2wikimqa": qa_f1_score,
#     "musique": qa_f1_score,
#     "dureader": rouge_zh_score,
#     "gov_report": rouge_score,
#     "qmsum": rouge_score,
#     "multi_news": rouge_score,
#     "vcsum": rouge_zh_score,
#     "trec": classification_score,
#     "triviaqa": qa_f1_score,
#     "samsum": rouge_score,
#     "lsht": classification_score,
#     "passage_retrieval_en": retrieval_score,
#     "passage_count": count_score,
#     "passage_retrieval_zh": retrieval_zh_score,
#     "lcc": code_sim_score,
#     "repobench-p": code_sim_score,
# }

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--compress_ratio', type=str, default=None)
    parser.add_argument('--com_channel', type=str, default='0.5')
    parser.add_argument('--threshold_ratio', type=str, default=None)
    parser.add_argument('--tem', type=str, default='0.0')
    parser.add_argument('--compress_q', type=str, default=None)
    parser.add_argument('--pooling_ratio', type=str, default='0.0')
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# def scorer_e(dataset, predictions, answers, lengths, all_classes):
#     scores = {"0-4k": [], "4-8k": [], "8k+": []}
#     for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
#         score = 0.
#         if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
#             prediction = prediction.lstrip('\n').split('\n')[0]
#         for ground_truth in ground_truths:
#             score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
#         if length < 4000:
#             scores["0-4k"].append(score)
#         elif length < 8000:
#             scores["4-8k"].append(score)
#         else:
#             scores["8k+"].append(score)
#     for key in scores.keys():
#         scores[key] = round(100 * np.mean(scores[key]), 2)
#     return scores
# def scorer(dataset, predictions, answers, all_classes):
#     total_score = 0.
#     for (prediction, ground_truths) in zip(predictions, answers):
#         score = 0.
#         if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
#             prediction = prediction.lstrip('\n').split('\n')[0]
#         for ground_truth in ground_truths:
#             score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
#         total_score += score
#     return round(100 * total_score / len(predictions), 2)

def read_res_json(directory, method_key):
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
        ['snap_adathink'],
        ['pyramid'],
        ['pyramid_think'],
        ['pyramid_adathink']
    ]
    
    model2maxlen = json.load(open("longbench/config/model2maxlen.json", "r"))
    max_context_length = model2maxlen[args.results_dir.split('/')[-1]]

    for dataset in dataset_list:
        
        results_list[0].append(dataset)
        
        for idx, method in enumerate(["full_kv", "streaming_llm", 'observed_attention', "expected_attention", "adasnapkv", "criti_snapkv", "tova", "random", "snapkv", "snap_think", 'snap_adathink', "pyramid", "pyramid_think", "pyramid_adathink"]):
            # "streaming_llm" "snapkv" "snap_think" "expected_attention" "adasnapkv" "criti_snapkv" "tova" "random"
        # for idx, method in enumerate(["H2_global", "PyramidKV_global", "local"]):
            try:
                args.method = method
                args.dataset = dataset
                if args.compress_q is not None:
                    if 'adathink' in method:
                        # breakpoint()
                        prefix = ""
                        if args.threshold_ratio is not None:
                            # breakpoint()
                            prefix += f"__threshold{args.threshold_ratio}"
                        if args.pooling_ratio != '0.0':
                            # breakpoint()
                            prefix += f"__no{args.pooling_ratio}"
                            # args.eval_file = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio, 'longbench', dataset, f"{method}__max_context{max_context_length}__channel{args.com_channel}.json")
                        
                        # else:
                        args.eval_file = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio, 'longbench', dataset)
                        # breakpoint()
                    elif 'think' in method:
                        # args.eval_file = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio, 'longbench', dataset)
                        import re
                        pattern = r'output[^/]*'
                        args.eval_file = os.path.join(re.sub(pattern, 'output_norm0', args.results_dir), 'compress_questions', args.tem, args.compress_ratio, 'longbench', dataset)
                    elif 'full_kv' in method:
                        import re
                        pattern = r'output[^/]*'
                        args.eval_file = os.path.join(re.sub(pattern, 'output', args.results_dir), 'compress_questions', args.tem, '0', 'longbench', dataset)    
                    else:
                        import re
                        pattern = r'output[^/]*'
                        args.eval_file = os.path.join(re.sub(pattern, 'output_norm0', args.results_dir), 'compress_questions', args.tem, args.compress_ratio, 'longbench', dataset)
                else:
                    if 'snap_adathink' in method:
                        # if args.pooling_ratio != '0.0':
                        #     args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'longbench', dataset, f"{method}__max_context{max_context_length}__no{args.pooling_ratio}__channel{args.com_channel}.json")
                        # else:
                        prefix = ""
                        if args.threshold_ratio is not None:
                            # breakpoint()
                            prefix += f"__threshold{args.threshold_ratio}"
                        if args.pooling_ratio != '0.0':
                            # breakpoint()
                            prefix += f"__no{args.pooling_ratio}"
                            # args.eval_file = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio, 'longbench', dataset, f"{method}__max_context{max_context_length}__channel{args.com_channel}.json")
                        
                        args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'longbench', dataset)
                    elif 'think' in method:
                        args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'longbench', dataset)
                    elif 'full_kv' in method:
                        args.eval_file = os.path.join(args.results_dir, args.tem, '0', 'longbench', dataset)    
                    else:
                        args.eval_file = os.path.join(args.results_dir, args.tem, args.compress_ratio, 'longbench', dataset)
                
                
                # try:
                
                scores = dict()
                # breakpoint()
                if 'full_kv' in method:
                    score = read_res_json(args.eval_file, f"{method}_0.0")
                    # if score == -1:
                    #     score = read_res_json(args.eval_file, f"{method}")
                elif 'think' in method:
                    score = read_res_json(args.eval_file, f"{method}_{args.com_channel}")
                    if score == -1:
                        score = read_res_json(args.eval_file, f"{method}_{args.com_channel}_{args.pooling_ratio}")
                else:
                    score = read_res_json(args.eval_file, f"{method}_0.5")
                if score == -1:
                    score = read_res_json(args.eval_file, f"{method}")


                scores[args.dataset] = score
                    # if dataset == 'qasper':
                    #     scores[dataset + '_e'] = score_e
                    
                # if args.longbench_e:
                #     out_path = f"H2O/results/{args.model}/result.json"
                # else:
                #     out_path = f"H2O/results/{args.model}/result.json"
                    # out_path_e = f"pred/{args.model}/result_e.json"
                    # with open(out_path_e, "w") as f:
                    #     json.dump(score_e, f, ensure_ascii=False, indent=4)
                    
                output_dir = os.path.dirname(args.eval_file)
                
                results_list[idx+1].append(score)
                
                with open(os.path.join(output_dir, f"{args.method}_metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
            
                print(f"dataset {args.dataset} method {args.method} scores {scores}")
            except:
                # breakpoint()
                results_list[idx+1].append(-1)
                
                print(f"dataset {args.dataset} method {args.method} scores {None}")

    # df['Average'] = df.iloc[:, 1:].mean(axis=1)  
    for i in range(1, len(results_list)):
        row_values = [val for val in results_list[i][1:] if isinstance(val, (int, float)) and val != -1] # Exclude header and -1 scores
        if row_values:
            average = round(sum(row_values) / len(row_values), 2)
            results_list[i].append(average)
        else:
            results_list[i].append(0) # Or some other placeholder if all scores were -1 or no scores

    # Add "Average" header for the new column
    results_list[0].append("Average")
    import csv
    if args.compress_q is None:
        outpath = os.path.join(args.results_dir, args.tem, args.compress_ratio)
    else:
        outpath = os.path.join(args.results_dir, 'compress_questions', args.tem, args.compress_ratio)
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, f"results_{args.com_channel}.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)
