# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM



# "simonjegou/ruler"
from datasets import load_dataset

# for l in ['4096', '8192', '16384']:
#     ds = load_dataset("simonjegou/ruler", l, split='test')
# for task in [
#     "passkey",
#     "kv_retrieval",
#     "number_string",
#     "longdialogue_qa_eng",
#     "longbook_qa_eng",
#     "longbook_choice_eng",
#     "code_run",
#     "code_debug",
#     "math_find",
#     "math_calc",
#     "longbook_sum_eng",
#     "longbook_qa_chn",
# ]:
#     ds = load_dataset("MaxJeblick/InfiniteBench", task)

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
#     ds = load_dataset("Xnhyacinth/LongBench", task)