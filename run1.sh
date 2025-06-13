
# kvpress_path=$(python -c "import os; import kvpress; kvpress_dir = os.path.dirname(kvpress.__file__); print(kvpress_dir)")
# echo $kvpress_path
gpus=${1:-"0"}
model=${2:-"llama3.1-8b-inst"}
press_name=${3:-"full_kv"}
compress_questions=${4:-"0"}
temperature=${5:-"0.0"}
dataset=${6:-"longbench"}

if [ "$model" = "llama3-8b-inst" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
fi
if [ "$model" = "llama3.1-8b-inst" ];then
    model_name_or_path=meta-llama/Llama-3.1-8B-Instruct
fi
if [ "$model" = "llama3.1-70b-inst" ];then
    model_name_or_path=meta-llama/Llama-3.1-70B-Instruct
fi
if [ "$model" = "qwen3-32b" ];then
    model_name_or_path=Qwen/Qwen3-32B
fi
if [ "$model" = "llama3-70b-inst" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-70B-Instruct
fi
if [ "$model" = "qwen3-8b" ];then
    model_name_or_path=Qwen/Qwen3-8B
fi
if [ "$model" = "qwen2.5-3b-inst" ];then
    model_name_or_path=Qwen/Qwen2.5-3B-Instruct
fi
if [ "$model" = "qwen2.5-32b-inst" ];then
    model_name_or_path=Qwen/Qwen2.5-32B-Instruct
fi

extra_args=""
if [[ $compress_questions != 0 ]];then
  extra_args="${extra_args} --compress_questions True"
fi

# bash install.sh
if [ "$dataset" = "longbench" ];then
    dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
fi
if [ "$dataset" = "ruler" ];then
    dataset_list="4096 8192 16384"
fi
if [ "$dataset" = "infinitebench" ];then
    dataset_list="passkey kv_retrieval number_string longdialogue_qa_eng longbook_qa_eng longbook_choice_eng code_run code_debug math_find math_calc longbook_sum_eng longbook_qa_chn"
fi

output_prefix=logs/${model}/${dataset}
mkdir -p ${output_prefix}
(
  for datadir in $dataset_list
    do
      # bash evaluate.sh ${dataset}
      CUDA_VISIBLE_DEVICES=${gpus} python -u eval.py --dataset ${dataset} --data_dir ${datadir} --model ${model_name_or_path} --press_name ${press_name} --compression_ratio 0 --device "auto" --temperature ${temperature} --save_dir output ${extra_args}
    done
) > ${output_prefix}/${press_name}_${compress_questions}_t${temperature}.log 2>&1 &

# 等待所有后台任务完成
wait