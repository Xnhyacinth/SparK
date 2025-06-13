dataset=${10:-"longbench"}
model=${1:-"llama3.1-8b-inst"}
compress_questions=${2:-"0"}
key_channel_compression_ratio=${3:-"0.5"}
press=${4:-"snapkv"}
gpus=${5:-"0"}
temperature=${6:-"0.0"}
threshold_ratio=${7:-"0.0"}
pooling_ratio=${8:-"0.0"}
mode=${9:-"no"}

if [ "$dataset" = "longbench" ];then
    dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
fi
if [ "$dataset" = "ruler" ];then
    dataset_list="4096 8192 16384"
fi
if [ "$dataset" = "infinitebench" ];then
    dataset_list="passkey kv_retrieval number_string longdialogue_qa_eng longbook_qa_eng longbook_choice_eng code_run code_debug math_find math_calc longbook_sum_eng longbook_qa_chn"
fi
compression_ratios=(1024 2048)

# press_names=("expected_attention" "knorm" "streaming_llm" "snapkv" "snap_think" "adakv" "observed_attention")
# press_names=("snapkv")
# press_names=("streaming_llm" "snapkv" "snap_think" "expected_attention" "adasnapkv" "criti_snapkv" "tova" "observed_attention")

# Check if the number of press names is less than or equal to the number of available GPUs
# num_gpus=$(nvidia-smi --list-gpus | wc -l)
# if [ ${#press_names[@]} -gt $num_gpus ]; then
#   echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs ($num_gpus)"
#   exit 1
# fi

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
if [ "$model" = "mistral-8b-inst" ];then
    model_name_or_path=mistralai/Ministral-8B-Instruct-2410
fi



extra_args=""
if [[ $compress_questions != 0 ]];then
  extra_args="${extra_args} --compress_questions True"
fi

output_prefix=logs/${model}/${dataset}
mkdir -p ${output_prefix}
# Iterate over press names and compression ratios
# for i in "${!press_names[@]}"; do
#   press="${press_names[$i]}"
  
#   # Run each press_name on a different GPU in the background
#   (
#     for compression_ratio in "${compression_ratios[@]}"; do
#       for data_dir in $dataset_list; do
#         echo "Running press_name: $press on dataset ${data_dir} with compression_ratio: $compression_ratio on GPU cuda:$i"
#         python eval.py --dataset $dataset --data_dir $data_dir --model $model_name_or_path --press_name $press --max_capacity_prompt $compression_ratio --key_channel_compression_ratio ${key_channel_compression_ratio} --device "auto" --save_dir output ${extra_args}
#       done
#     done
#   ) > ${output_prefix}/${press}_${compression_ratio}_${compress_questions}_channel${key_channel_compression_ratio}.log 2>&1 &
# done

for compression_ratio in "${compression_ratios[@]}"; do
  (
    for data_dir in $dataset_list; do
      echo "Running press_name: $press on dataset ${data_dir} with compression_ratio: $compression_ratio on GPU cuda:$i"
      CUDA_VISIBLE_DEVICES=${gpus} python -u eval.py --dataset $dataset --data_dir $data_dir --model $model_name_or_path --press_name $press --max_capacity_prompt $compression_ratio --threshold_ratio ${threshold_ratio} --pooling_ratio ${pooling_ratio} --mode ${mode} --key_channel_compression_ratio ${key_channel_compression_ratio} --temperature ${temperature} --device "auto" --save_dir output_norm0 ${extra_args}
    done
  )
  #  > ${output_prefix}/${press}_${compression_ratio}_${compress_questions}_channel${key_channel_compression_ratio}_t${temperature}_${mode}${pooling_ratio}.log 2>&1
done


# Wait for all background jobs to finish
wait
echo "All evaluations completed."
