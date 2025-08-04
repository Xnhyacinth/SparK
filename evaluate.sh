
model=${1:-"llama3.1-8b-inst"}
compress_questions=${2:-"0"}
key_channel_compression_ratio=${3:-"0.5"}
press=${4:-"snapkv"}
gpus=${5:-"0"}
temperature=${6:-"0.0"}
threshold_ratio=${7:-"0.0"}
pooling_ratio=${8:-"0.0"}
mode=${9:-"no"}
dataset=${10:-"longbench"}
value_channel_compression_ratio=${11:-"0"}

extra_args=""
extra_name=""


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

if [[ $compress_questions != 0 ]];then
  extra_args="${extra_args} --compress_questions True"
fi
if [[ $value_channel_compression_ratio != 0 ]];then
  extra_args="${extra_args} --value_channel_compression_ratio ${value_channel_compression_ratio}"
  extra_name="${extra_name}__value${value_channel_compression_ratio}"

fi

if [ "$dataset" = "longbench" ];then
    dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
    compression_ratios=(128 512 1024 2048)
    extra_args="${extra_args} --max_capacity_prompt  "
fi
if [ "$dataset" = "ruler" ];then
    dataset_list="4096 8192 16384"
    compression_ratios=(0.8 0.5)
    extra_args="${extra_args} --compression_ratio  "
fi
if [ "$dataset" = "infinitebench" ];then
    dataset_list="passkey kv_retrieval number_string longdialogue_qa_eng longbook_qa_eng longbook_choice_eng code_run code_debug math_find math_calc longbook_sum_eng longbook_qa_chn"
fi

model_basename=$(basename "$model_name_or_path")
output_prefix=logs/${model}/${dataset}
mkdir -p ${output_prefix}


for compression_ratio in "${compression_ratios[@]}"; do
  (
    for data_dir in $dataset_list; do
      echo "Running press_name: $press on dataset ${data_dir} with compression_ratio: $compression_ratio on GPU cuda:$i"
      log_file="output_norm/results/${model_basename}/compress_questions/0.0/${compression_ratio}/${dataset}/${data_dir}/${press}__max_context127500${extra_name}.json"
      if [ -f "$log_file" ]; then
        echo "Log file $log_file already exists, skipping."
        continue
      fi
      log_file="output_norm/results/${model_basename}/compress_questions/0.0/${compression_ratio}/${dataset}/${data_dir}/${press}__max_context127500__channel${key_channel_compression_ratio}${extra_name}.json"
      if [ -f "$log_file" ]; then
        echo "Log file $log_file already exists, skipping."
        continue
      fi

      log_file="output_norm/results/${model_basename}/compress_questions/0.0/${compression_ratio}/${dataset}/${data_dir}/${press}__max_context31500${extra_name}.json"
      if [ -f "$log_file" ]; then
        echo "Log file $log_file already exists, skipping."
        continue
      fi
      log_file="output_norm/results/${model_basename}/compress_questions/0.0/${compression_ratio}/${dataset}/${data_dir}/${press}__max_context31500__channel${key_channel_compression_ratio}${extra_name}.json"
      if [ -f "$log_file" ]; then
        echo "Log file $log_file already exists, skipping."
        continue
      fi

      log_file="output_norm/results/${model_basename}/compress_questions/0.0/${compression_ratio}/${dataset}/${data_dir}/${press}__max_context7950${extra_name}.json"
      if [ -f "$log_file" ]; then
        echo "Log file $log_file already exists, skipping."
        continue
      fi
      log_file="output_norm/results/${model_basename}/compress_questions/0.0/${compression_ratio}/${dataset}/${data_dir}/${press}__max_context7950__channel${key_channel_compression_ratio}${extra_name}.json"
      if [ -f "$log_file" ]; then
        echo "Log file $log_file already exists, skipping."
        continue
      fi
      CUDA_VISIBLE_DEVICES=${gpus} python -u eval.py --dataset $dataset --data_dir $data_dir --model $model_name_or_path --press_name $press --threshold_ratio ${threshold_ratio} --pooling_ratio ${pooling_ratio} --mode ${mode} --key_channel_compression_ratio ${key_channel_compression_ratio} --temperature ${temperature} --device "auto" --save_dir output_norm ${extra_args} ${compression_ratio}
    done
  )  > ${output_prefix}/${press}_${compression_ratio}_${compress_questions}_channel${key_channel_compression_ratio}_t${temperature}_${mode}${pooling_ratio}${extra_name}.log 2>&1
done


# Wait for all background jobs to finish
wait
echo "All evaluations completed."
