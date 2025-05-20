dataset="longbench"
model=${1:-"llama3.1-8b-inst"}
compress_questions=${2:-"0"}
key_channel_compression_ratio=${3:-"0.5"}
press=${4:-"snapkv"}
gpus=${5:-"0"}
temperature=${6:-"0.0"}
threshold_ratio=${7:-"0.0"}
pooling_ratio=${8:-"0.0"}
mode=${9:-"no"}
# model="/modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct"
dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
# compression_ratios=(0.1 0.25 0.5)
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
    model_name_or_path=/modelopsnas/modelops/models/meta-llama/Meta-Llama-3-8B-Instruct
    # model_name_or_path=/modelopsnas/modelops/models/meta-llama/unsloth__llama-3-8b-Instruct
fi
if [ "$model" = "llama3.1-8b-inst" ];then
    model_name_or_path=/modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct
fi

extra_args=""
if [[ $compress_questions != 0 ]];then
  extra_args="${extra_args} --compress_questions True"
fi

output_prefix=/modelopsnas/modelops/468440/kvpress/logs/${model}/${press}
mkdir -p ${output_prefix}
# Iterate over press names and compression ratios
# for i in "${!press_names[@]}"; do
#   press="${press_names[$i]}"
  
#   # Run each press_name on a different GPU in the background
#   (
#     for compression_ratio in "${compression_ratios[@]}"; do
#       for data_dir in $dataset_list; do
#         echo "Running press_name: $press on dataset ${data_dir} with compression_ratio: $compression_ratio on GPU cuda:$i"
#         python eval.py --dataset $dataset --data_dir $data_dir --model $model_name_or_path --press_name $press --max_capacity_prompt $compression_ratio --key_channel_compression_ratio ${key_channel_compression_ratio} --device "auto" --save_dir /modelopsnas/modelops/468440/kvpress/output ${extra_args}
#       done
#     done
#   ) > ${output_prefix}/${press}_${compression_ratio}_${compress_questions}_channel${key_channel_compression_ratio}.log 2>&1 &
# done

for compression_ratio in "${compression_ratios[@]}"; do
  (
    for data_dir in $dataset_list; do
      echo "Running press_name: $press on dataset ${data_dir} with compression_ratio: $compression_ratio on GPU cuda:$i"
      CUDA_VISIBLE_DEVICES=${gpus} python -u eval.py --dataset $dataset --data_dir $data_dir --model $model_name_or_path --press_name $press --max_capacity_prompt $compression_ratio --threshold_ratio ${threshold_ratio} --pooling_ratio ${pooling_ratio} --mode ${mode} --key_channel_compression_ratio ${key_channel_compression_ratio} --temperature ${temperature} --device "auto" --save_dir /modelopsnas/modelops/468440/kvpress/output ${extra_args}
    done
  ) > ${output_prefix}/${compression_ratio}_${compress_questions}_channel${key_channel_compression_ratio}_t${temperature}_${mode}${pooling_ratio}.log 2>&1
done


# Wait for all background jobs to finish
wait
echo "All evaluations completed."
