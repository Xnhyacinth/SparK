
if [ ! -d "/modelopsnas" ]; then
    mkdir -p "/modelopsnas"
    echo "/modelopsnas created"
    nas="alipayheyuan2-33-fdf14.cn-heyuan-alipay.nas.aliyuncs.com" #10T
    sudo mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport $nas:/ /modelopsnas
fi


#pip install transformers==4.37.0

#model_path=/codenas/user/bingchang/checkpoints/merged/qwen-1.8b-quality-classifier-v1-0118-55000/
# model_path=/ainative/modelops/246872/models/Qwen2-7B-Instruct

#######################################################################
#            
#                         Group 1
######################################################################
# kvpress_path=$(python -c "import os; import kvpress; kvpress_dir = os.path.dirname(kvpress.__file__); print(kvpress_dir)")
# echo $kvpress_path

# cp /modelopsnas/modelops/468440/kvpress/kvpress/presses/duo_attention_press.py $kvpress_path/presses
# bash install.sh
# dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
# # dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum lcc repobench-p"
# # dataset_list="passage_count passage_retrieval_en"
# compression_ratios=(128 512 1024)
# for compression_ratio in "${compression_ratios[@]}"; do
#   (
#   for dataset in $dataset_list
#     do
#       # bash evaluate.sh ${dataset}
#       CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --dataset longbench --data_dir ${dataset} --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snap_adathink --max_capacity_prompt ${compression_ratio} --key_channel_compression_ratio 0.9 --compress_questions True --device "auto" --save_dir /modelopsnas/modelops/468440/kvpress/output_score_0.9
#     done
#   ) > /modelopsnas/modelops/468440/kvpress/logs/myscore_0.9_${compression_ratio}.log 2>&1 &
# done

gpus=${1:-"0"}
model=${2:-"llama3.1-8b-inst"}
press_name=${3:-"snap_adathink"}
compress_questions=${4:-"0"}
key_channel_compression_ratio=${5:-"0.0"}
threshold_ratio=${6:-"0.0"}

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

# bash install.sh
dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
# dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum lcc repobench-p"
# dataset_list="passage_count passage_retrieval_en"
output_prefix=/modelopsnas/modelops/468440/kvpress/logs/${model}
mkdir -p ${output_prefix}

compression_ratios=(128 512 1024)
for compression_ratio in "${compression_ratios[@]}"; do
  (
  for dataset in $dataset_list
    do
      # bash evaluate.sh ${dataset}
      CUDA_VISIBLE_DEVICES=${gpus} python eval.py --dataset longbench --data_dir ${dataset} --model ${model_name_or_path} --press_name ${press_name} --max_capacity_prompt ${compression_ratio} --threshold_ratio ${threshold_ratio} --key_channel_compression_ratio ${key_channel_compression_ratio} --device "auto" --save_dir /modelopsnas/modelops/468440/kvpress/output ${extra_args}
    done
  ) 
  > ${output_prefix}/${press_name}_${compression_ratio}_${compress_questions}_channel${key_channel_compression_ratio}_threshold_${threshold_ratio}.log 2>&1 &
done

# 等待所有后台任务完成
wait
# bash run1.sh 0,1 /modelopsnas/modelops/models/meta-llama/Meta-Llama-3-8B-Instruct full_kv 0
# bash run2.sh 6,7 /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct snap_adathink 0
