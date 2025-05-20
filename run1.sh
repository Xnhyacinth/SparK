
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
gpus=${1:-"0"}
model=${2:-"llama3.1-8b-inst"}
press_name=${3:-"full_kv"}
compress_questions=${4:-"0"}
temperature=${5:-"0.0"}

if [ "$model" = "llama3-8b-inst" ];then
    model_name_or_path=/modelopsnas/modelops/models/meta-llama/Meta-Llama-3-8B-Instruct
    # model_name_or_path=/modelopsnas/modelops/models/meta-llama/unsloth__llama-3-8b-Instruct
fi
if [ "$model" = "llama3.1-8b-inst" ];then
    model_name_or_path=/modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct
fi
if [ "$model" = "qwen3-32b" ];then
    model_name_or_path=/modelopsnas/modelops/models/Qwen3-32B
fi
if [ "$model" = "llama3-70b-inst" ];then
    model_name_or_path=/modelopsnas/modelops/models/LLM-Research/Meta-Llama-3-70B-Instruct
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
(
  for dataset in $dataset_list
    do
      # bash evaluate.sh ${dataset}
      CUDA_VISIBLE_DEVICES=${gpus} python -u eval.py --dataset longbench --data_dir ${dataset} --model ${model_name_or_path} --press_name ${press_name} --compression_ratio 0 --device "auto" --temperature ${temperature} --save_dir /modelopsnas/modelops/468440/kvpress/output ${extra_args}
    done
) > ${output_prefix}/${press_name}_${compress_questions}_t${temperature}.log 2>&1 &

# 等待所有后台任务完成
wait