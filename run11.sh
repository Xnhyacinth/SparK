
if [ ! -d "/modelopsnas" ]; then
    mkdir -p "/modelopsnas"
    echo "/modelopsnas created"
fi

nas="alipayheyuan2-33-fdf14.cn-heyuan-alipay.nas.aliyuncs.com" #10T
sudo mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport $nas:/ /modelopsnas
#pip install transformers==4.37.0

#model_path=/codenas/user/bingchang/checkpoints/merged/qwen-1.8b-quality-classifier-v1-0118-55000/
# model_path=/ainative/modelops/246872/models/Qwen2-7B-Instruct

#######################################################################
#            
#                         Group 1
######################################################################
# pip install transformers==4.51.0
bash install.sh
# bash evaluate.sh llama3-8b-inst 1
# bash run1.sh 0,1,2,3,4,5,6,7  qwen3-32b full_kv 1 0.0
press_names=("streaming_llm" "snapkv" "tova" "observed_attention" "expected_attention" "adasnapkv")
# press_names=("snap_think" "snap_adathink")
# press_names=("snap_adathink")
# for press in $press_names
for press in "${press_names[@]}"; 
  do
    bash evaluate1.sh qwen3-32b 1 0.5 ${press} 0,1,2,3 0.0 0.0
    # bash evaluate.sh qwen3-32b 1 0.5 ${press} 0,1,2,3 0.0 0.0
    # bash evaluate0.sh llama3-70b-inst 1 0.5 ${press} 0,1,2,3 0.0 0.0
    # bash evaluate.sh llama3-70b-inst 1 0.5 ${press} 0,1,2,3 0.0 0.0
    # bash evaluate1.sh qwen3-32b 1 0.5 snap_think 0,1 0.0 0.0
    # bash evaluate.sh llama3-8b-inst 1 0.5 ${press} 0,1 0.0 0.0
    # bash evaluate.sh llama3-8b-inst 1 0.6 ${press} 0,1 0.0 0.0
    # bash evaluate.sh llama3-8b-inst 1 0.4 ${press} 0,1 0.0 0.0
    # bash evaluate.sh llama3-8b-inst 1 0.7 ${press} 0,1 0.0 0.0
    # bash evaluate.sh llama3-8b-inst 1 0.8 ${press} 0,1 0.0 0.0
    # bash evaluate.sh llama3-8b-inst 1 0.9 ${press} 0,1 0.0 0.0
    # bash evaluate.sh llama3-8b-inst 1 0.3 ${press} 0,1 0.0 0.0
    # bash evaluate.sh llama3-8b-inst 1 0.2 ${press} 0,1 0.0 0.0
    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test0.sh llama3-8b-inst 1 0.6 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test0.sh llama3-8b-inst 1 0.4 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test0.sh llama3-8b-inst 1 0.3 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test0.sh llama3-8b-inst 1 0.7 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test0.sh llama3-8b-inst 1 0.8 snap_adathink 0,1 0.0 0.0 0.7554

    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test1.sh llama3-8b-inst 1 0.6 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test1.sh llama3-8b-inst 1 0.4 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test1.sh llama3-8b-inst 1 0.3 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test1.sh llama3-8b-inst 1 0.7 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test1.sh llama3-8b-inst 1 0.8 snap_adathink 0,1 0.0 0.0 0.7554
    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.994 

    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.994 

    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.995 

    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.995 
    # echo ${press}
    done
# bash eval.sh llama3-8b-inst 1
# dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"

# for dataset in $dataset_list
#   do
#     bash evaluate.sh ${dataset} compress
#     # python eval.py --dataset longbench --data_dir ${dataset} --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snapkv --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output
#   done

# 等待所有后台任务完成
wait

# python eval.py --dataset longbench --data_dir narrativeqa --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snap_adathink --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output00