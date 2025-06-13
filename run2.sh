
# pip install transformers==4.51.0
# bash install.sh
# bash evaluate.sh llama3-8b-inst 1
# bash run1.sh 4,5,6,7 llama3-8b-inst full_kv 1 0.0
press_names=("pyramid_adathink")
compression_ratios=(0.5 0.6 0.7 0.8 0.9 0.4 0.3)
# press_names=("snap_think" "snap_adathink")
# press_names=("snap_adathink")
# for press in $press_names
for ratio in "${compression_ratios[@]}"; do
    for press in "${press_names[@]}"; do
      echo "Running evaluation for press: ${press} with ratio: ${ratio}"
      bash test0.sh llama3-8b-inst 1 ${ratio} ${press} 4,5 0.0 0.0 0.65
    done
done
# bash eval.sh llama3-8b-inst 1
# dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"

# for dataset in $dataset_list
#   do
#     bash evaluate.sh ${dataset} compress
#     # python eval.py --dataset longbench --data_dir ${dataset} --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snapkv --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output
#   done
# bash evaluate.sh qwen2.5-3b-inst 1 0.5 snap_think 0,1,2,3,4,5,6,7 0.0 0.0
# bash evaluate.sh mistral-8b-inst 1 0.5 snap_think 0,1,2,3,4,5,6,7 0.0 0.0
# bash evaluate.sh qwen2.5-3b-inst 1 0.5 snap_think 0 0.0 0.0
# ps -ef |grep snap|grep -v grep |cut -c 9-16|xargs kill -9
# ps -ef |grep down.py|grep -v grep |cut -c 9-16|xargs kill -9
# 等待所有后台任务完成
wait
# ps -ef |grep pyramidkv|grep -v grep |cut -c 9-16|xargs kill -9
# python eval.py --dataset longbench --data_dir narrativeqa --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snap_adathink --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output00