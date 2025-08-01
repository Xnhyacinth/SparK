
#######################################################################
#            
#                         Group 1
######################################################################

bash install.sh
bash run1.sh 0,1,2,3,4,5,6,7 llama3.1-8b-inst full_kv 1 0.0
press_names=("streaming_llm" "snapkv" "snap_think" "snap_adathink" "tova" "expected_attention" "adasnapkv" "criti_snapkv" "observed_attention")
press_names=("streaming_llm" "snapkv" "snap_think" "snap_adathink" "tova" "observed_attention" "expected_attention" "adasnapkv" "criti_snapkv")
# for press in $press_names
for press in "${press_names[@]}"; 
    do
      bash evaluate.sh llama3.1-8b-inst 1 0.5 ${press} 0,1,2,3,4,5,6,7 0.0 0.0
    done

press_names=("streaming_llm" "snapkv" "snap_think" "snap_adathink" "tova" "observed_attention" "expected_attention" "adasnapkv" "criti_snapkv")
# press_names=("snapkv" "snap_think" "snap_adathink" "expected_attention" "adasnapkv" "criti_snapkv" "tova" "observed_attention")
# for press in $press_names
for press in "${press_names[@]}"; 
  do
    bash evaluate.sh llama3-8b-inst 1 0.5 ${press} 0,1,2,3,4,5,6,7 0.0 0.0
    # echo ${press}
    done
# bash evaluate.sh llama3.1-8b-inst 1 
# dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"

# for dataset in $dataset_list
#   do
#     bash evaluate.sh ${dataset} compress
#     # python eval.py --dataset longbench --data_dir ${dataset} --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snapkv --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output
#   done
# bash test.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.0 0.8 max
# bash test.sh llama3-8b-inst 1 0.5 snap_adathink 1 0.0 0.0 0.9 max gamma
# bash test.sh llama3-8b-inst 1 0.4 snap_adathink 0 0.0 0.0 0.7 normal
# bash test.sh llama3-8b-inst 1 0.5 snap_adathink 1 0.0 0.0 0.6 exp
# 等待所有后台任务完成
wait

# python eval.py --dataset longbench --data_dir narrativeqa --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snap_adathink --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output00