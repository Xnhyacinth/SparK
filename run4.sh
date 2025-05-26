
#######################################################################
#            
#                         Group 1
######################################################################
# pip install -U transformers
bash install.sh
# bash run1.sh 0,1,2,3 llama3-70b-inst full_kv 0 0.0
press_names=("streaming_llm" "snapkv" "snap_think" "snap_adathink" "expected_attention" "adasnapkv" "criti_snapkv" "tova" "observed_attention")
press_names=("snapkv" "snap_think" "snap_adathink" "expected_attention" "adasnapkv" "criti_snapkv" "tova" "observed_attention")

press_names=("snap_adathink" "snap_think")
# for press in $press_names
for press in "${press_names[@]}"; 
  do
    # bash evaluate0.sh qwen3-32b 1 0.5 ${press} 0,1 0.0 0.0
    # bash evaluate.sh qwen3-32b 1 0.5 ${press} 0,1 0.0 0.0
   
    bash evaluate0.sh llama3-70b-inst 1 0.5 ${press} 0,1,2,3 0.0 0.0
    bash evaluate0.sh llama3-70b-inst 1 0.6 ${press} 0,1,2,3 0.0 0.0
    bash evaluate0.sh llama3-70b-inst 1 0.4 ${press} 0,1,2,3 0.0 0.0
    bash evaluate0.sh llama3-70b-inst 1 0.7 ${press} 0,1,2,3 0.0 0.0
    bash evaluate0.sh llama3-70b-inst 1 0.8 ${press} 0,1,2,3 0.0 0.0
    bash evaluate0.sh llama3-70b-inst 1 0.9 ${press} 0,1,2,3 0.0 0.0
    bash evaluate0.sh llama3-70b-inst 1 0.3 ${press} 0,1,2,3 0.0 0.0
    bash evaluate0.sh llama3-70b-inst 1 0.2 ${press} 0,1,2,3 0.0 0.0


    bash evaluate.sh llama3-70b-inst 1 0.5 ${press} 0,1,2,3 0.0 0.0
    bash evaluate.sh llama3-70b-inst 1 0.6 ${press} 0,1,2,3 0.0 0.0
    bash evaluate.sh llama3-70b-inst 1 0.4 ${press} 0,1,2,3 0.0 0.0
    bash evaluate.sh llama3-70b-inst 1 0.7 ${press} 0,1,2,3 0.0 0.0
    bash evaluate.sh llama3-70b-inst 1 0.8 ${press} 0,1,2,3 0.0 0.0
    bash evaluate.sh llama3-70b-inst 1 0.9 ${press} 0,1,2,3 0.0 0.0
    bash evaluate.sh llama3-70b-inst 1 0.3 ${press} 0,1,2,3 0.0 0.0
    bash evaluate.sh llama3-70b-inst 1 0.2 ${press} 0,1,2,3 0.0 0.0


    # bash evaluate0.sh qwen3-32b 1 0.5 ${press} 0,1 0.0 0.0
    # bash evaluate0.sh qwen3-32b 1 0.6 ${press} 0,1 0.0 0.0
    # bash evaluate0.sh qwen3-32b 1 0.4 ${press} 0,1 0.0 0.0
    # bash evaluate0.sh qwen3-32b 1 0.7 ${press} 0,1 0.0 0.0
    # bash evaluate0.sh qwen3-32b 1 0.8 ${press} 0,1 0.0 0.0
    # bash evaluate0.sh qwen3-32b 1 0.9 ${press} 0,1 0.0 0.0
    # bash evaluate0.sh qwen3-32b 1 0.3 ${press} 0,1 0.0 0.0
    # bash evaluate0.sh qwen3-32b 1 0.2 ${press} 0,1 0.0 0.0


    # bash evaluate.sh qwen3-32b 1 0.5 ${press} 0,1 0.0 0.0
    # bash evaluate.sh qwen3-32b 1 0.6 ${press} 0,1 0.0 0.0
    # bash evaluate.sh qwen3-32b 1 0.4 ${press} 0,1 0.0 0.0
    # bash evaluate.sh qwen3-32b 1 0.7 ${press} 0,1 0.0 0.0
    # bash evaluate.sh qwen3-32b 1 0.8 ${press} 0,1 0.0 0.0
    # bash evaluate.sh qwen3-32b 1 0.9 ${press} 0,1 0.0 0.0
    # bash evaluate.sh qwen3-32b 1 0.3 ${press} 0,1 0.0 0.0
    # bash evaluate.sh qwen3-32b 1 0.2 ${press} 0,1 0.0 0.0


    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.0 0.654
    # bash test0.sh llama3-8b-inst 1 0.6 snap_adathink 0,1 0.0 0.0 0.654
    # bash test0.sh llama3-8b-inst 1 0.4 snap_adathink 0,1 0.0 0.0 0.654
    # bash test0.sh llama3-8b-inst 1 0.3 snap_adathink 0,1 0.0 0.0 0.654
    # bash test0.sh llama3-8b-inst 1 0.7 snap_adathink 0,1 0.0 0.0 0.654
    # bash test0.sh llama3-8b-inst 1 0.8 snap_adathink 0,1 0.0 0.0 0.654

    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.0 0.654
    # bash test1.sh llama3-8b-inst 1 0.6 snap_adathink 0,1 0.0 0.0 0.654
    # bash test1.sh llama3-8b-inst 1 0.4 snap_adathink 0,1 0.0 0.0 0.654
    # bash test1.sh llama3-8b-inst 1 0.3 snap_adathink 0,1 0.0 0.0 0.654
    # bash test1.sh llama3-8b-inst 1 0.7 snap_adathink 0,1 0.0 0.0 0.654
    # bash test1.sh llama3-8b-inst 1 0.8 snap_adathink 0,1 0.0 0.0 0.654

    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.994 0.994

    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.994 0.994

    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.995 0.995

    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1 0.0 0.995 0.995
    # echo ${press}
    done

# for press in $press_names
# for press in "${press_names[@]}"; 
#   do
#     bash evaluate.sh llama3-8b-inst 0 0.5 ${press} 0,1,2,3 0.0 0.0
#     done
# bash run1.sh 0,1 llama3.1-8b-inst full_kv 1 &
# bash run1.sh 2,3 llama3-8b-inst full_kv 1 &
# bash run2.sh 4,5 llama3.1-8b-inst snap_adathink 1 0.5 0 &
# bash run2.sh 6,7 llama3-8b-inst snap_adathink 1 0.5 0 &
# bash test.sh llama3-8b-inst 1 0.5 snap_adathink 0 0.0 0.99 0.99
wait
# bash run1.sh 0 llama3-8b-inst full_kv 1 0.5 0
# python eval.py --dataset longbench --data_dir narrativeqa --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snap_adathink --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output00