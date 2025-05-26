
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

bash install.sh
# bash run1.sh 0,1,2,3,2,3 llama3-8b-inst full_kv 0 0.0
press_names=("streaming_llm" "snapkv" "snap_think" "snap_adathink" "expected_attention" "adasnapkv" "criti_snapkv" "tova" "observed_attention")
press_names=("snapkv" "snap_think" "snap_adathink" "expected_attention" "adasnapkv" "criti_snapkv" "tova" "observed_attention")

press_names=("snap_adathink")
# for press in $press_names
for press in "${press_names[@]}"; 
  do
    # bash test0.sh llama3-8b-inst 1 0.4 snap_adathink 0,1,2,3 0.0 0.0 0.7555
    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.0 0.7555
    # bash test0.sh llama3-8b-inst 1 0.6 snap_adathink 0,1,2,3 0.0 0.0 0.7555
    # bash test0.sh llama3-8b-inst 1 0.7 snap_adathink 0,1,2,3 0.0 0.0 0.7555
    # bash test0.sh llama3-8b-inst 1 0.8 snap_adathink 0,1,2,3 0.0 0.0 0.7555

    # bash test1.sh llama3-8b-inst 1 0.4 snap_adathink 0,1,2,3 0.0 0.0 0.7555
    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.0 0.7555
    # bash test1.sh llama3-8b-inst 1 0.6 snap_adathink 0,1,2,3 0.0 0.0 0.7555
    # bash test1.sh llama3-8b-inst 1 0.7 snap_adathink 0,1,2,3 0.0 0.0 0.7555
    # bash test1.sh llama3-8b-inst 1 0.8 snap_adathink 0,1,2,3 0.0 0.0 0.7555

    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.9922 

    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.9922 

    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.993 

    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.993 


    bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 4,5,6,7 0.0 0.992

    bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.992

    bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.99 0.99

    bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.99 0.99

    bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.993

    bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.993

    bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.993 0.993

    bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.993 0.993

    # echo ${press}
    # bash test1.sh llama3-8b-inst 1 0.5 snap_adathink 0,1,2,3 0.0 0.0 0.755
    done

# for press in $press_names
# for press in "${press_names[@]}"; 
#   do
#     bash evaluate.sh llama3-8b-inst 0 0.5 ${press} 0,1,2,3,2,3 0.0 0.0
#     done
# bash run1.sh 0,1,2,3 llama3-8b-inst full_kv 1 &
# bash run1.sh 2,3 llama3-8b-inst full_kv 1 &
# bash run2.sh 4,5 llama3-8b-inst snap_adathink 1 0.5 0 &
# bash run2.sh 6,7 llama3-8b-inst snap_adathink 1 0.5 0 &
wait
# bash run1.sh 0 llama3-8b-inst full_kv 1 0.5 0
# python eval.py --dataset longbench --data_dir narrativeqa --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snap_adathink --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output00