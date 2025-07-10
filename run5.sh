
# pip install transformers==4.51.0
# bash install.sh
# bash evaluate.sh llama3-8b-inst 1
# bash run1.sh 4,5 llama3.1-8b-inst full_kv 1 0.0 ruler
press_names=("snapkv" "pyramidkv" "streaming_llm" "tova" "observed_attention" "expected_attention")
press_names=("streaming_llm" "tova" "observed_attention" "expected_attention")
# press_names=("snap_think" "snap_adathink")
# press_names=("snap_adathink")
# for press in $press_names
for press in "${press_names[@]}"; do
  bash evaluate1.sh llama3.1-8b-inst 1 0.5 ${press} 4,5 0.0 0.0 0.0 no ruler

done

wait

# python eval.py --dataset longbench --data_dir narrativeqa --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snap_adathink --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output00