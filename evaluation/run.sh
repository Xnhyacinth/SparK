

# dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum lcc repobench-p"

# for dataset in $dataset_list
#   do
#     # bash evaluate.sh ${dataset}
#     python eval.py --dataset longbench --data_dir ${dataset} --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name snapkv --compression_ratio 0.25 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output
#   done

  # python eval.py --dataset longbench --data_dir narrativeqa --model /modelopsnas/modelops/models/meta-llama/Meta-Llama-3.1-8B-Instruct --press_name think --compression_ratio 0.25 --key_channel_compression_ratio 0.5 --device "cuda:0" --save_dir /modelopsnas/modelops/468440/kvpress/output00 --fraction 0.1
# coms=(0.4 0.3 0.2)
coms=(0.7)
ratios=(128 512 1024 2048)
for com in "${coms[@]}"
  do 
  for ratio in "${ratios[@]}"
    do
    echo ${com}
    echo ${ratio}
    python metric.py --results_dir /modelopsnas/modelops/468440/kvpress/output000_0.7554_0.0/results/Meta-Llama-3-8B-Instruct --compress_ratio ${ratio} --compress_q com --tem 0.0 --com_channel ${com} --pooling_ratio "0.7554" #--threshold_ratio 0.99 # --pooling_ratio "0.755"
    done 
  done
# /modelopsnas/modelops/468440/kvpress/output/results/Meta-Llama-3-8B-Instruct/0.0

# python me.py --results_dir /modelopsnas/modelops/468440/kvpress/output_norm/results/Meta-Llama-3-8B-Instruct --compress_ratio 128 --compress_q com --pooling_ratio "0.5" &

# python me.py --results_dir /modelopsnas/modelops/468440/kvpress/output000_0.6/results/Meta-Llama-3-8B-Instruct --compress_ratio 1024 --compress_q com --pooling_ratio "0.6"  &
# python me.py --results_dir /modelopsnas/modelops/468440/kvpress/output000_0.7/results/Meta-Llama-3-8B-Instruct --compress_ratio 128 --compress_q com --pooling_ratio "0.7"
# wait