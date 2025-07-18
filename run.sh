

# dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum lcc repobench-p"

coms=(0.5)
ratios=(128 512 1024 2048)
# ratios=(128)
xxx=0.992
for com in "${coms[@]}"
  do 
  for ratio in "${ratios[@]}"
    do
    echo ${com}
    echo ${ratio}
    # python metric.py --results_dir output000_0.0_${xxx}/results/Meta-Llama-3-8B-Instruct --compress_ratio ${ratio} --compress_q com --tem 0.0 --com_channel ${com} --threshold_ratio ${xxx} #--pooling_ratio ${xxx}
    python metri.py --results_dir output000_0.65_0.0/results/Meta-Llama-3-8B-Instruct --compress_ratio ${ratio} --compress_q com --tem 0.0 --com_channel ${com} --threshold_ratio ${xxx} #--pooling_ratio ${xxx}
    done 
  done