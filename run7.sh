press_names=("snap_adathink")
# for press in $press_names
for press in "${press_names[@]}"; 
  do

    # bash test0.sh llama3.1-8b-inst 1 0.5 ${press} 2,3 0.0 0.0 0.65 no ruler
    # bash test0.sh llama3-8b-inst 1 0.5 snap_adathink 2 0.0 0.0 0.65 no longbench 0.5
    # bash test0.sh llama3.1-8b-inst 1 0.6 ${press} 2,3 0.0 0.0 0.65 no ruler
    bash test0.sh llama3.1-8b-inst 1 0.4 ${press} 2,3 0.0 0.0 0.65 no ruler
    bash test0.sh llama3.1-8b-inst 1 0.7 ${press} 2,3 0.0 0.0 0.65 no ruler
    bash test0.sh llama3.1-8b-inst 1 0.8 ${press} 2,3 0.0 0.0 0.65 no ruler
    bash test0.sh llama3.1-8b-inst 1 0.9 ${press} 2,3 0.0 0.0 0.65 no ruler
    bash test0.sh llama3.1-8b-inst 1 0.3 ${press} 2,3 0.0 0.0 0.65 no ruler

done

wait