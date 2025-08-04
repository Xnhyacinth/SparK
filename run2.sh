press_names=("snap_quark" "pyramid_quark")

for press in "${press_names[@]}"; 
  do
    # quark for channel pruning ratio 0.5
    bash evaluate_quark.sh llama3.1-8b-inst 1 0.5 ${press} 0,1 0.0 0.0 0.65 no 

    # quark for channel pruning ratio 0.8
    bash evaluate_quark.sh llama3.1-8b-inst 1 0.8 ${press} 0,1 0.0 0.0 0.65 no 


done

wait