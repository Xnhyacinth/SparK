press_names=("snap_quark" "pyramid_quark")
# for press in $press_names
for press in "${press_names[@]}"; 
  do

    bash test1.sh llama3.1-8b-inst 1 0.5 ${press} 0,1 0.0 0.0 0.65 no 
    # bash test0.sh llama3.1-8b-inst 1 0.6 ${press} 0,1 0.0 0.0 0.65 no 
    # bash test0.sh llama3.1-8b-inst 1 0.4 ${press} 0,1 0.0 0.0 0.65 no 
    # bash test0.sh llama3.1-8b-inst 1 0.7 ${press} 0,1 0.0 0.0 0.65 no 
    bash test1.sh llama3.1-8b-inst 1 0.8 ${press} 0,1 0.0 0.0 0.65 no 
    # bash test0.sh llama3.1-8b-inst 1 0.9 ${press} 0,1 0.0 0.0 0.65 no 
    # bash test0.sh llama3.1-8b-inst 1 0.3 ${press} 0,1 0.0 0.0 0.65 no 

done

wait