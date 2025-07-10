press_names=("snap_adathink")
# for press in $press_names
for press in "${press_names[@]}"; 
  do

    # bash test0.sh llama3.1-8b-inst 1 0.5 ${press} 6,7 0.0 0.99 0.0 no ruler
    # bash test0.sh llama3.1-8b-inst 1 0.5 ${press} 6,7 0.0 0.99 0.99 no ruler

    bash test0.sh llama3.1-8b-inst 1 0.5 ${press} 6,7 0.0 0.996 0.0 no ruler
    bash test0.sh llama3.1-8b-inst 1 0.5 ${press} 6,7 0.0 0.996 0.996 no ruler

    bash test0.sh llama3.1-8b-inst 1 0.5 ${press} 6,7 0.0 0.992 0.0 no ruler
    bash test0.sh llama3.1-8b-inst 1 0.5 ${press} 6,7 0.0 0.992 0.992 no ruler

    bash test0.sh llama3.1-8b-inst 1 0.5 ${press} 6,7 0.0 0.997 0.0 no ruler
    bash test0.sh llama3.1-8b-inst 1 0.5 ${press} 6,7 0.0 0.997 0.997 no ruler

done

wait