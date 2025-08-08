press_names=("snap_spark" "pyramid_spark")

for press in "${press_names[@]}"; 
  do
    # spark for channel pruning ratio 0.5
    bash evaluate_spark.sh llama3.1-8b-inst 1 0.5 ${press} 0,1 0.0 0.0 0.65 no 

    # spark for channel pruning ratio 0.8
    bash evaluate_spark.sh llama3.1-8b-inst 1 0.8 ${press} 0,1 0.0 0.0 0.65 no 


done

wait