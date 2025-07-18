press_names=("pyramid_adathink")
compression_ratios=(0.5 0.8 0.6 0.7 0.9 0.4 0.3)
# press_names=("snap_think" "snap_adathink")
# press_names=("snap_adathink")
# for press in $press_names
for ratio in "${compression_ratios[@]}"; do
    for press in "${press_names[@]}"; do
      echo "Running evaluation for press: ${press} with ratio: ${ratio}"
      bash test0.sh llama3.1-8b-inst 1 ${ratio} ${press} 2,3 0.0 0.0 0.65
    done
done