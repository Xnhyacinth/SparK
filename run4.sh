press_names=("snapkv" "pyramidkv" "streaming_llm" "tova" "observed_attention" "expected_attention" "pyramid_quark" "snap_quark" "pyramid_think" "snap_think")
for press in "${press_names[@]}"; do
  bash evaluate.sh llama3.1-8b-inst 1 0.5 ${press} 2,3 0.0 0.0 0.0 no

done
wait