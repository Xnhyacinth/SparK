


## Install

```bash
bash preinstall.sh
bash insatll.sh
```

## Usage

```bash
press_names=("snapkv" "pyramidkv" "streaming_llm" "tova" "observed_attention" "expected_attention")


model=${1:-"llama3.1-8b-inst"}
compress_questions=${2:-"0"}
key_channel_compression_ratio=${3:-"0.5"}
press=${4:-"snapkv"}
gpus=${5:-"0"}
temperature=${6:-"0.0"}
threshold_ratio=${7:-"0.0"}
pooling_ratio=${8:-"0.0"}

# threshold_ratio choices: 0.0 0.99 0.992 0.996 0.998 0.997...   control dynamic group and topp
# pooling_ratio choices: 0.0 0.65 0.655 0.75...   control recover method  6* is exp and 7* is norm

bash run2.sh # adathink recover with avg
bash run4.sh # all baselines with think and adathink
```

The specific parameters can be found in the method's implementation in `kvpress0/presses/adathink_press.py`