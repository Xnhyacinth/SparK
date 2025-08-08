


## Installation

```bash
git clone https://github.com/Xnhyacinth/SparK.git
cd SparK
poetry install --with dev
```

## Usage

```bash
press_names=("snapkv" "pyramidkv" "streaming_llm" "tova" "observed_attention" "expected_attention" "pyramid_spark" "snap_spark" "pyramid_think" "snap_think")


model=${1:-"llama3.1-8b-inst"} # model name
compress_questions=${2:-"0"} # compress questions default 1
key_channel_compression_ratio=${3:-"0.5"}
press=${4:-"snapkv"} # compress methods
gpus=${5:-"0"} # gpus
temperature=${6:-"0.0"} 
threshold_ratio=${7:-"0.0"}
pooling_ratio=${8:-"0.0"}

# threshold_ratio choices: 0.0 0.99 0.992 0.996 0.998 0.997...   control dynamic group and topp
# pooling_ratio choices: 0.0 0.65 0.655 0.75...   control recover method  6* is exp and 7* is norm

bash run2.sh # spark recover with avg
bash run4.sh # all baselines with think and spark
```

The specific parameters can be found in the method's implementation in [spark_press](kvpress/presses/spark_press.py).

For more methods (press), see `PRESS_DICT` in [eval.py](eval.py).

### Citation

If you find SparK or this project is helpful, please kindly consider cite our paper 😊.

```bibtex

```

### Acknowledgement

We would like to express our sincere appreciation to [kvpress](https://github.com/NVIDIA/kvpress) for their invaluable open-source contributions, which have substantially accelerated the progress and development of this project.