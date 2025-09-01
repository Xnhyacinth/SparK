
# SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning

[![Paper](https://img.shields.io/badge/cs.CL-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2508.15212)
[![Static Badge](https://img.shields.io/badge/LongBench-🤗-blue)](https://huggingface.co/datasets/Xnhyacinth/LongBench)
[![Static Badge](https://img.shields.io/badge/LongBenchv2-🤗-blue)](https://huggingface.co/datasets/Xnhyacinth/LongBench-v2)
[![Static Badge](https://img.shields.io/badge/Ruler-🤗-green)](https://huggingface.co/datasets/simonjegou/ruler)

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
@article{liao2025spark,
  title={SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning},
  author={Liao, Huanxuan and Xu, Yixing and He, Shizhu and Li, Guanchen and Yin, Xuanwu and Li, Dong and Barsoum, Emad and Zhao, Jun and Liu, Kang},
  journal={arXiv preprint arXiv:2508.15212},
  year={2025}
}
```

### Acknowledgement

We would like to express our sincere appreciation to [kvpress](https://github.com/NVIDIA/kvpress) for their invaluable open-source contributions, which have substantially accelerated the progress and development of this project.