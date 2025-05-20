# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
from typing import Optional
import os
from numpy import ndarray
import torch
from datasets import load_dataset, Dataset
from fire import Fire
from infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from loogle.calculate_metrics import calculate_metrics as loogle_scorer
from ruler.calculate_metrics import calculate_metrics as ruler_scorer
from tqdm import tqdm
from transformers import pipeline
from zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer
from longbench.evaluate import scorer
import sys

from kvpress import (
    CriticalKVPress,
    CriticalAdaKVPress,
    AdaKVPress,
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    DuoAttentionPress,
    ComposedPress,
    AdaThinKPress
)

logger = logging.getLogger(__name__)

DATASET_DICT = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench"
}

SCORER_DICT = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": scorer
}

PRESS_DICT = {
    "criti_adasnapkv": CriticalAdaKVPress(SnapKVPress()),
    "criti_ada_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "criti_snapkv": CriticalKVPress(SnapKVPress()),
    "criti_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "adasnapkv": AdaKVPress(SnapKVPress()),
    "ada_expected_attention": AdaKVPress(ExpectedAttentionPress()),
    "expected_attention": ExpectedAttentionPress(),
    "ada_expected_attention_e2": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "random": RandomPress(),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "duo_attention": DuoAttentionPress(),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "full_kv": ExpectedAttentionPress(0.0),
    "snap_adathink": ComposedPress([SnapKVPress(), AdaThinKPress()]),
}


def evaluate(
    dataset: str,
    data_dir: Optional[str] = None,
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device: Optional[str] = None,
    press_name: str = "expected_attention",
    compression_ratio: float = 0.1,
    key_channel_compression_ratio: float = 0.0,
    fraction: float = 1.0,
    max_new_tokens: Optional[int] = None,
    max_context_length: Optional[int] = None,
    compress_questions: bool = False,
    save_dir: Optional[str] = None,
    max_capacity_prompt: Optional[int] = None,
    threshold_ratio: float = 0.0,
    temperature: float = 0.0,
    pooling_ratio: float = 0.0,
    mode: Optional[str] = None,
):
    """
    Evaluate a model on a dataset using a press and save the results

    Parameters
    ----------
    dataset : str
        Dataset to evaluate
    data_dir : str, optional
        Subdirectory of the dataset to evaluate, by default None
    model : str, optional
        Model to use, by default "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device : str, optional
        Model device, by default cuda:0 if available else cpu. For multi-GPU use "auto"
    press_name : str, optional
        Press to use (see PRESS_DICT), by default "expected_attention"
    compression_ratio : float, optional
        Compression ratio for the press, by default 0.1
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default use the default for the task (recommended)
    fraction : float, optional
        Fraction of the dataset to evaluate, by default 1.0
    max_context_length : int, optional
        Maximum number of tokens to use in the context. By default will use the maximum length supported by the model.
    compress_questions : bool, optional
        Whether to compress the questions as well, by default False
    """

    assert dataset in DATASET_DICT, f"No dataset found for {dataset}"
    assert dataset in SCORER_DICT, f"No scorer found for {dataset}"
    data_dir = str(data_dir) if data_dir else None

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ratio = str(compression_ratio) if max_capacity_prompt is None else str(max_capacity_prompt)
    save_prefix =  Path(save_dir)
    save_dir = save_prefix / "results" / model.split('/')[-1] / str(temperature) / ratio / dataset / data_dir 
    # save_dir.mkdir(exist_ok=False)
    # os.makedirs(str(save_dir), exist_ok=True)
    save_filename = save_dir / (
       press_name
        + ".json"
    )
    
    # Load dataframe
    # df = load_dataset(DATASET_DICT[dataset], data_dir=data_dir, split="test").to_pandas()
    df = load_dataset(DATASET_DICT[dataset], data_dir=data_dir, split="test").to_pandas()
    # ds = load_dataset("Xnhyacinth/LongBench", data_dir, split="test", cache_dir=DATASET_DICT[dataset])
    # ds = Dataset.from_pandas(df)
    # breakpoint()

    if compress_questions:
        df["context"] = df["context"] + df["question"]
        df["question"] = ""
        save_dir = save_prefix / "results" / model.split('/')[-1] / "comress_questions" / str(temperature) / ratio / dataset / data_dir
       
        save_filename = save_dir / (
            press_name
                + ".json"
            )
        # save_filename = save_filename.with_name(save_filename.stem + "__compressed_questions" + save_filename.suffix)

    os.makedirs(str(save_dir), exist_ok=True)
    if save_filename.exists():
        logger.warning(f"Results already exist at {save_filename}")
        os.remove(save_filename)
        
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=42)
        save_filename = save_filename.with_name(save_filename.stem + f"__fraction{fraction:.2f}" + save_filename.suffix)
    
    model2maxlen = json.load(open("longbench/config/model2maxlen.json", "r"))
    max_context_length = model2maxlen[model.split('/')[-1]]
    
    if max_context_length is not None:
        save_filename = save_filename.with_name(
            save_filename.stem + f"__max_context{max_context_length}" + save_filename.suffix
        )

    # Load press
    assert press_name in PRESS_DICT
    press = PRESS_DICT[press_name]

    if isinstance(press, (DuoAttentionPress)):
        press.head_compression_ratio = compression_ratio
    elif isinstance(press, (ComposedPress)):
        for ps in press.presses:
            if isinstance(ps, (ThinKPress)):
                ps.key_channel_compression_ratio = key_channel_compression_ratio
                save_filename = save_filename.with_name(
                    save_filename.stem + f"__channel{key_channel_compression_ratio}" + save_filename.suffix
                )
            elif isinstance(ps, (AdaThinKPress)):
                if threshold_ratio != 0:
                    ps.threshold_ratio = threshold_ratio
                    save_filename = save_filename.with_name(
                        save_filename.stem + f"__threshold{threshold_ratio}" + save_filename.suffix
                    )
                elif pooling_ratio != 0:
                    ps.pooling_ratio = pooling_ratio
                    ps.mode = mode
                    save_filename = save_filename.with_name(
                        save_filename.stem + f"__{mode}{pooling_ratio}" + save_filename.suffix
                    )
                # else:
                ps.key_channel_compression_ratio = key_channel_compression_ratio
                ps.outpath = save_dir
                save_filename = save_filename.with_name(
                    save_filename.stem + f"__channel{key_channel_compression_ratio}" + save_filename.suffix
                )
                
            else:
                ps.compression_ratio = compression_ratio
            ps.max_capacity_prompt = max_capacity_prompt
    elif isinstance(press, (ThinKPress)) or isinstance(press, (AdaThinKPress)):
        press.key_channel_compression_ratio = key_channel_compression_ratio
        press.max_capacity_prompt = max_capacity_prompt
    else:
        press.compression_ratio = compression_ratio  # type:ignore[attr-defined]
        press.max_capacity_prompt = max_capacity_prompt

    if os.path.exists(save_filename): 
        print(f"{save_filename} exist! exit!")
        sys.exit()  # 退出程序

    # Initialize pipeline with the correct attention implementation
    model_kwargs = {"torch_dtype": "auto"}
    if isinstance(press, ObservedAttentionPress):
        model_kwargs["attn_implementation"] = "eager"
    else:
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    if device == "auto":
        pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", model_kwargs=model_kwargs)
    else:
        pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

    if data_dir in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        pipe.tokenizer.chat_template = None
        pipe.tokenizer.bos_token = ""
        if data_dir in ["samsum"]:
            pipe.model.generation_config.eos_token_id = [pipe.tokenizer.eos_token_id, pipe.tokenizer.encode("\n", add_special_tokens=False)[-1]]

    # Run pipeline on each context
    df["predicted_answer"] = None
    ds = Dataset.from_pandas(df)
    # breakpoint()
    df_context = df.groupby("context")
    assert all(df_context["answer_prefix"].nunique() == 1)

    predictions, answers = [], []
    for context, df_ in tqdm(df_context, total=df["context"].nunique()):
        questions = df_["question"].to_list()
        lengths = df_["length"].to_list()
        max_new_tokens_ = max_new_tokens if max_new_tokens is not None else df_["max_new_tokens"].iloc[0]
        answer_prefix = df_["answer_prefix"].iloc[0]
        output = pipe(
            context,
            questions=questions,
            answer_prefix=answer_prefix,
            press=press,
            max_new_tokens=max_new_tokens_,
            max_context_length=max_context_length,
            temperature=temperature,
        )
        df.loc[df_.index, "predicted_answer"] = output["answers"]
        df.loc[df_.index, "compression_ratio"] = press.compression_ratio
        predictions.extend(output["answers"])
        golds = [arr.tolist() for arr in df_["answers"].to_list()]
        answers.extend(golds)
        torch.cuda.empty_cache()
        
        # breakpoint()
        
    # Save answers
    # ds = Dataset.from_pandas(df)
    # df[["predicted_answer", "compression_ratio"]].to_csv(str(save_filename), index=False)
        # breakpoint()
    # Calculate metrics
    scorer = SCORER_DICT[dataset]
    if dataset == "longbench":
        metrics = scorer(data_dir, predictions, answers, ds[0]["all_classes"])
    else:
        metrics = scorer(df)
    
    df = df.rename(columns={"predicted_answer": "pred"})
    df = df[['pred', 'answers', 'all_classes', 'length']]
    # 将 DataFrame 转换为 JSONL 文件
    with open(save_filename, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json_obj = row.to_dict()  # 转换为字典格式
            for k, v in json_obj.items():
                # breakpoint()
                json_obj[k] = v.tolist() if isinstance(v, ndarray) else v
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    # with open(save_filename, "a", encoding="utf-8") as f:
    #     for pred, answer, length in zip(predictions, answers, lengths):
    #         json.dump({"pred": pred, "answers": answer, "all_classes": df_["all_classes"].iloc[0].tolist() if df_["all_classes"].iloc[0] is not None else df_["all_classes"].iloc[0], "length": length}, f, ensure_ascii=False)
    #         f.write('\n')
    with open(f"{'/'.join(str(save_filename).split('/')[:-1])}/res.json", "a") as f:
        json.dump(f'{press_name}_{key_channel_compression_ratio}: {metrics}', f, indent=4)
        f.write('\n')

    
    # print(f"Average compression ratio: {df['compression_ratio'].mean():.2f}")
    print(metrics)


if __name__ == "__main__":
    Fire(evaluate)
