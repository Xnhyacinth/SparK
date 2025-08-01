# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import random
import string
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

import torch
from fire import Fire
from tqdm import tqdm
from transformers import pipeline

from kvpress import (
    AdaKVPress,
    ChunkKVPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    QFilterPress,
    PyramidKVPress,
    FinchPress,
)

logger = logging.getLogger(__name__)

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
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "full_kv": ExpectedAttentionPress(0.0),
    "pyramidkv": PyramidKVPress(),
    "finch": FinchPress(),
}


class NeedleInAHaystackTester:
    """
    A comprehensive testing framework for evaluating model performance on retrieval tasks
    across various context lengths and needle positions (Needle in a Haystack test).
    """
    
    def __init__(self):
        self.results = []
        
    def generate_random_text(self, length: int) -> str:
        """Generate random text of specified length."""
        words = []
        current_length = 0
        
        # Common English words for more realistic text
        common_words = [
            "the", "and", "to", "of", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on", 
            "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from", "or", "one",
            "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can", 
            "said", "there", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other"
        ]
        
        while current_length < length:
            word = random.choice(common_words)
            if current_length + len(word) + 1 <= length:  # +1 for space
                words.append(word)
                current_length += len(word) + 1
            else:
                # Add remaining characters
                remaining = length - current_length
                if remaining > 0:
                    words.append(''.join(random.choices(string.ascii_lowercase, k=remaining)))
                break
                
        return ' '.join(words)
    
    def insert_needle(self, haystack: str, needle: str, position: float) -> str:
        """Insert needle at specified relative position in haystack."""
        haystack_length = len(haystack)
        insertion_point = int(haystack_length * position)
        
        return haystack[:insertion_point] + " " + needle + " " + haystack[insertion_point:]
    
    def generate_test_data(
        self, 
        context_lengths: List[int],
        needle_positions: List[float],
        num_needles: int = 5,
        needle_template: str = "The special magic number is {number}."
    ) -> List[Dict[str, Any]]:
        """Generate test data for needle in haystack tests."""
        test_data = []
        
        for context_length in context_lengths:
            for position in needle_positions:
                for i in range(num_needles):
                    # Generate a unique number for this needle
                    magic_number = random.randint(10000, 99999)
                    needle = needle_template.format(number=magic_number)
                    
                    # Generate haystack text
                    # Account for needle length in haystack generation
                    haystack_length = context_length - len(needle) - 20  # Buffer for spaces
                    haystack = self.generate_random_text(haystack_length)
                    
                    # Insert needle
                    context = self.insert_needle(haystack, needle, position)
                    
                    # Generate question
                    question = "What is the special magic number mentioned in the text?"
                    
                    test_case = {
                        "context": context,
                        "question": question,
                        "needle": needle,
                        "magic_number": str(magic_number),
                        "context_length": len(context),
                        "target_length": context_length,
                        "needle_position": position,
                        "answer_prefix": "The special magic number is ",
                        "expected_answer": str(magic_number),
                        "max_new_tokens": 10
                    }
                    test_data.append(test_case)
        
        return test_data
    
    def evaluate_response(self, predicted_answer: str, expected_answer: str) -> Dict[str, Any]:
        """Evaluate the model's response against the expected answer."""
        # Clean up the predicted answer
        predicted_clean = predicted_answer.strip().lower()
        expected_clean = expected_answer.strip().lower()
        
        # Exact match
        exact_match = predicted_clean == expected_clean
        
        # Contains the number
        contains_number = expected_clean in predicted_clean
        
        # Extract numbers from the response
        import re
        predicted_numbers = re.findall(r'\d+', predicted_answer)
        expected_numbers = re.findall(r'\d+', expected_answer)
        
        number_match = len(predicted_numbers) > 0 and predicted_numbers[0] == expected_numbers[0] if expected_numbers else False
        
        return {
            "exact_match": exact_match,
            "contains_number": contains_number,
            "number_match": number_match,
            "predicted_answer": predicted_answer,
            "expected_answer": expected_answer
        }
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for the test results."""
        if not results:
            return {}
        
        total_tests = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        contains_matches = sum(1 for r in results if r["contains_number"])
        number_matches = sum(1 for r in results if r["number_match"])
        
        # Group by context length
        by_length = {}
        for result in results:
            length = result["context_length"]
            if length not in by_length:
                by_length[length] = []
            by_length[length].append(result)
        
        length_metrics = {}
        for length, length_results in by_length.items():
            length_total = len(length_results)
            length_metrics[f"length_{length}"] = {
                "total": length_total,
                "exact_match_rate": sum(1 for r in length_results if r["exact_match"]) / length_total,
                "contains_rate": sum(1 for r in length_results if r["contains_number"]) / length_total,
                "number_match_rate": sum(1 for r in length_results if r["number_match"]) / length_total,
            }
        
        # Group by needle position
        by_position = {}
        for result in results:
            position = result["needle_position"]
            if position not in by_position:
                by_position[position] = []
            by_position[position].append(result)
        
        position_metrics = {}
        for position, position_results in by_position.items():
            position_total = len(position_results)
            position_metrics[f"position_{position}"] = {
                "total": position_total,
                "exact_match_rate": sum(1 for r in position_results if r["exact_match"]) / position_total,
                "contains_rate": sum(1 for r in position_results if r["contains_number"]) / position_total,
                "number_match_rate": sum(1 for r in position_results if r["number_match"]) / position_total,
            }
        
        return {
            "overall": {
                "total_tests": total_tests,
                "exact_match_rate": exact_matches / total_tests,
                "contains_rate": contains_matches / total_tests,
                "number_match_rate": number_matches / total_tests,
            },
            "by_length": length_metrics,
            "by_position": position_metrics,
        }


def needle_in_haystack_test(
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device: Optional[str] = None,
    press_name: str = "expected_attention",
    compression_ratio: float = 0.1,
    context_lengths: Optional[List[int]] = None,
    needle_positions: Optional[List[float]] = None,
    num_needles: int = 5,
    max_new_tokens: int = 10,
    needle_template: str = "The special magic number is {number}.",
    save_results: bool = True,
):
    """
    Run Needle in a Haystack test to evaluate model's ability to retrieve information
    from long contexts at various positions.
    
    Parameters
    ----------
    model : str
        Model to test
    device : str, optional
        Device to run on
    press_name : str
        Compression method to use
    compression_ratio : float
        Compression ratio for KV cache
    context_lengths : List[int], optional
        List of context lengths to test
    needle_positions : List[float], optional
        List of relative positions (0.0 to 1.0) to place the needle
    num_needles : int
        Number of different needles to test per context length/position combination
    max_new_tokens : int
        Maximum tokens to generate
    needle_template : str
        Template for the needle text
    save_results : bool
        Whether to save results to file
    """
    
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if context_lengths is None:
        context_lengths = [1000, 2000, 4000, 8000, 16000, 32000]
    
    if needle_positions is None:
        needle_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Initialize tester
    tester = NeedleInAHaystackTester()
    
    # Generate test data
    print("Generating test data...")
    test_data = tester.generate_test_data(
        context_lengths=context_lengths,
        needle_positions=needle_positions,
        num_needles=num_needles,
        needle_template=needle_template
    )
    
    print(f"Generated {len(test_data)} test cases")
    
    # Load press
    assert press_name in PRESS_DICT
    press = PRESS_DICT[press_name]
    
    if isinstance(press, (DuoAttentionPress)):
        press.head_compression_ratio = compression_ratio
    else:
        press.compression_ratio = compression_ratio
    
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
    
    # Run tests
    print("Running needle in haystack tests...")
    results = []
    
    for test_case in tqdm(test_data, desc="Testing"):
        try:
            output = pipe(
                test_case["context"],
                questions=[test_case["question"]],
                answer_prefix=test_case["answer_prefix"],
                press=press,
                max_new_tokens=max_new_tokens,
            )
            
            predicted_answer = output["answers"][0] if output["answers"] else ""
            
            # Evaluate response
            evaluation = tester.evaluate_response(predicted_answer, test_case["expected_answer"])
            
            # Combine test case info with evaluation results
            result = {**test_case, **evaluation}
            results.append(result)
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error processing test case: {e}")
            result = {
                **test_case,
                "exact_match": False,
                "contains_number": False,
                "number_match": False,
                "predicted_answer": f"ERROR: {str(e)}",
                "error": str(e)
            }
            results.append(result)
    
    # Calculate metrics
    metrics = tester.calculate_metrics(results)
    
    # Save results
    if save_results:
        save_dir = Path(__file__).parent / "needle_results"
        save_dir.mkdir(exist_ok=True)
        
        timestamp = torch.jit.annotate(str, f"{torch.utils.data.get_worker_info()}")
        if hasattr(torch.utils.data, '_utils'):
            timestamp = "manual"
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_filename = save_dir / (
            f"needle_test__{model.replace('/', '--')}__{press_name}__{compression_ratio}__{timestamp}"
        )
        
        # Save detailed results
        with open(f"{save_filename}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save metrics
        with open(f"{save_filename}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Results saved to {save_filename}_*.json")
    
    # Print summary
    print("\n" + "="*50)
    print("NEEDLE IN A HAYSTACK TEST RESULTS")
    print("="*50)
    
    overall = metrics["overall"]
    print(f"Overall Results (Total tests: {overall['total_tests']}):")
    print(f"  Exact Match Rate: {overall['exact_match_rate']:.3f}")
    print(f"  Contains Number Rate: {overall['contains_rate']:.3f}")
    print(f"  Number Match Rate: {overall['number_match_rate']:.3f}")
    
    print(f"\nBy Context Length:")
    for length_key, length_metrics in metrics["by_length"].items():
        length = length_key.replace("length_", "")
        print(f"  {length} tokens: Exact={length_metrics['exact_match_rate']:.3f}, "
              f"Contains={length_metrics['contains_rate']:.3f}, "
              f"Number={length_metrics['number_match_rate']:.3f}")
    
    print(f"\nBy Needle Position:")
    for position_key, position_metrics in metrics["by_position"].items():
        position = position_key.replace("position_", "")
        print(f"  Position {position}: Exact={position_metrics['exact_match_rate']:.3f}, "
              f"Contains={position_metrics['contains_rate']:.3f}, "
              f"Number={position_metrics['number_match_rate']:.3f}")
    
    return results, metrics


if __name__ == "__main__":
    Fire(needle_in_haystack_test)
