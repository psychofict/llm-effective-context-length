"""
Effective Context Length Benchmark.

Implements RULER-style benchmark tasks (needle-in-a-haystack, multi-key
retrieval, variable tracking) and evaluates LLMs across multiple context
lengths to quantify the gap between declared and effective context length.

The "effective context length" is defined as the maximum length at which
task accuracy remains above 90% of the short-context (4K) baseline.

Author: Ebenezer Tarubinga, Korea University
Based on: STRING (An et al., ICLR 2025)
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Task Generators
# ---------------------------------------------------------------------------

def generate_needle_in_haystack(
    tokenizer: AutoTokenizer,
    context_length: int,
    needle_depth: float = 0.5,
    seed: int = 42,
) -> Tuple[str, str]:
    """Generate a needle-in-a-haystack retrieval task.

    Places a unique fact inside a long passage of filler text and asks the
    model to retrieve it.

    Args:
        tokenizer: Tokenizer for measuring token counts.
        context_length: Target total context length in tokens.
        needle_depth: Relative position of the needle (0.0 = start, 1.0 = end).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (prompt, expected_answer).
    """
    rng = random.Random(seed)

    # The needle -- a unique, verifiable fact
    city = rng.choice(["Berlin", "Tokyo", "Cairo", "Lima", "Oslo"])
    number = rng.randint(1000, 9999)
    needle = f"The secret code for {city} is {number}."
    expected_answer = str(number)

    # Filler sentences
    filler_topics = [
        "The weather patterns across different regions vary significantly throughout the year.",
        "Many researchers have studied the impact of urbanization on local ecosystems.",
        "International trade agreements have shaped economic policies for decades.",
        "Advances in renewable energy technology continue to accelerate globally.",
        "The history of mathematics spans thousands of years and many civilizations.",
        "Modern transportation networks connect cities across vast distances.",
        "Agricultural practices have evolved dramatically since the industrial revolution.",
        "Ocean currents play a critical role in regulating the global climate system.",
        "The development of new materials has enabled advances in engineering.",
        "Cultural exchanges between nations have enriched societies throughout history.",
    ]

    # Build the filler to reach target length
    question = f"\n\nQuestion: What is the secret code for {city}?\nAnswer:"
    question_tokens = len(tokenizer.encode(question))
    needle_tokens = len(tokenizer.encode(needle))
    available_tokens = context_length - question_tokens - needle_tokens - 10

    filler_parts: List[str] = []
    current_tokens = 0
    while current_tokens < available_tokens:
        sentence = rng.choice(filler_topics)
        sentence_tokens = len(tokenizer.encode(sentence))
        if current_tokens + sentence_tokens > available_tokens:
            break
        filler_parts.append(sentence)
        current_tokens += sentence_tokens

    # Insert needle at specified depth
    insert_pos = max(0, int(len(filler_parts) * needle_depth))
    filler_parts.insert(insert_pos, needle)

    prompt = " ".join(filler_parts) + question
    return prompt, expected_answer


def generate_multi_key_retrieval(
    tokenizer: AutoTokenizer,
    context_length: int,
    num_keys: int = 5,
    seed: int = 42,
) -> Tuple[str, str]:
    """Generate a multi-key retrieval task.

    Places multiple key-value pairs in the context and asks about one of them.

    Args:
        tokenizer: Tokenizer.
        context_length: Target context length.
        num_keys: Number of key-value pairs to place.
        seed: Random seed.

    Returns:
        Tuple of (prompt, expected_answer).
    """
    rng = random.Random(seed)

    # Generate key-value pairs
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "silver", "gold"]
    objects = ["box", "envelope", "folder", "container", "jar", "chest", "bag", "case"]
    values = [rng.randint(100, 999) for _ in range(num_keys)]

    selected_colors = rng.sample(colors, num_keys)
    selected_objects = rng.sample(objects, num_keys)

    kv_pairs = []
    for i in range(num_keys):
        kv_pairs.append(
            f"The {selected_colors[i]} {selected_objects[i]} contains the number {values[i]}."
        )

    # Pick one to ask about
    query_idx = rng.randint(0, num_keys - 1)
    question = (
        f"\n\nQuestion: What number is in the {selected_colors[query_idx]} "
        f"{selected_objects[query_idx]}?\nAnswer:"
    )
    expected_answer = str(values[query_idx])

    # Fill remaining context with filler
    filler_sentences = [
        "This document contains various pieces of information for reference.",
        "Please read through the entire text carefully before answering.",
        "Some details may appear more relevant than others upon close inspection.",
        "The information is spread throughout the passage at various locations.",
        "Each piece of data has been placed deliberately within this text.",
    ]

    question_tokens = len(tokenizer.encode(question))
    kv_tokens = sum(len(tokenizer.encode(kv)) for kv in kv_pairs)
    available_tokens = context_length - question_tokens - kv_tokens - 10

    filler_parts: List[str] = []
    current_tokens = 0
    while current_tokens < available_tokens:
        sentence = rng.choice(filler_sentences)
        stokens = len(tokenizer.encode(sentence))
        if current_tokens + stokens > available_tokens:
            break
        filler_parts.append(sentence)
        current_tokens += stokens

    # Distribute KV pairs evenly through the filler
    combined = list(filler_parts)
    for i, kv in enumerate(kv_pairs):
        pos = int(len(combined) * (i + 1) / (num_keys + 1))
        combined.insert(pos, kv)

    prompt = " ".join(combined) + question
    return prompt, expected_answer


def generate_variable_tracking(
    tokenizer: AutoTokenizer,
    context_length: int,
    num_variables: int = 3,
    num_updates: int = 8,
    seed: int = 42,
) -> Tuple[str, str]:
    """Generate a variable tracking task.

    Assigns values to variables, updates them multiple times, and asks for
    the final value. Requires tracking state across the context.

    Args:
        tokenizer: Tokenizer.
        context_length: Target context length.
        num_variables: Number of variables to track.
        num_updates: Total number of update operations.
        seed: Random seed.

    Returns:
        Tuple of (prompt, expected_answer).
    """
    rng = random.Random(seed)

    var_names = [f"var_{chr(65 + i)}" for i in range(num_variables)]  # var_A, var_B...
    state = {name: rng.randint(0, 99) for name in var_names}

    operations: List[str] = []
    # Initial assignments
    for name, val in state.items():
        operations.append(f"Set {name} = {val}.")

    # Random updates
    for _ in range(num_updates):
        target = rng.choice(var_names)
        op = rng.choice(["set", "add", "subtract"])
        operand = rng.randint(1, 50)
        if op == "set":
            state[target] = operand
            operations.append(f"Set {target} = {operand}.")
        elif op == "add":
            state[target] += operand
            operations.append(f"Add {operand} to {target}. (Now {target} = {state[target]})")
        else:
            state[target] -= operand
            operations.append(f"Subtract {operand} from {target}. (Now {target} = {state[target]})")

    query_var = rng.choice(var_names)
    question = f"\n\nQuestion: What is the final value of {query_var}?\nAnswer:"
    expected_answer = str(state[query_var])

    # Pad with filler to reach target length
    filler = "Additional context information is provided for reference. "
    question_tokens = len(tokenizer.encode(question))
    ops_text = " ".join(operations)
    ops_tokens = len(tokenizer.encode(ops_text))
    available_tokens = context_length - question_tokens - ops_tokens - 10

    filler_tokens = len(tokenizer.encode(filler))
    num_filler = max(0, available_tokens // filler_tokens)

    # Interleave filler with operations
    all_parts: List[str] = []
    filler_per_op = num_filler // (len(operations) + 1)
    for op in operations:
        for _ in range(filler_per_op):
            all_parts.append(filler)
        all_parts.append(op)
    # Remaining filler at the end
    remaining = num_filler - filler_per_op * len(operations)
    for _ in range(remaining):
        all_parts.append(filler)

    prompt = " ".join(all_parts) + question
    return prompt, expected_answer


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASK_GENERATORS = {
    "needle": generate_needle_in_haystack,
    "multi_key": generate_multi_key_retrieval,
    "variable_tracking": generate_variable_tracking,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    expected_answer: str,
    device: torch.device,
    max_new_tokens: int = 20,
) -> bool:
    """Evaluate a single task instance.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        prompt: Full task prompt.
        expected_answer: Ground-truth answer string.
        device: Torch device.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        True if the expected answer appears in the generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    # Decode only the newly generated tokens
    generated = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return expected_answer.strip() in generated.strip()


def run_benchmark(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task_name: str,
    context_lengths: List[int],
    num_samples: int,
    device: torch.device,
) -> Dict[int, float]:
    """Run a benchmark task across multiple context lengths.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        task_name: Name of the task (key in TASK_GENERATORS).
        context_lengths: List of context lengths to evaluate.
        num_samples: Number of samples per context length.
        device: Torch device.

    Returns:
        Dictionary mapping context_length to accuracy (0.0 - 1.0).
    """
    generator = TASK_GENERATORS[task_name]
    results: Dict[int, float] = {}

    for ctx_len in tqdm(context_lengths, desc=f"Task: {task_name}"):
        correct = 0
        total = 0
        for sample_idx in range(num_samples):
            seed = ctx_len * 1000 + sample_idx
            try:
                prompt, expected = generator(tokenizer, ctx_len, seed=seed)
                is_correct = evaluate_task(
                    model, tokenizer, prompt, expected, device
                )
                correct += int(is_correct)
                total += 1
            except Exception as e:
                print(f"  Error at length {ctx_len}, sample {sample_idx}: {e}")
                continue

        accuracy = correct / max(total, 1)
        results[ctx_len] = accuracy
        print(f"  {task_name} @ {ctx_len}: {accuracy:.1%} ({correct}/{total})")

    return results


def compute_effective_context_length(
    results: Dict[int, float],
    baseline_length: int = 4096,
    threshold_ratio: float = 0.9,
) -> Optional[int]:
    """Compute effective context length from benchmark results.

    Args:
        results: Dictionary mapping context_length to accuracy.
        baseline_length: Short-context baseline length.
        threshold_ratio: Fraction of baseline accuracy to use as cutoff.

    Returns:
        Maximum context length where accuracy >= threshold_ratio * baseline,
        or None if baseline length not in results.
    """
    if baseline_length not in results:
        # Use the smallest available length as baseline
        baseline_length = min(results.keys())

    baseline_acc = results[baseline_length]
    threshold = threshold_ratio * baseline_acc

    effective_length = baseline_length
    for ctx_len in sorted(results.keys()):
        if results[ctx_len] >= threshold:
            effective_length = ctx_len

    return effective_length


def save_results(
    all_results: Dict[str, Dict[int, float]],
    effective_lengths: Dict[str, int],
    model_name: str,
    output_dir: str,
) -> None:
    """Save benchmark results to JSON and print a summary table.

    Args:
        all_results: Nested dict of {task: {length: accuracy}}.
        effective_lengths: Dict of {task: effective_length}.
        model_name: Name of the model.
        output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "model": model_name,
        "results": {
            task: {str(k): v for k, v in lengths.items()}
            for task, lengths in all_results.items()
        },
        "effective_lengths": effective_lengths,
    }

    output_path = os.path.join(
        output_dir, f"{model_name.split('/')[-1]}_benchmark.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY: {model_name}")
    print("=" * 70)

    all_lengths = sorted(
        set(l for r in all_results.values() for l in r.keys())
    )
    header = f"{'Task':<20}" + "".join(f"{l:>8}" for l in all_lengths) + f"{'Effective':>12}"
    print(header)
    print("-" * len(header))

    for task in sorted(all_results.keys()):
        row = f"{task:<20}"
        for l in all_lengths:
            acc = all_results[task].get(l, float("nan"))
            row += f"{acc:>7.1%} "
        row += f"{effective_lengths.get(task, 'N/A'):>12}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RULER-style effective context length benchmarks."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["needle", "multi_key", "variable_tracking"],
        choices=list(TASK_GENERATORS.keys()),
        help="Benchmark tasks to run.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768],
        help="Context lengths to evaluate.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples per context length per task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/tables",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'.",
    )
    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto" if args.device == "auto" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Run benchmarks
    all_results: Dict[str, Dict[int, float]] = {}
    effective_lengths: Dict[str, int] = {}

    for task in args.tasks:
        print(f"\n--- Running task: {task} ---")
        results = run_benchmark(
            model, tokenizer, task, args.lengths, args.num_samples, device
        )
        all_results[task] = results

        eff_len = compute_effective_context_length(results)
        effective_lengths[task] = eff_len
        print(f"  Effective context length for {task}: {eff_len}")

    # Save and display results
    save_results(all_results, effective_lengths, args.model_name, args.output_dir)


if __name__ == "__main__":
    main()
