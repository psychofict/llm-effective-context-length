"""
Attention Pattern Analysis for Long-Context LLMs.

Extracts and analyzes attention weight distributions across varying sequence
lengths to characterize how attention entropy and effective span degrade as
context grows. This provides empirical evidence for the gap between declared
and effective context length.

Author: Ebenezer Tarubinga, Korea University
Based on: STRING (An et al., ICLR 2025)
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy of attention distributions.

    Args:
        attention_weights: Attention probabilities of shape
            (batch, num_heads, seq_len, seq_len).

    Returns:
        Entropy per head, shape (batch, num_heads, seq_len).
    """
    # Clamp to avoid log(0)
    attn = attention_weights.clamp(min=1e-12)
    entropy = -(attn * attn.log()).sum(dim=-1)
    return entropy


def compute_effective_attention_span(
    attention_weights: torch.Tensor, mass_threshold: float = 0.9
) -> torch.Tensor:
    """Compute effective attention span -- the smallest window covering
    `mass_threshold` fraction of total attention mass.

    Args:
        attention_weights: Attention probabilities of shape
            (batch, num_heads, seq_len, seq_len).
        mass_threshold: Fraction of attention mass to capture (default 0.9).

    Returns:
        Effective span per query position, shape (batch, num_heads, seq_len).
    """
    # Sort attention weights in descending order along the key dimension
    sorted_attn, _ = attention_weights.sort(dim=-1, descending=True)
    cumulative = sorted_attn.cumsum(dim=-1)
    # Find the first index where cumulative mass exceeds the threshold
    above_threshold = (cumulative >= mass_threshold).float()
    # argmax returns the index of the first True value
    span = above_threshold.argmax(dim=-1) + 1
    return span.float()


def extract_attention_weights(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    seq_length: int,
    layers: List[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Generate a random input and extract attention weights from specified layers.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        seq_length: Number of tokens in the input sequence.
        layers: List of layer indices to extract attention from.
        device: Torch device.

    Returns:
        Dictionary mapping layer index to attention weight tensor.
    """
    # Use random token ids from the vocabulary (excluding special tokens)
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(
        low=100, high=vocab_size - 100, size=(1, seq_length), device=device
    )

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            use_cache=False,
        )

    attention_weights = {}
    all_attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
    for layer_idx in layers:
        if layer_idx < len(all_attentions):
            attention_weights[layer_idx] = all_attentions[layer_idx].cpu()

    return attention_weights


def analyze_sequence_length(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    seq_length: int,
    layers: List[int],
    device: torch.device,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Analyze attention patterns for a single sequence length.

    Returns:
        Tuple of (entropy_by_layer, span_by_layer) where each is a dict
        mapping layer index to the mean value across heads and positions.
    """
    attn_weights = extract_attention_weights(
        model, tokenizer, seq_length, layers, device
    )

    entropy_by_layer = {}
    span_by_layer = {}

    for layer_idx, attn in attn_weights.items():
        entropy = compute_attention_entropy(attn)
        mean_entropy = entropy.mean().item()
        entropy_by_layer[layer_idx] = mean_entropy

        span = compute_effective_attention_span(attn, mass_threshold=0.9)
        mean_span = span.mean().item()
        span_by_layer[layer_idx] = mean_span

    return entropy_by_layer, span_by_layer


def plot_entropy_vs_length(
    results: Dict[int, Dict[int, float]],
    layers: List[int],
    output_path: str,
) -> None:
    """Plot attention entropy as a function of sequence length.

    Args:
        results: Nested dict of {seq_length: {layer_idx: mean_entropy}}.
        layers: Layer indices to include.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    seq_lengths = sorted(results.keys())

    for layer_idx in layers:
        entropies = [results[sl].get(layer_idx, 0.0) for sl in seq_lengths]
        ax.plot(seq_lengths, entropies, marker="o", label=f"Layer {layer_idx}")

    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("Mean Attention Entropy (nats)", fontsize=12)
    ax.set_title("Attention Entropy vs. Sequence Length", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved entropy plot to {output_path}")


def plot_span_vs_layer(
    results: Dict[int, Dict[int, float]],
    layers: List[int],
    output_path: str,
) -> None:
    """Plot effective attention span as a function of layer depth.

    Args:
        results: Nested dict of {seq_length: {layer_idx: mean_span}}.
        layers: Layer indices to include.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    seq_lengths = sorted(results.keys())

    for sl in seq_lengths:
        spans = [results[sl].get(layer_idx, 0.0) for layer_idx in layers]
        ax.plot(layers, spans, marker="s", label=f"{sl} tokens")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Effective Attention Span (90% mass)", fontsize=12)
    ax.set_title("Effective Attention Span vs. Layer Depth", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved span plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze attention patterns across sequence lengths."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length to test.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--layers_to_analyze",
        type=int,
        nargs="+",
        default=[0, 4, 8, 16, 24, 31],
        help="Layer indices to extract attention from.",
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

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device if args.device != "auto" else "auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Need explicit attention weights
    )
    model.eval()

    # Sequence lengths to test (powers of 2, up to max_length)
    seq_lengths = []
    length = 1024
    while length <= args.max_length:
        seq_lengths.append(length)
        length *= 2

    # Run analysis
    entropy_results: Dict[int, Dict[int, float]] = {}
    span_results: Dict[int, Dict[int, float]] = {}

    for sl in tqdm(seq_lengths, desc="Analyzing sequence lengths"):
        print(f"\nProcessing sequence length: {sl}")
        try:
            entropy_by_layer, span_by_layer = analyze_sequence_length(
                model, tokenizer, sl, args.layers_to_analyze, device
            )
            entropy_results[sl] = entropy_by_layer
            span_results[sl] = span_by_layer
        except RuntimeError as e:
            print(f"  Skipped length {sl}: {e}")
            break

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate plots
    model_short = args.model_name.split("/")[-1]
    plot_entropy_vs_length(
        entropy_results,
        args.layers_to_analyze,
        os.path.join(args.output_dir, f"{model_short}_entropy_vs_length.png"),
    )
    plot_span_vs_layer(
        span_results,
        args.layers_to_analyze,
        os.path.join(args.output_dir, f"{model_short}_span_vs_layer.png"),
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for sl in sorted(entropy_results.keys()):
        mean_ent = np.mean(list(entropy_results[sl].values()))
        mean_span = np.mean(list(span_results[sl].values()))
        print(f"  Length {sl:>6d}: mean entropy = {mean_ent:.3f}, "
              f"mean span = {mean_span:.1f}")


if __name__ == "__main__":
    main()
