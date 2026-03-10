"""
Publication-Quality Attention Heatmap Visualization.

Loads saved attention weights (produced by attention_patterns.py or
supplied as numpy arrays) and creates multi-panel heatmap figures showing
different attention pattern types at various sequence lengths.

Visualizes: local attention, strided patterns, global/sink token patterns,
and how these change as context length increases.

Author: Ebenezer Tarubinga, Korea University
Based on: STRING (An et al., ICLR 2025)
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec


def load_attention_weights(path: str) -> Dict[str, np.ndarray]:
    """Load attention weights from a numpy archive.

    Expected keys in the .npz file:
        - 'attention_layer_{i}': shape (num_heads, seq_len, seq_len)

    Args:
        path: Path to .npz file.

    Returns:
        Dictionary mapping layer names to attention arrays.
    """
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def generate_synthetic_attention(
    seq_length: int,
    num_heads: int = 4,
    pattern_types: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic attention patterns for demonstration.

    Creates attention matrices exhibiting known pattern types for
    visualization when real model outputs are not available.

    Args:
        seq_length: Sequence length.
        num_heads: Number of attention heads.
        pattern_types: List of pattern names to generate. Defaults to
            ["local", "strided", "global", "sink"].
        seed: Random seed.

    Returns:
        Dictionary mapping pattern names to attention arrays of shape
        (seq_len, seq_len).
    """
    rng = np.random.RandomState(seed)
    if pattern_types is None:
        pattern_types = ["local", "strided", "global", "sink"]

    patterns: Dict[str, np.ndarray] = {}

    for ptype in pattern_types:
        attn = np.zeros((seq_length, seq_length), dtype=np.float32)

        if ptype == "local":
            # Each token attends primarily to a local window
            window_size = max(seq_length // 16, 4)
            for i in range(seq_length):
                start = max(0, i - window_size)
                end = min(seq_length, i + 1)  # Causal: only attend to past
                attn[i, start:end] = 1.0
                # Add noise
                attn[i, :end] += rng.exponential(0.05, size=end)

        elif ptype == "strided":
            # Attend to every k-th token (plus local window)
            stride = max(seq_length // 32, 2)
            local_window = max(seq_length // 32, 2)
            for i in range(seq_length):
                # Local window
                start = max(0, i - local_window)
                attn[i, start:i + 1] = 0.5
                # Strided positions
                strided_pos = np.arange(0, i + 1, stride)
                attn[i, strided_pos] += 1.0

        elif ptype == "global":
            # Some tokens attend globally, others locally
            global_tokens = np.sort(rng.choice(seq_length, size=max(2, seq_length // 32), replace=False))
            local_window = max(seq_length // 16, 4)
            for i in range(seq_length):
                # Everyone has local attention
                start = max(0, i - local_window)
                attn[i, start:i + 1] = 0.3
                # Global tokens get high attention from everyone
                past_global = global_tokens[global_tokens <= i]
                if len(past_global) > 0:
                    attn[i, past_global] += 1.0

        elif ptype == "sink":
            # First few tokens act as attention sinks
            num_sinks = max(2, seq_length // 128)
            local_window = max(seq_length // 16, 4)
            for i in range(seq_length):
                # Local attention
                start = max(0, i - local_window)
                attn[i, start:i + 1] = 0.3
                # Sink tokens get disproportionate attention
                attn[i, :num_sinks] += 2.0 * np.exp(-0.5 * np.arange(num_sinks))

        # Apply causal mask
        causal_mask = np.tril(np.ones((seq_length, seq_length)))
        attn = attn * causal_mask

        # Normalize rows to form valid probability distributions
        row_sums = attn.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        attn = attn / row_sums

        patterns[ptype] = attn

    return patterns


def plot_single_heatmap(
    ax: plt.Axes,
    attention: np.ndarray,
    title: str,
    show_colorbar: bool = True,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    """Plot a single attention heatmap on the given axes.

    Args:
        ax: Matplotlib axes.
        attention: 2D attention matrix (seq_len x seq_len).
        title: Title for the subplot.
        show_colorbar: Whether to add a colorbar.
        vmax: Maximum value for color normalization.
        cmap: Colormap name.
    """
    if vmax is None:
        vmax = np.percentile(attention, 99)

    im = ax.imshow(
        attention,
        aspect="auto",
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Key Position", fontsize=9)
    ax.set_ylabel("Query Position", fontsize=9)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)


def plot_attention_pattern_comparison(
    patterns: Dict[str, np.ndarray],
    seq_length: int,
    output_path: str,
) -> None:
    """Create a multi-panel figure comparing different attention patterns.

    Args:
        patterns: Dictionary of {pattern_name: attention_matrix}.
        seq_length: Sequence length (for title).
        output_path: Path to save figure.
    """
    n_patterns = len(patterns)
    fig, axes = plt.subplots(1, n_patterns, figsize=(5 * n_patterns, 4.5))
    if n_patterns == 1:
        axes = [axes]

    for ax, (name, attn) in zip(axes, patterns.items()):
        # For large sequences, subsample for visualization
        display_size = min(256, attn.shape[0])
        step = max(1, attn.shape[0] // display_size)
        attn_sub = attn[::step, ::step]

        plot_single_heatmap(ax, attn_sub, f"{name.capitalize()} Pattern")

    fig.suptitle(
        f"Attention Pattern Types (seq_len={seq_length})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved attention pattern comparison to {output_path}")


def plot_length_comparison(
    seq_lengths: List[int],
    pattern_type: str,
    output_path: str,
    seed: int = 42,
) -> None:
    """Compare the same attention pattern type at different sequence lengths.

    Args:
        seq_lengths: List of sequence lengths to compare.
        pattern_type: Which pattern type to visualize.
        output_path: Path to save figure.
        seed: Random seed.
    """
    n_lengths = len(seq_lengths)
    fig, axes = plt.subplots(1, n_lengths, figsize=(5 * n_lengths, 4.5))
    if n_lengths == 1:
        axes = [axes]

    for ax, sl in zip(axes, seq_lengths):
        patterns = generate_synthetic_attention(sl, pattern_types=[pattern_type], seed=seed)
        attn = patterns[pattern_type]

        # Subsample for display
        display_size = min(256, sl)
        step = max(1, sl // display_size)
        attn_sub = attn[::step, ::step]

        plot_single_heatmap(ax, attn_sub, f"Length = {sl}")

    fig.suptitle(
        f"{pattern_type.capitalize()} Attention at Different Context Lengths",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved length comparison ({pattern_type}) to {output_path}")


def plot_sink_token_study(
    seq_lengths: List[int],
    output_path: str,
    seed: int = 42,
) -> None:
    """Visualize attention sink phenomenon across sequence lengths.

    Shows the fraction of attention allocated to the first few tokens
    as a function of query position and sequence length.

    Args:
        seq_lengths: Sequence lengths to analyze.
        output_path: Path to save figure.
        seed: Random seed.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    num_sinks = 4

    for sl in seq_lengths:
        patterns = generate_synthetic_attention(sl, pattern_types=["sink"], seed=seed)
        attn = patterns["sink"]

        # Attention mass on first `num_sinks` tokens per query position
        sink_mass = attn[:, :num_sinks].sum(axis=-1)
        positions = np.arange(sl) / sl  # Normalize to [0, 1]

        # Subsample for readability
        step = max(1, sl // 200)
        ax1.plot(
            positions[::step], sink_mass[::step],
            label=f"{sl} tokens", alpha=0.8, linewidth=1.2,
        )

    ax1.set_xlabel("Relative Query Position", fontsize=11)
    ax1.set_ylabel(f"Attention on First {num_sinks} Tokens", fontsize=11)
    ax1.set_title("(a) Attention Sink Mass vs. Position", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bar chart: average sink mass by sequence length
    avg_masses = []
    for sl in seq_lengths:
        patterns = generate_synthetic_attention(sl, pattern_types=["sink"], seed=seed)
        attn = patterns["sink"]
        avg_mass = attn[:, :num_sinks].sum(axis=-1).mean()
        avg_masses.append(avg_mass)

    bars = ax2.bar(
        range(len(seq_lengths)), avg_masses,
        tick_label=[f"{sl//1024}K" if sl >= 1024 else str(sl) for sl in seq_lengths],
        color=sns.color_palette("Blues_d", len(seq_lengths)),
        edgecolor="black", linewidth=0.5,
    )
    ax2.set_xlabel("Sequence Length", fontsize=11)
    ax2.set_ylabel(f"Mean Attention on First {num_sinks} Tokens", fontsize=11)
    ax2.set_title("(b) Average Sink Mass by Length", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sink token study to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality attention heatmaps."
    )
    parser.add_argument(
        "--attention_path",
        type=str,
        default=None,
        help="Path to .npz file with saved attention weights. "
             "If not provided, synthetic patterns are generated.",
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Sequence lengths for comparison plots.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic patterns.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })

    if args.attention_path is not None:
        # Load real attention weights
        print(f"Loading attention weights from {args.attention_path}")
        attn_data = load_attention_weights(args.attention_path)
        # Plot each layer's attention
        for key, attn_array in attn_data.items():
            if attn_array.ndim == 3:
                # (num_heads, seq, seq) -- plot head 0
                patterns = {"head_0": attn_array[0]}
            elif attn_array.ndim == 2:
                patterns = {key: attn_array}
            else:
                continue
            plot_attention_pattern_comparison(
                patterns,
                seq_length=attn_array.shape[-1],
                output_path=os.path.join(args.output_dir, f"heatmap_{key}.png"),
            )
    else:
        print("No attention file provided; generating synthetic patterns.")

    # Generate pattern type comparison at a representative length
    ref_length = args.seq_lengths[-1]
    patterns = generate_synthetic_attention(
        ref_length, pattern_types=["local", "strided", "global", "sink"],
        seed=args.seed,
    )
    plot_attention_pattern_comparison(
        patterns, ref_length,
        os.path.join(args.output_dir, "attention_pattern_types.png"),
    )

    # Length comparison for each pattern type
    for ptype in ["local", "sink"]:
        plot_length_comparison(
            args.seq_lengths, ptype,
            os.path.join(args.output_dir, f"length_comparison_{ptype}.png"),
            seed=args.seed,
        )

    # Sink token study
    plot_sink_token_study(
        args.seq_lengths,
        os.path.join(args.output_dir, "sink_token_study.png"),
        seed=args.seed,
    )

    print("\nAll heatmap figures generated successfully.")


if __name__ == "__main__":
    main()
