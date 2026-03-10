"""
Positional Encoding Decay Study.

Implements RoPE and ALiBi positional encoding schemes from scratch and
characterizes their theoretical attention decay profiles as a function of
relative position distance. Visualizes the mathematical breakdown points
that explain why effective context length falls short of declared maximums.

Author: Ebenezer Tarubinga, Korea University
Based on: STRING (An et al., ICLR 2025)
"""

import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def compute_rope_rotation_matrix(
    position: int, head_dim: int, base: float = 10000.0
) -> np.ndarray:
    """Compute RoPE rotation angles for a given position.

    Args:
        position: Absolute position index.
        head_dim: Dimension of each attention head (must be even).
        base: Base frequency for RoPE (default 10000.0).

    Returns:
        Array of rotation angles, shape (head_dim // 2,).
    """
    dim_pairs = head_dim // 2
    freqs = 1.0 / (base ** (np.arange(0, dim_pairs) * 2.0 / head_dim))
    angles = position * freqs
    return angles


def rope_attention_score(
    relative_distance: int, head_dim: int, base: float = 10000.0
) -> float:
    """Compute theoretical RoPE attention score between two positions
    separated by `relative_distance`, assuming unit-norm query/key vectors.

    The attention logit contribution from RoPE is:
        sum_i cos(distance * theta_i)
    where theta_i are the per-dimension rotation frequencies.

    Args:
        relative_distance: Distance between query and key positions.
        head_dim: Head dimension.
        base: RoPE base frequency.

    Returns:
        Unnormalized attention score (higher = stronger attention).
    """
    dim_pairs = head_dim // 2
    freqs = 1.0 / (base ** (np.arange(0, dim_pairs) * 2.0 / head_dim))
    angles = relative_distance * freqs
    score = np.cos(angles).sum() / dim_pairs  # Normalize by number of pairs
    return score


def alibi_attention_bias(
    relative_distance: int, num_heads: int, head_index: int
) -> float:
    """Compute ALiBi attention bias for a given head and distance.

    ALiBi adds a linear bias: -m * |distance| where m is head-specific.
    Slopes are set as geometric sequence: m_i = 2^(-8/n * i) for head i.

    Args:
        relative_distance: Distance between query and key positions.
        num_heads: Total number of attention heads.
        head_index: Index of the current head (0-indexed).

    Returns:
        Attention bias (negative value, subtracted from logit).
    """
    ratio = 8.0 / num_heads
    slope = 2.0 ** (-ratio * (head_index + 1))
    return -slope * abs(relative_distance)


def compute_rope_decay_curve(
    max_distance: int, head_dim: int, base: float = 10000.0, step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RoPE attention scores across a range of distances.

    Args:
        max_distance: Maximum relative distance.
        head_dim: Head dimension.
        base: RoPE base frequency.
        step: Step size for distance sampling.

    Returns:
        Tuple of (distances, scores) arrays.
    """
    distances = np.arange(0, max_distance, step)
    scores = np.array(
        [rope_attention_score(d, head_dim, base) for d in distances]
    )
    return distances, scores


def compute_alibi_decay_curve(
    max_distance: int, num_heads: int, head_index: int, step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ALiBi attention biases across a range of distances.

    Args:
        max_distance: Maximum relative distance.
        num_heads: Total number of heads.
        head_index: Which head to compute for.
        step: Step size.

    Returns:
        Tuple of (distances, biases) arrays.
    """
    distances = np.arange(0, max_distance, step)
    biases = np.array(
        [alibi_attention_bias(d, num_heads, head_index) for d in distances]
    )
    return distances, biases


def find_effective_reach(
    distances: np.ndarray, scores: np.ndarray, threshold: float = 0.1
) -> int:
    """Find the distance at which scores drop below `threshold` of the max.

    Args:
        distances: Array of position distances.
        scores: Corresponding attention scores.
        threshold: Fraction of peak score to use as cutoff.

    Returns:
        Distance at which the score first drops below threshold * max_score.
    """
    max_score = scores.max()
    cutoff = threshold * max_score
    below = np.where(scores < cutoff)[0]
    if len(below) == 0:
        return int(distances[-1])
    return int(distances[below[0]])


def plot_rope_angles(head_dim: int, max_pos: int, output_path: str) -> None:
    """Plot RoPE rotation angles vs position for different frequency bands.

    Args:
        head_dim: Head dimension.
        max_pos: Maximum position to plot.
        output_path: Path to save the figure.
    """
    positions = np.arange(0, max_pos)
    dim_pairs = head_dim // 2

    # Select a few representative frequency bands
    band_indices = [0, dim_pairs // 4, dim_pairs // 2, 3 * dim_pairs // 4, dim_pairs - 1]
    freqs = 1.0 / (10000.0 ** (np.arange(0, dim_pairs) * 2.0 / head_dim))

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in band_indices:
        angles = positions * freqs[idx]
        cos_vals = np.cos(angles)
        ax.plot(positions, cos_vals, label=f"dim pair {idx} (freq={freqs[idx]:.2e})",
                alpha=0.8, linewidth=1.2)

    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("cos(position * frequency)", fontsize=12)
    ax.set_title(f"RoPE Rotation Cosines (head_dim={head_dim})", fontsize=14)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RoPE angles plot to {output_path}")


def plot_decay_comparison(
    max_distance: int, head_dims: list, num_heads: int, output_path: str
) -> None:
    """Create multi-panel figure comparing RoPE and ALiBi decay profiles.

    Panel (a): RoPE decay for different head dimensions.
    Panel (b): ALiBi decay for different heads.
    Panel (c): Side-by-side RoPE vs ALiBi normalized comparison.

    Args:
        max_distance: Maximum distance to plot.
        head_dims: List of head dimensions to compare.
        num_heads: Number of attention heads for ALiBi.
        output_path: Path to save the figure.
    """
    step = max(1, max_distance // 2000)  # Limit to ~2000 points for plotting

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # Panel (a): RoPE decay for different head dimensions
    ax1 = fig.add_subplot(gs[0, 0])
    for hd in head_dims:
        distances, scores = compute_rope_decay_curve(max_distance, hd, step=step)
        ax1.plot(distances, scores, label=f"d={hd}", linewidth=1.5)
        reach = find_effective_reach(distances, scores, threshold=0.1)
        ax1.axvline(x=reach, linestyle="--", alpha=0.4,
                    color=ax1.get_lines()[-1].get_color())

    ax1.set_xlabel("Relative Distance", fontsize=11)
    ax1.set_ylabel("Attention Score (normalized)", fontsize=11)
    ax1.set_title("(a) RoPE Decay by Head Dim", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_distance)

    # Panel (b): ALiBi decay for different heads
    ax2 = fig.add_subplot(gs[0, 1])
    head_indices = [0, num_heads // 4, num_heads // 2, num_heads - 1]
    for hi in head_indices:
        distances, biases = compute_alibi_decay_curve(
            max_distance, num_heads, hi, step=step
        )
        # Convert bias to approximate attention weight (softmax-like)
        weights = np.exp(biases)
        weights = weights / weights.max()
        ax2.plot(distances, weights, label=f"head {hi}", linewidth=1.5)

    ax2.set_xlabel("Relative Distance", fontsize=11)
    ax2.set_ylabel("Relative Attention Weight", fontsize=11)
    ax2.set_title(f"(b) ALiBi Decay ({num_heads} heads)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_distance)

    # Panel (c): RoPE vs ALiBi direct comparison
    ax3 = fig.add_subplot(gs[0, 2])
    hd = head_dims[len(head_dims) // 2]  # Use middle head dim
    dist_rope, scores_rope = compute_rope_decay_curve(max_distance, hd, step=step)
    # Normalize to [0, 1]
    scores_rope_norm = (scores_rope - scores_rope.min()) / (
        scores_rope.max() - scores_rope.min() + 1e-12
    )

    dist_alibi, biases_alibi = compute_alibi_decay_curve(
        max_distance, num_heads, num_heads // 2, step=step
    )
    weights_alibi = np.exp(biases_alibi)
    weights_alibi_norm = weights_alibi / weights_alibi.max()

    ax3.plot(dist_rope, scores_rope_norm, label=f"RoPE (d={hd})",
             linewidth=1.5, color="tab:blue")
    ax3.plot(dist_alibi, weights_alibi_norm, label=f"ALiBi (head {num_heads // 2})",
             linewidth=1.5, color="tab:orange")

    # Mark effective reach
    reach_rope = find_effective_reach(dist_rope, scores_rope_norm, 0.1)
    reach_alibi = find_effective_reach(dist_alibi, weights_alibi_norm, 0.1)
    ax3.axvline(x=reach_rope, linestyle="--", color="tab:blue", alpha=0.5,
                label=f"RoPE reach: {reach_rope}")
    ax3.axvline(x=reach_alibi, linestyle="--", color="tab:orange", alpha=0.5,
                label=f"ALiBi reach: {reach_alibi}")

    ax3.set_xlabel("Relative Distance", fontsize=11)
    ax3.set_ylabel("Normalized Score", fontsize=11)
    ax3.set_title("(c) RoPE vs ALiBi", fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_distance)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved decay comparison plot to {output_path}")


def print_breakdown_study(head_dim: int, max_distance: int) -> None:
    """Print study of where positional encoding breaks down.

    Args:
        head_dim: Head dimension.
        max_distance: Maximum distance to examine.
    """
    print("\n" + "=" * 60)
    print(f"POSITIONAL ENCODING BREAKDOWN STUDY (head_dim={head_dim})")
    print("=" * 60)

    distances, scores = compute_rope_decay_curve(max_distance, head_dim, step=1)

    for threshold in [0.5, 0.25, 0.1, 0.05]:
        reach = find_effective_reach(distances, scores, threshold)
        print(f"  RoPE effective reach at {threshold*100:.0f}% threshold: "
              f"{reach:,} positions")

    # Show where the score first goes negative (destructive interference)
    negative_mask = np.where(scores < 0)[0]
    if len(negative_mask) > 0:
        first_negative = distances[negative_mask[0]]
        print(f"  RoPE first negative score at distance: {first_negative:,}")

    # Compute the frequency at which different dimension pairs oscillate
    print(f"\n  Dimension pair frequency examination:")
    dim_pairs = head_dim // 2
    freqs = 1.0 / (10000.0 ** (np.arange(0, dim_pairs) * 2.0 / head_dim))
    print(f"    Fastest frequency (pair 0): period = {2 * np.pi / freqs[0]:.1f}")
    print(f"    Slowest frequency (pair {dim_pairs-1}): "
          f"period = {2 * np.pi / freqs[-1]:.1f}")
    print(f"    Median frequency: period = {2 * np.pi / freqs[dim_pairs // 2]:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate positional encoding decay profiles."
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=128,
        help="Attention head dimension.",
    )
    parser.add_argument(
        "--head_dims",
        type=int,
        nargs="+",
        default=[64, 96, 128],
        help="Head dimensions to compare in decay plot.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="Number of attention heads (for ALiBi).",
    )
    parser.add_argument(
        "--max_distance",
        type=int,
        default=32768,
        help="Maximum relative distance to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory to save figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 1: RoPE rotation angles
    plot_rope_angles(
        head_dim=args.head_dim,
        max_pos=min(args.max_distance, 4096),
        output_path=os.path.join(args.output_dir, "rope_rotation_angles.png"),
    )

    # Plot 2: Decay comparison (RoPE vs ALiBi)
    plot_decay_comparison(
        max_distance=args.max_distance,
        head_dims=args.head_dims,
        num_heads=args.num_heads,
        output_path=os.path.join(args.output_dir, "pe_decay_comparison.png"),
    )

    # Print mathematical breakdown study
    print_breakdown_study(args.head_dim, args.max_distance)


if __name__ == "__main__":
    main()
