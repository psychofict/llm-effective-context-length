"""
Positional Bias Visualization.

Visualizes position-dependent biases in LLM attention, including the
"lost in the middle" phenomenon where information placed at intermediate
positions in the context receives less attention than information near
the beginning or end.

Creates:
  - Retrieval accuracy as a function of needle position (U-shaped curve)
  - Heatmaps of attention-to-position across layers
  - Position bias comparison across models

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


# ---------------------------------------------------------------------------
# Simulated positional bias data
# ---------------------------------------------------------------------------

def simulate_lost_in_middle(
    context_length: int,
    num_positions: int = 20,
    primacy_strength: float = 0.15,
    recency_strength: float = 0.20,
    base_accuracy: float = 0.85,
    noise_std: float = 0.02,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the "lost in the middle" retrieval accuracy curve.

    Models the U-shaped curve where accuracy is highest when the target
    information is near the beginning (primacy) or end (recency) of the
    context, and lowest in the middle.

    Args:
        context_length: Total context length in tokens.
        num_positions: Number of evenly spaced positions to test.
        primacy_strength: Boost to accuracy at the start.
        recency_strength: Boost to accuracy at the end.
        base_accuracy: Minimum accuracy at the center.
        noise_std: Standard deviation of Gaussian noise.
        seed: Random seed.

    Returns:
        Tuple of (relative_positions, accuracies) arrays.
    """
    rng = np.random.RandomState(seed)
    positions = np.linspace(0, 1, num_positions)

    # U-shaped curve: quadratic in distance from center
    center = 0.5
    distance_from_center = np.abs(positions - center) / center  # 0 at center, 1 at edges

    # Asymmetric U: primacy and recency have different strengths
    primacy_mask = positions < center
    recency_mask = positions >= center

    accuracy = np.full(num_positions, base_accuracy)
    accuracy[primacy_mask] += primacy_strength * (distance_from_center[primacy_mask] ** 1.5)
    accuracy[recency_mask] += recency_strength * (distance_from_center[recency_mask] ** 1.5)

    # Add noise
    accuracy += rng.normal(0, noise_std, num_positions)
    accuracy = np.clip(accuracy, 0, 1)

    return positions, accuracy


def simulate_attention_to_position(
    num_layers: int,
    context_length: int,
    num_positions: int = 64,
    sink_strength: float = 0.3,
    local_bias: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """Simulate average attention weight as a function of key position
    across layers.

    Produces a (num_layers, num_positions) matrix showing how much attention
    each layer allocates to each position on average.

    Args:
        num_layers: Number of transformer layers.
        context_length: Context length being simulated.
        num_positions: Number of position bins.
        sink_strength: Strength of attention sink at position 0.
        local_bias: Bias toward recent positions.
        seed: Random seed.

    Returns:
        Attention heatmap of shape (num_layers, num_positions).
    """
    rng = np.random.RandomState(seed)
    positions = np.linspace(0, 1, num_positions)

    attn_map = np.zeros((num_layers, num_positions))

    for layer in range(num_layers):
        layer_frac = layer / max(num_layers - 1, 1)

        # Base: uniform attention
        base = np.ones(num_positions) / num_positions

        # Sink effect (stronger in early layers)
        sink = np.zeros(num_positions)
        sink_width = max(1, num_positions // 16)
        sink[:sink_width] = sink_strength * (1.0 - 0.5 * layer_frac)
        sink[:sink_width] *= np.exp(-np.arange(sink_width) * 0.5)

        # Local/recency bias (stronger in later layers)
        recency = np.zeros(num_positions)
        recency_window = max(1, num_positions // 8)
        recency[-recency_window:] = local_bias * layer_frac
        recency[-recency_window:] *= np.linspace(0.3, 1.0, recency_window)

        # Middle suppression (grows with depth)
        middle = np.zeros(num_positions)
        mid_start = num_positions // 4
        mid_end = 3 * num_positions // 4
        middle[mid_start:mid_end] = -0.15 * layer_frac

        combined = base + sink + recency + middle
        combined += rng.normal(0, 0.01, num_positions)
        combined = np.maximum(combined, 0)
        combined /= combined.sum() + 1e-12

        attn_map[layer] = combined

    return attn_map


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_lost_in_middle(
    model_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
) -> None:
    """Plot the lost-in-the-middle phenomenon for multiple models.

    Args:
        model_curves: {model_name: (positions, accuracies)}.
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("husl", len(model_curves))

    for (model_name, (positions, accuracies)), color in zip(
        model_curves.items(), palette
    ):
        ax.plot(
            positions * 100, accuracies * 100,
            marker="o", markersize=5, linewidth=2,
            color=color, label=model_name,
        )

    ax.set_xlabel("Needle Position (% of context)", fontsize=12)
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=12)
    ax.set_title("Lost in the Middle: Retrieval Accuracy vs. Position",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # Annotate regions
    ax.annotate(
        "Primacy\nregion", xy=(10, ax.get_ylim()[1] * 0.92),
        fontsize=9, ha="center", color="gray", fontstyle="italic",
    )
    ax.annotate(
        "Lost in the\nmiddle", xy=(50, ax.get_ylim()[0] + 5),
        fontsize=9, ha="center", color="gray", fontstyle="italic",
    )
    ax.annotate(
        "Recency\nregion", xy=(90, ax.get_ylim()[1] * 0.92),
        fontsize=9, ha="center", color="gray", fontstyle="italic",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lost-in-the-middle plot to {output_path}")


def plot_attention_position_heatmap(
    attn_map: np.ndarray,
    model_name: str,
    output_path: str,
) -> None:
    """Plot attention-to-position heatmap across layers.

    Args:
        attn_map: Array of shape (num_layers, num_positions).
        model_name: Name of the model (for title).
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        attn_map,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="bilinear",
        extent=[0, 100, attn_map.shape[0] - 0.5, -0.5],
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average Attention Weight", fontsize=10)

    ax.set_xlabel("Key Position (% of context)", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    ax.set_title(
        f"Attention Distribution Across Layers -- {model_name}",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved position heatmap for {model_name} to {output_path}")


def plot_multi_model_position_heatmaps(
    model_configs: Dict[str, Dict],
    output_path: str,
    seed: int = 42,
) -> None:
    """Create a multi-panel figure comparing position bias across models.

    Args:
        model_configs: {model_name: {num_layers, context_length, ...}}.
        output_path: Path to save figure.
        seed: Random seed.
    """
    n_models = len(model_configs)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, cfg) in zip(axes, model_configs.items()):
        num_layers = cfg.get("num_layers", 32)
        context_length = cfg.get("context_length", 4096)
        sink_strength = cfg.get("sink_strength", 0.3)

        attn_map = simulate_attention_to_position(
            num_layers=num_layers,
            context_length=context_length,
            sink_strength=sink_strength,
            seed=seed,
        )

        im = ax.imshow(
            attn_map,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="bilinear",
            extent=[0, 100, num_layers - 0.5, -0.5],
        )
        ax.set_xlabel("Key Position (%)", fontsize=10)
        ax.set_ylabel("Layer", fontsize=10)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Position-Dependent Attention Bias Across Models",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-model position heatmaps to {output_path}")


def plot_position_bias_by_length(
    context_lengths: List[int],
    num_layers: int,
    output_path: str,
    seed: int = 42,
) -> None:
    """Show how position bias changes with context length.

    For each context length, computes the average attention at each
    position (averaged across layers) and overlays the curves.

    Args:
        context_lengths: List of context lengths to compare.
        num_layers: Number of model layers.
        output_path: Path to save figure.
        seed: Random seed.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("viridis", len(context_lengths))
    num_positions = 64

    for cl, color in zip(context_lengths, palette):
        attn_map = simulate_attention_to_position(
            num_layers=num_layers,
            context_length=cl,
            num_positions=num_positions,
            seed=seed,
        )
        # Average across layers
        avg_attn = attn_map.mean(axis=0)
        positions = np.linspace(0, 100, num_positions)

        label = f"{cl // 1024}K" if cl >= 1024 else str(cl)
        ax.plot(positions, avg_attn, linewidth=2, color=color, label=label)

    ax.set_xlabel("Key Position (% of context)", fontsize=12)
    ax.set_ylabel("Average Attention Weight", fontsize=12)
    ax.set_title("Position Bias at Different Context Lengths",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Context Length", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved position bias by length to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize positional biases in LLM attention."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768],
        help="Context lengths to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
    })

    # --- Plot 1: Lost in the Middle ---
    model_curves = {}
    model_params = {
        "Llama-2-7B": {"primacy": 0.12, "recency": 0.18, "base": 0.82},
        "Llama-3-8B": {"primacy": 0.10, "recency": 0.15, "base": 0.88},
        "Mistral-7B": {"primacy": 0.11, "recency": 0.16, "base": 0.85},
        "Qwen2-7B": {"primacy": 0.09, "recency": 0.14, "base": 0.89},
    }
    for model_name, params in model_params.items():
        positions, accuracies = simulate_lost_in_middle(
            context_length=8192,
            primacy_strength=params["primacy"],
            recency_strength=params["recency"],
            base_accuracy=params["base"],
            seed=args.seed + hash(model_name) % 1000,
        )
        model_curves[model_name] = (positions, accuracies)

    plot_lost_in_middle(
        model_curves,
        os.path.join(args.output_dir, "lost_in_the_middle.png"),
    )

    # --- Plot 2: Single model attention position heatmap ---
    attn_map = simulate_attention_to_position(
        num_layers=32, context_length=8192, seed=args.seed,
    )
    plot_attention_position_heatmap(
        attn_map, "Llama-2-7B (8K context)",
        os.path.join(args.output_dir, "attention_position_heatmap.png"),
    )

    # --- Plot 3: Multi-model comparison ---
    model_configs = {
        "Llama-2-7B": {"num_layers": 32, "context_length": 4096, "sink_strength": 0.35},
        "Llama-3-8B": {"num_layers": 32, "context_length": 8192, "sink_strength": 0.28},
        "Mistral-7B": {"num_layers": 32, "context_length": 32768, "sink_strength": 0.30},
        "Qwen2-7B": {"num_layers": 28, "context_length": 32768, "sink_strength": 0.25},
    }
    plot_multi_model_position_heatmaps(
        model_configs,
        os.path.join(args.output_dir, "multi_model_position_bias.png"),
        seed=args.seed,
    )

    # --- Plot 4: Position bias by context length ---
    plot_position_bias_by_length(
        args.context_lengths,
        num_layers=32,
        output_path=os.path.join(args.output_dir, "position_bias_by_length.png"),
        seed=args.seed,
    )

    print("\nAll positional bias figures generated successfully.")


if __name__ == "__main__":
    main()
