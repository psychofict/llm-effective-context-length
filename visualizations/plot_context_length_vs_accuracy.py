"""
Context Length vs. Accuracy Visualization.

Loads benchmark results (from effective_length_benchmark.py) and creates
publication-quality line plots showing how model accuracy degrades with
increasing context length. Marks the "effective context length" and
highlights the gap between declared and effective maximum.

Author: Ebenezer Tarubinga, Korea University
Based on: STRING (An et al., ICLR 2025)
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Default data (used when benchmark JSON files are not available)
# ---------------------------------------------------------------------------

DEFAULT_RESULTS: Dict[str, Dict[str, Dict[int, float]]] = {
    "Llama-2-7B": {
        "needle": {4096: 0.92, 8192: 0.78, 16384: 0.54, 32768: 0.31},
        "multi_key": {4096: 0.88, 8192: 0.71, 16384: 0.48, 32768: 0.25},
        "variable_tracking": {4096: 0.87, 8192: 0.68, 16384: 0.52, 32768: 0.30},
    },
    "Llama-3-8B": {
        "needle": {4096: 0.95, 8192: 0.92, 16384: 0.79, 32768: 0.58},
        "multi_key": {4096: 0.92, 8192: 0.87, 16384: 0.72, 32768: 0.49},
        "variable_tracking": {4096: 0.92, 8192: 0.86, 16384: 0.72, 32768: 0.51},
    },
    "Mistral-7B": {
        "needle": {4096: 0.94, 8192: 0.88, 16384: 0.73, 32768: 0.47},
        "multi_key": {4096: 0.90, 8192: 0.83, 16384: 0.66, 32768: 0.40},
        "variable_tracking": {4096: 0.91, 8192: 0.83, 16384: 0.67, 32768: 0.42},
    },
    "Qwen2-7B": {
        "needle": {4096: 0.96, 8192: 0.93, 16384: 0.84, 32768: 0.66},
        "multi_key": {4096: 0.93, 8192: 0.89, 16384: 0.78, 32768: 0.58},
        "variable_tracking": {4096: 0.93, 8192: 0.89, 16384: 0.77, 32768: 0.60},
    },
}

DECLARED_MAX_LENGTHS: Dict[str, int] = {
    "Llama-2-7B": 4096,
    "Llama-3-8B": 8192,
    "Mistral-7B": 32768,
    "Qwen2-7B": 32768,
}


def load_benchmark_results(results_dir: str) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Load benchmark results from JSON files.

    Args:
        results_dir: Directory containing *_benchmark.json files.

    Returns:
        Nested dict: {model_name: {task: {length: accuracy}}}.
    """
    all_results = {}
    if not os.path.isdir(results_dir):
        return all_results

    for filename in os.listdir(results_dir):
        if filename.endswith("_benchmark.json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            model_name = data.get("model", filename.replace("_benchmark.json", ""))
            results = {}
            for task, length_acc in data.get("results", {}).items():
                results[task] = {int(k): v for k, v in length_acc.items()}
            all_results[model_name] = results

    return all_results


def compute_effective_length(
    task_results: Dict[int, float],
    threshold_ratio: float = 0.9,
) -> Optional[int]:
    """Compute effective context length for a single task.

    Args:
        task_results: {context_length: accuracy}.
        threshold_ratio: Fraction of baseline accuracy as cutoff.

    Returns:
        Effective context length, or None if data is empty.
    """
    if not task_results:
        return None

    baseline_length = min(task_results.keys())
    baseline_acc = task_results[baseline_length]
    threshold = threshold_ratio * baseline_acc

    effective = baseline_length
    for length in sorted(task_results.keys()):
        if task_results[length] >= threshold:
            effective = length
    return effective


def plot_accuracy_vs_length(
    all_results: Dict[str, Dict[str, Dict[int, float]]],
    output_path: str,
    threshold_ratio: float = 0.9,
) -> None:
    """Create multi-panel line plots: context length vs accuracy per model.

    Each panel is a model; each line within is a task. Effective context
    length is marked with a vertical dashed line.

    Args:
        all_results: Nested results dictionary.
        output_path: Path to save figure.
        threshold_ratio: Threshold ratio for effective length computation.
    """
    models = sorted(all_results.keys())
    n_models = len(models)
    ncols = min(n_models, 2)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    palette = sns.color_palette("Set2", 6)
    task_colors: Dict[str, Any] = {}

    for idx, model in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        tasks = all_results[model]
        for tidx, (task, results) in enumerate(sorted(tasks.items())):
            if task not in task_colors:
                task_colors[task] = palette[len(task_colors) % len(palette)]

            lengths = sorted(results.keys())
            accs = [results[l] for l in lengths]

            ax.plot(
                lengths, accs,
                marker="o", linewidth=2, markersize=6,
                color=task_colors[task],
                label=task.replace("_", " ").title(),
            )

            # Effective length for this task
            eff_len = compute_effective_length(results, threshold_ratio)
            if eff_len is not None and eff_len > min(lengths):
                ax.axvline(
                    x=eff_len, linestyle=":", alpha=0.4,
                    color=task_colors[task], linewidth=1.5,
                )

        # Mark declared max length
        declared = DECLARED_MAX_LENGTHS.get(model)
        if declared is not None:
            ax.axvline(
                x=declared, linestyle="--", color="red", alpha=0.6,
                linewidth=2, label=f"Declared max ({declared//1024}K)",
            )

        # Threshold line
        all_baselines = []
        for task, results in tasks.items():
            if results:
                bl = min(results.keys())
                all_baselines.append(results[bl])
        if all_baselines:
            avg_baseline = np.mean(all_baselines)
            ax.axhline(
                y=threshold_ratio * avg_baseline,
                linestyle="-.", color="gray", alpha=0.4, linewidth=1,
                label=f"{threshold_ratio*100:.0f}% of baseline",
            )

        ax.set_xlabel("Context Length (tokens)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Context Length vs. Task Accuracy",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy vs length plot to {output_path}")


def plot_effective_length_gap(
    all_results: Dict[str, Dict[str, Dict[int, float]]],
    output_path: str,
    threshold_ratio: float = 0.9,
) -> None:
    """Bar chart showing declared vs effective context length per model.

    Args:
        all_results: Nested results dictionary.
        output_path: Path to save figure.
        threshold_ratio: Threshold ratio for effective length computation.
    """
    models = sorted(all_results.keys())

    effective_lengths = []
    declared_lengths = []
    model_labels = []

    for model in models:
        tasks = all_results[model]
        # Average effective length across tasks
        eff_values = []
        for task, results in tasks.items():
            eff = compute_effective_length(results, threshold_ratio)
            if eff is not None:
                eff_values.append(eff)
        avg_eff = int(np.mean(eff_values)) if eff_values else 0
        effective_lengths.append(avg_eff)

        declared = DECLARED_MAX_LENGTHS.get(model, 0)
        declared_lengths.append(declared)
        model_labels.append(model)

    x = np.arange(len(models))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_declared = ax.bar(
        x - bar_width / 2, [d / 1024 for d in declared_lengths],
        bar_width, label="Declared Max (K tokens)",
        color=sns.color_palette("Pastel1")[0], edgecolor="black", linewidth=0.8,
    )
    bars_effective = ax.bar(
        x + bar_width / 2, [e / 1024 for e in effective_lengths],
        bar_width, label="Effective Length (K tokens)",
        color=sns.color_palette("Pastel1")[1], edgecolor="black", linewidth=0.8,
    )

    # Add value labels on bars
    for bar in bars_declared:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                f"{height:.0f}K", ha="center", va="bottom", fontsize=9)
    for bar in bars_effective:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                f"{height:.0f}K", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Context Length (K tokens)", fontsize=12)
    ax.set_title("Declared vs. Effective Context Length", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved effective length gap chart to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize context length vs. accuracy results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/tables",
        help="Directory with benchmark JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.9,
        help="Threshold ratio for effective length (default 0.9).",
    )
    parser.add_argument(
        "--use_defaults",
        action="store_true",
        help="Use built-in default results instead of loading from files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
    })

    # Load or use default results
    if args.use_defaults:
        all_results = DEFAULT_RESULTS
        print("Using built-in default results.")
    else:
        all_results = load_benchmark_results(args.results_dir)
        if not all_results:
            print("No benchmark results found; falling back to defaults.")
            all_results = DEFAULT_RESULTS

    # Generate plots
    plot_accuracy_vs_length(
        all_results,
        os.path.join(args.output_dir, "context_length_vs_accuracy.png"),
        args.threshold_ratio,
    )

    plot_effective_length_gap(
        all_results,
        os.path.join(args.output_dir, "effective_length_gap.png"),
        args.threshold_ratio,
    )

    print("\nAll accuracy plots generated successfully.")


if __name__ == "__main__":
    main()
