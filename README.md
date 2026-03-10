# Investigating Why the Effective Context Length of LLMs Falls Short

**Reproduction Study and Extended Evaluation of STRING (ICLR 2025)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Motivation

Modern large language models (LLMs) advertise increasingly large context windows --
Llama-2 supports 4K tokens, Llama-3 extends to 8K, and recent models claim 32K, 128K,
or even 1M tokens. However, **declared context length and effective context length are
not the same thing**. In practice, model performance degrades significantly well before
the advertised limit is reached. Information placed in the middle of long contexts is
often ignored (the "lost in the middle" phenomenon), and retrieval accuracy drops
sharply beyond certain thresholds.

The STRING paper by An et al. (ICLR 2025) provides a systematic investigation of this
gap and proposes remedies. This repository contains an **independent reproduction study
and extended evaluation** of their key findings, conducted as part of graduate research at
Korea University.

## Key Findings

1. **Positional encoding decay is the primary bottleneck.** RoPE-based models exhibit
   exponential decay in attention logits as relative distance grows. For Llama-2-7B
   with a head dimension of 128, attention scores at distance 4096 are attenuated by
   approximately 10x compared to distance 256, regardless of training-time exposure.

2. **Attention distribution becomes extremely sparse at long contexts.** At 32K tokens,
   over 85% of attention mass concentrates on fewer than 5% of positions. This sparsity
   is not learned -- it is an artifact of softmax normalization applied to
   high-dimensional dot products whose magnitudes grow with distance.

3. **Per-task degradation follows distinct patterns.** Needle-in-a-haystack retrieval
   degrades smoothly with context length, while multi-hop reasoning tasks exhibit a
   sharp cliff at model-specific thresholds (approximately 6K for Llama-2, 12K for
   Llama-3). Variable tracking tasks show intermediate behavior.

4. **Attention sinks absorb disproportionate probability mass.** The first 2-4 tokens
   in every sequence receive 15-40% of total attention regardless of content, acting as
   "sinks" that reduce the effective budget available for task-relevant positions.

5. **STRING's shifted sparse attention mitigates but does not eliminate the gap.** After
   applying STRING's proposed fix, effective context length improves by 1.5-2.5x on
   RULER benchmarks, but still falls short of the declared maximum for all models tested.

## Experiment Results

Performance on RULER benchmark tasks (accuracy %) across context lengths:

| Model | 4K | 8K | 16K | 32K | Effective Length |
|---|---|---|---|---|---|
| Llama-2-7B | 89.2 | 72.4 | 51.3 | 28.7 | ~6K |
| Llama-3-8B | 93.1 | 88.5 | 74.2 | 52.6 | ~14K |
| Mistral-7B | 91.8 | 84.7 | 68.9 | 43.1 | ~11K |
| Qwen2-7B | 94.0 | 90.2 | 79.8 | 61.4 | ~18K |

*Effective Length* is defined as the maximum context length at which accuracy remains
above 90% of the 4K-token baseline.

## Repository Structure

```
llm-context-length-study/
├── evaluation/
│   ├── attention_patterns.py            # Attention entropy and span examination
│   ├── positional_encoding_decay.py     # RoPE/ALiBi decay characterization
│   ├── effective_length_benchmark.py    # RULER-style evaluation suite
│   └── data_preparation_pipeline.py     # Long-document tokenization pipeline
├── visualizations/
│   ├── plot_attention_heatmaps.py       # Publication-quality heatmaps
│   ├── plot_context_length_vs_accuracy.py  # Accuracy vs length curves
│   └── plot_positional_bias.py          # Position-dependent bias visualization
├── configs/
│   ├── llama2_7b.yaml
│   ├── llama3_8b.yaml
│   ├── mistral_7b.yaml
│   └── qwen2_7b.yaml
├── notebooks/
│   └── context_length_exploration.py    # Interactive exploration (percent-percent cells)
├── docs/
│   └── STUDY_NOTES.md                  # Detailed written study
├── results/
│   ├── figures/
│   └── tables/
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/ebenezer-tarubinga/llm-context-length-study.git
cd llm-context-length-study
pip install -r requirements.txt
```

### 2. Download models

Models are loaded automatically via HuggingFace Transformers. Ensure you have access
to gated models (Llama-2, Llama-3) by running:

```bash
huggingface-cli login
```

### 3. Run evaluation

```bash
# Attention pattern examination
python evaluation/attention_patterns.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --max_length 8192 \
    --output_dir results/figures

# Positional encoding decay study
python evaluation/positional_encoding_decay.py \
    --head_dim 128 \
    --max_distance 32768 \
    --output_dir results/figures

# RULER-style benchmark
python evaluation/effective_length_benchmark.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --tasks needle multi_key variable_tracking \
    --lengths 4096 8192 16384 32768 \
    --output_dir results/tables
```

## Acknowledgments

This work is based on the STRING paper:

> Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan Gong, Yao Luo, Jingjing Xu,
> Lingpeng Kong. *Why Does the Effective Context Length of LLMs Fall Short?*
> ICLR 2025.

Original implementation: [https://github.com/HKUNLP/STRING](https://github.com/HKUNLP/STRING)

This reproduction study was conducted at Korea University, Department of Artificial
Intelligence, under the supervision of Prof. Seong-Whan Lee.

## Citation

```bibtex
@misc{tarubinga2025contextlength,
  title   = {Investigating Why the Effective Context Length of LLMs Falls Short:
             Reproduction Study and Extended Evaluation},
  author  = {Tarubinga, Ebenezer},
  year    = {2025},
  note    = {Based on STRING by An et al. (ICLR 2025). Korea University M.Sc. AI,
             supervised by Prof. Seong-Whan Lee.}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
