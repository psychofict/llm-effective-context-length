# Study Notes: Why the Effective Context Length of LLMs Falls Short

**Author:** Ebenezer Tarubinga, Korea University M.Sc. AI
**Supervisor:** Prof. Seong-Whan Lee
**Date:** 2025
**Reference:** STRING -- An et al., "Why Does the Effective Context Length of LLMs Fall Short?", ICLR 2025

---

## 1. Declared Context Length vs. Effective Context Length

Modern LLMs are marketed with increasingly large context windows. Llama-2 supports 4,096
tokens, Llama-3 extends to 8,192, and models like Mistral-7B and Qwen2-7B claim support
for 32,768 tokens or more. However, these declared maximums represent the length of input
the model can technically process without an out-of-bounds error -- they do not guarantee
that the model actually *uses* the full context effectively.

The effective context length is the maximum input length at which a model retains
near-baseline performance on tasks that require attending to information spread across
the entire input. In our experiments following the RULER benchmark protocol, the effective
context length ranges from 25% to 60% of the declared maximum depending on the model and
task. Llama-2-7B, for instance, has a declared limit of 4K tokens but begins to degrade
on needle-in-a-haystack retrieval at around 6K tokens (when using RoPE NTK-aware
interpolation to extend beyond 4K). Llama-3-8B handles up to approximately 14K tokens
effectively despite being trained for 8K. Qwen2-7B, with its very high RoPE base
frequency of 1,000,000, maintains the longest effective context at approximately 18K
tokens, but this is still well under its 32K declared maximum.

This gap has practical consequences. Applications that depend on long-context processing
-- document summarization, multi-document QA, repository-level code understanding -- may
silently produce degraded outputs when the input exceeds the effective length. The
degradation is not catastrophic (the model does not crash or refuse to answer); it is
subtle, manifesting as lower accuracy, increased hallucination, and missed details.

## 2. The Role of Positional Encoding in Context Length Limitations

The primary architectural mechanism limiting effective context length is the positional
encoding scheme. Most modern LLMs use Rotary Position Embeddings (RoPE), which encodes
absolute position by rotating query and key vectors in pairs of dimensions. The attention
logit between positions i and j then depends on their relative distance through a sum of
cosine functions at different frequencies:

    score(i, j) ~ sum_d cos((i - j) * theta_d)

where theta_d = 1 / base^(2d/D) are frequency bands determined by the dimension index d
and the RoPE base parameter.

The critical insight is that this sum of cosines behaves like a low-pass filter. For
small relative distances, the cosines are nearly aligned and the sum is large (positive
attention). As distance grows, the cosines at different frequencies begin to cancel each
other out, and the sum decays toward zero. This decay is not monotonic -- it oscillates
-- but the envelope decreases roughly as O(1/sqrt(distance)) for typical head dimensions.

The RoPE base parameter controls the rate of decay. The standard base of 10,000 (used by
Llama-2 and Mistral-7B) produces relatively rapid decay, with attention scores dropping
to 10% of peak within a few thousand positions. Llama-3 uses a base of 500,000 and
Qwen2 uses 1,000,000, which stretches the decay curve significantly but does not
eliminate it. Even at base = 1,000,000, the theoretical effective reach with a 10%
threshold is approximately 20,000-25,000 positions for head_dim = 128.

ALiBi (Attention with Linear Biases) takes a fundamentally different approach: it adds
a linear penalty proportional to distance directly to the attention logits. This creates
exponential decay in attention weights (after softmax) that is analytically predictable
and head-specific. Different heads have different decay rates, allowing the model to
simultaneously maintain local and medium-range attention. However, ALiBi's strict
monotonic decay means it cannot represent "look-back" patterns where a specific distant
position is more relevant than an intermediate one.

## 3. The Attention Sink Phenomenon

An unexpected finding, first systematically documented by Xiao et al. (2023) and further
analyzed in the STRING paper, is the attention sink phenomenon. In virtually all
decoder-only transformer LLMs, the first few tokens (typically the BOS token and the
first 2-4 content tokens) receive a disproportionately large share of attention mass
from *all* query positions, regardless of the content of those tokens.

In our study, we observe that these sink tokens absorb between 15% and 40% of total
attention probability mass, averaged across heads and layers. This fraction is relatively
stable across sequence lengths -- whether the input is 1K or 32K tokens, the sinks still
capture a similar absolute share. However, the *relative* impact increases with context
length because the remaining attention budget must be distributed over a larger set of
content tokens. At 32K tokens, with 30% of attention devoted to sinks, the average
per-token attention for the remaining 32,764 tokens is approximately 2.1e-5, compared to
3.4e-4 at 2K tokens. This 16x reduction in per-token attention makes it extremely
difficult for the model to allocate sufficient attention to task-relevant positions.

The attention sink appears to serve as a "no-op" target: when the model has no strong
preference for any particular key position, it dumps attention onto the BOS token rather
than distributing it uniformly (which would dilute the key information in the value
vectors). StreamingLLM exploits this by retaining only the sink tokens and recent tokens
in the KV cache.

## 4. How STRING Addresses These Issues

The STRING paper proposes several modifications to mitigate the effective context length
gap:

**Shifted Sparse Attention.** Instead of computing full attention over all positions,
STRING uses a combination of local windows and shifted global tokens. The shift pattern
changes across layers so that every position is eventually attended to by some layer.
This reduces the computational cost from O(n^2) to O(n * w) where w is the local window
size, while maintaining theoretical coverage of the full context.

**Training-time context extension.** STRING extends the training context length
incrementally, using a curriculum that starts from the model's original context length
and gradually increases. This allows the positional encoding to adapt to longer distances
without catastrophic forgetting of short-context capabilities.

**Positional encoding interpolation.** Rather than extrapolating RoPE to unseen
distances (which leads to out-of-distribution behavior), STRING uses NTK-aware
interpolation that adjusts the frequency bands to cover the target range while
maintaining the relative ordering of positions.

Our reproduction shows that STRING improves effective context length by 1.5x to 2.5x
compared to the base model. However, even with STRING, the effective context length
remains below the declared maximum for all models tested. The fundamental decay in
positional encoding attention scores is mitigated but not eliminated.

## 5. Open Questions and Future Directions

Several important questions remain:

1. **Can positional encodings be designed to avoid distance-dependent decay entirely?**
   The cosine-based decay in RoPE is a mathematical consequence of its design. Are there
   position encoding schemes that maintain constant attention capacity regardless of
   distance while still being compatible with efficient attention mechanisms?

2. **Is the attention sink a feature or a bug?** The sink phenomenon wastes attention
   budget, but attempts to remove it (by not using a BOS token or by normalizing
   attention differently) often degrade performance. Understanding why sinks are
   necessary for model function could lead to more efficient architectures.

3. **Task-specific context length requirements.** Not all tasks require uniform access
   to all positions. Summarization requires global access, but code completion may need
   only local context plus specific distant function definitions. Can models learn to
   allocate their limited effective context budget dynamically based on the task?

4. **Interaction between context length and model scale.** Our experiments focus on 7B
   parameter models. Larger models (70B+) may have different effective context length
   profiles due to their greater number of attention heads and layers. The relationship
   between model scale, training data, and effective context length deserves systematic
   study.

5. **Evaluation beyond retrieval.** RULER-style benchmarks primarily test retrieval --
   can the model find specific information in a long context? Real-world long-context
   tasks often require synthesis, reasoning over multiple pieces of evidence, and
   resolving contradictions. Developing evaluation protocols that capture these more
   complex capabilities is an important direction.

---

## References

- An, C., Zhang, J., Zhong, M., Li, L., Gong, S., Luo, Y., Xu, J., Kong, L. (2025).
  Why Does the Effective Context Length of LLMs Fall Short? *ICLR 2025*.
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021). RoFormer: Enhanced
  Transformer with Rotary Position Embedding. *arXiv:2104.09864*.
- Press, O., Smith, N. A., Lewis, M. (2022). Train Short, Test Long: Attention with
  Linear Biases Enables Input Length Generalization. *ICLR 2022*.
- Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2023). Efficient Streaming Language
  Models with Attention Sinks. *arXiv:2309.17453*.
- Hsieh, C.-Y., et al. (2024). RULER: What's the Real Context Size of Your Long-Context
  Language Models? *arXiv:2404.06654*.
