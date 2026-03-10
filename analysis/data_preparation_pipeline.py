"""
Data Preparation Pipeline for Long-Context Evaluation.

Handles tokenization, chunking, and validation of long documents for
context length experiments. Supports sliding window and strided approaches
for creating evaluation sequences from documents that exceed model limits.

Inspired by the auto_prepare_data.py improvements in the STRING codebase.

Author: Ebenezer Tarubinga, Korea University
Based on: STRING (An et al., ICLR 2025)
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking strategies."""

    max_seq_length: int = 4096
    stride: int = 2048
    min_chunk_length: int = 256
    add_bos: bool = True
    add_eos: bool = True
    truncation_side: str = "right"  # "left" or "right"


@dataclass
class PipelineStats:
    """Statistics collected during data preparation."""

    total_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    skipped_documents: int = 0
    too_short_chunks: int = 0
    max_doc_length: int = 0
    min_doc_length: int = field(default=int(1e9))
    length_distribution: Dict[str, int] = field(default_factory=dict)


def tokenize_document(
    text: str,
    tokenizer: AutoTokenizer,
    add_special_tokens: bool = False,
) -> List[int]:
    """Tokenize a single document.

    Args:
        text: Raw document text.
        tokenizer: HuggingFace tokenizer.
        add_special_tokens: Whether to add BOS/EOS tokens.

    Returns:
        List of token IDs.
    """
    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return encoded


def sliding_window_chunks(
    token_ids: List[int],
    config: ChunkingConfig,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Generator[List[int], None, None]:
    """Generate overlapping chunks using a sliding window approach.

    Args:
        token_ids: Full document token IDs.
        config: Chunking configuration.
        bos_token_id: BOS token ID to prepend (if config.add_bos is True).
        eos_token_id: EOS token ID to append (if config.add_eos is True).

    Yields:
        Token ID lists for each chunk.
    """
    # Account for special tokens in the effective window
    special_prefix = [bos_token_id] if (config.add_bos and bos_token_id is not None) else []
    special_suffix = [eos_token_id] if (config.add_eos and eos_token_id is not None) else []
    effective_length = config.max_seq_length - len(special_prefix) - len(special_suffix)

    if effective_length <= 0:
        logger.warning(
            f"max_seq_length ({config.max_seq_length}) is too small after "
            f"accounting for special tokens. Skipping."
        )
        return

    doc_length = len(token_ids)
    if doc_length == 0:
        return

    start = 0
    while start < doc_length:
        end = min(start + effective_length, doc_length)
        chunk = token_ids[start:end]

        # Skip chunks that are too short
        if len(chunk) < config.min_chunk_length:
            break

        yield special_prefix + chunk + special_suffix
        start += config.stride

        # If the stride overshoots, handle the last chunk
        if start >= doc_length:
            break


def strided_chunks(
    token_ids: List[int],
    config: ChunkingConfig,
) -> Generator[List[int], None, None]:
    """Generate non-overlapping fixed-length chunks.

    Unlike sliding_window_chunks, this uses stride == max_seq_length
    to produce disjoint segments.

    Args:
        token_ids: Full document token IDs.
        config: Chunking configuration with stride == max_seq_length.

    Yields:
        Token ID lists for each chunk.
    """
    non_overlap_config = ChunkingConfig(
        max_seq_length=config.max_seq_length,
        stride=config.max_seq_length,  # No overlap
        min_chunk_length=config.min_chunk_length,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
    )
    yield from sliding_window_chunks(token_ids, non_overlap_config)


def validate_chunk(
    chunk: List[int],
    config: ChunkingConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[bool, str]:
    """Validate a tokenized chunk.

    Args:
        chunk: Token ID list.
        config: Chunking configuration.
        tokenizer: Tokenizer for decoding checks.

    Returns:
        Tuple of (is_valid, reason).
    """
    if len(chunk) == 0:
        return False, "empty chunk"

    if len(chunk) > config.max_seq_length:
        return False, f"exceeds max length ({len(chunk)} > {config.max_seq_length})"

    if len(chunk) < config.min_chunk_length:
        return False, f"below min length ({len(chunk)} < {config.min_chunk_length})"

    # Check for degenerate content (all same token)
    unique_tokens = len(set(chunk))
    if unique_tokens < 3:
        return False, f"degenerate content (only {unique_tokens} unique tokens)"

    return True, "ok"


def categorize_length(length: int) -> str:
    """Categorize document length into bins for statistics."""
    if length < 1024:
        return "<1K"
    elif length < 4096:
        return "1K-4K"
    elif length < 8192:
        return "4K-8K"
    elif length < 16384:
        return "8K-16K"
    elif length < 32768:
        return "16K-32K"
    else:
        return "32K+"


def process_documents(
    input_path: str,
    tokenizer: AutoTokenizer,
    config: ChunkingConfig,
    text_field: str = "text",
    max_documents: Optional[int] = None,
) -> Tuple[List[List[int]], PipelineStats]:
    """Process a JSONL file of documents into tokenized chunks.

    Args:
        input_path: Path to JSONL file where each line has a text field.
        tokenizer: HuggingFace tokenizer.
        config: Chunking configuration.
        text_field: Name of the JSON field containing document text.
        max_documents: Optional limit on documents to process.

    Returns:
        Tuple of (list of token ID chunks, pipeline statistics).
    """
    stats = PipelineStats()
    all_chunks: List[List[int]] = []

    logger.info(f"Processing documents from {input_path}")
    logger.info(f"Config: max_seq_length={config.max_seq_length}, "
                f"stride={config.stride}, min_chunk={config.min_chunk_length}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if max_documents is not None:
        lines = lines[:max_documents]

    for line_idx, line in enumerate(tqdm(lines, desc="Processing documents")):
        try:
            doc = json.loads(line.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping line {line_idx}: JSON decode error: {e}")
            stats.skipped_documents += 1
            continue

        text = doc.get(text_field, "")
        if not text or not text.strip():
            logger.debug(f"Skipping line {line_idx}: empty text")
            stats.skipped_documents += 1
            continue

        # Tokenize
        token_ids = tokenize_document(text, tokenizer)
        doc_length = len(token_ids)

        # Update stats
        stats.total_documents += 1
        stats.total_tokens += doc_length
        stats.max_doc_length = max(stats.max_doc_length, doc_length)
        stats.min_doc_length = min(stats.min_doc_length, doc_length)

        length_bin = categorize_length(doc_length)
        stats.length_distribution[length_bin] = (
            stats.length_distribution.get(length_bin, 0) + 1
        )

        # Generate chunks
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        for chunk in sliding_window_chunks(token_ids, config, bos_id, eos_id):
            is_valid, reason = validate_chunk(chunk, config, tokenizer)
            if is_valid:
                all_chunks.append(chunk)
                stats.total_chunks += 1
            else:
                stats.too_short_chunks += 1
                logger.debug(f"Invalid chunk from doc {line_idx}: {reason}")

    # Fix min_doc_length if no documents were processed
    if stats.total_documents == 0:
        stats.min_doc_length = 0

    return all_chunks, stats


def save_processed_data(
    chunks: List[List[int]],
    output_path: str,
    stats: PipelineStats,
) -> None:
    """Save processed chunks to a binary numpy file and statistics to JSON.

    Args:
        chunks: List of token ID lists.
        output_path: Base path for output files (without extension).
        stats: Pipeline statistics.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save as numpy arrays
    np_path = output_path + ".npy"
    # Pad chunks to uniform length for efficient storage
    if chunks:
        max_len = max(len(c) for c in chunks)
        padded = np.zeros((len(chunks), max_len), dtype=np.int32)
        lengths = np.zeros(len(chunks), dtype=np.int32)
        for i, chunk in enumerate(chunks):
            padded[i, :len(chunk)] = chunk
            lengths[i] = len(chunk)
        np.savez(output_path + ".npz", token_ids=padded, lengths=lengths)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}.npz")

    # Save statistics
    stats_path = output_path + "_stats.json"
    stats_dict = {
        "total_documents": stats.total_documents,
        "total_chunks": stats.total_chunks,
        "total_tokens": stats.total_tokens,
        "skipped_documents": stats.skipped_documents,
        "too_short_chunks": stats.too_short_chunks,
        "max_doc_length": stats.max_doc_length,
        "min_doc_length": stats.min_doc_length,
        "length_distribution": stats.length_distribution,
    }
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")


def print_stats_summary(stats: PipelineStats) -> None:
    """Print a formatted summary of pipeline statistics."""
    logger.info("=" * 50)
    logger.info("PIPELINE STATISTICS")
    logger.info("=" * 50)
    logger.info(f"  Documents processed: {stats.total_documents:,}")
    logger.info(f"  Documents skipped:   {stats.skipped_documents:,}")
    logger.info(f"  Chunks generated:    {stats.total_chunks:,}")
    logger.info(f"  Total tokens:        {stats.total_tokens:,}")
    logger.info(f"  Min doc length:      {stats.min_doc_length:,}")
    logger.info(f"  Max doc length:      {stats.max_doc_length:,}")
    if stats.total_documents > 0:
        avg_len = stats.total_tokens / stats.total_documents
        logger.info(f"  Avg doc length:      {avg_len:,.0f}")
    logger.info(f"  Length distribution:")
    for bin_name in sorted(stats.length_distribution.keys()):
        count = stats.length_distribution[bin_name]
        logger.info(f"    {bin_name:>8s}: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare long-document data for context length experiments."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Base path for output files (without extension).",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace tokenizer name or path.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length per chunk.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2048,
        help="Stride for sliding window (set equal to max_seq_length for no overlap).",
    )
    parser.add_argument(
        "--min_chunk_length",
        type=int,
        default=256,
        help="Minimum chunk length to keep.",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="JSON field name containing document text.",
    )
    parser.add_argument(
        "--max_documents",
        type=int,
        default=None,
        help="Maximum number of documents to process.",
    )
    args = parser.parse_args()

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True
    )

    # Configure chunking
    config = ChunkingConfig(
        max_seq_length=args.max_seq_length,
        stride=args.stride,
        min_chunk_length=args.min_chunk_length,
    )

    # Process documents
    chunks, stats = process_documents(
        input_path=args.input_path,
        tokenizer=tokenizer,
        config=config,
        text_field=args.text_field,
        max_documents=args.max_documents,
    )

    # Print and save results
    print_stats_summary(stats)
    save_processed_data(chunks, args.output_path, stats)


if __name__ == "__main__":
    main()
