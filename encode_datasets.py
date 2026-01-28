"""
Encode text datasets into token IDs using trained BPE tokenizers.
Outputs NumPy arrays of dtype uint16 for memory-efficient storage.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def stream_file_lines(filepath: str, chunk_size: int = 10000):
    """Yield chunks of lines from a file for memory-efficient processing."""
    with open(filepath, "r", encoding="utf-8") as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def encode_dataset(
    tokenizer: Tokenizer,
    input_path: str,
    output_path: str,
    chunk_size: int = 10000,
):
    """
    Encode a text file into token IDs and save as uint16 NumPy array.
    
    Uses streaming to handle large files without loading everything into memory.
    """
    print(f"Encoding {input_path} -> {output_path}")
    
    all_token_ids = []
    total_lines = 0
    
    start_time = time.time()
    
    for chunk in stream_file_lines(input_path, chunk_size):
        # Join lines and encode the chunk
        text = "".join(chunk)
        token_ids = list(tokenizer.encode_iterable([text]))
        all_token_ids.extend(token_ids)
        
        total_lines += len(chunk)
        if total_lines % 100000 == 0:
            print(f"  Processed {total_lines:,} lines, {len(all_token_ids):,} tokens so far...")
    
    # Convert to uint16 numpy array
    token_array = np.array(all_token_ids, dtype=np.uint16)
    
    # Verify all token IDs fit in uint16
    max_token_id = token_array.max() if len(token_array) > 0 else 0
    if max_token_id > 65535:
        raise ValueError(f"Token ID {max_token_id} exceeds uint16 max (65535)")
    
    # Save to disk
    np.save(output_path, token_array)
    
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"  Done! {len(token_array):,} tokens in {elapsed:.1f}s")
    print(f"  Output size: {file_size_mb:.2f} MB")
    print(f"  Max token ID: {max_token_id}")
    
    return token_array


def main():
    parser = argparse.ArgumentParser(
        description="Encode text datasets using trained BPE tokenizers"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["tinystories", "openwebtext", "all"],
        help="Which dataset to encode",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of lines to process at a time (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/encoded",
        help="Directory to save encoded arrays (default: data/encoded)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets_config = {
        "tinystories": {
            "vocab": "output/ts_vocab.json",
            "merges": "output/ts_merges.pkl",
            "special_tokens": ["<|endoftext|>"],
            "files": [
                ("data/TinyStoriesV2-GPT4-train.txt", "ts_train.npy"),
                ("data/TinyStoriesV2-GPT4-valid.txt", "ts_valid.npy"),
            ],
        },
        "openwebtext": {
            "vocab": "output/owt_vocab.json",
            "merges": "output/owt_merges.pkl",
            "special_tokens": ["<|endoftext|>"],
            "files": [
                ("data/owt_train.txt", "owt_train.npy"),
                ("data/owt_valid.txt", "owt_valid.npy"),
            ],
        },
    }

    datasets_to_process = (
        ["tinystories", "openwebtext"] if args.dataset == "all" else [args.dataset]
    )

    for dataset_name in datasets_to_process:
        config = datasets_config[dataset_name]
        
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load tokenizer
        print(f"Loading tokenizer from {config['vocab']} and {config['merges']}...")
        tokenizer = Tokenizer.from_files(
            config["vocab"],
            config["merges"],
            config["special_tokens"],
        )
        print(f"  Vocabulary size: {len(tokenizer.vocab):,}")
        
        # Encode each file
        for input_file, output_name in config["files"]:
            if not os.path.exists(input_file):
                print(f"  Skipping {input_file} (file not found)")
                continue
                
            output_path = os.path.join(args.output_dir, output_name)
            encode_dataset(
                tokenizer,
                input_file,
                output_path,
                chunk_size=args.chunk_size,
            )

    print(f"\n{'='*60}")
    print("Encoding complete!")
    print(f"Encoded files saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
