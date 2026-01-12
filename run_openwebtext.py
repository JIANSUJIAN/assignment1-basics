import json
import os
import pickle
import time

import psutil

from train_bpe import train_bpe


def main():
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    start_time = time.time()
    print(f"Starting BPE training on {input_path}...")

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    end_time = time.time()
    duration = end_time - start_time

    print("\n--- Training Summary ---")
    print(f"Time taken: {duration / 60:.2f} minutes")
    print(f"Final vocabulary size: {len(vocab)}")

    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_str = longest_token_bytes.decode("utf-8")
    print(f"Longest token: '{longest_token_str}' ({len(longest_token_bytes)} bytes")

    # Serialize results
    os.makedirs("output", exist_ok=True)
    with open("output/owt_vocab.json", "w", encoding="utf-8") as f:
        # Convert bytes to list of integers for lossless JSON storage
        json_vocab = {k: list(v) for k, v in vocab.items()}
        json.dump(json_vocab, f, indent=2)

    with open("output/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    print("Results saved to output/ directory.")


if __name__ == "__main__":
    main()
