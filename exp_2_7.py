import random
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer

ts_tokenizer = Tokenizer.from_files("output/ts_vocab.json", "output/ts_merges.pkl", special_tokens=["<|endoftext|>"])

owt_tokenizer = Tokenizer.from_files("output/owt_vocab.json", "output/owt_merges.pkl", special_tokens=["<|endoftext|>"])


def sample_documents(filepath, num_docs=10, delimiter="<|endoftext|>"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    documents = text.split(delimiter)
    # Filter out empty documents
    documents = [doc.strip() for doc in documents if doc.strip()]
    return random.sample(documents, min(num_docs, len(documents)))


def calculate_compression_ratio(tokenizer, documents):
    total_bytes = 0
    total_tokens = 0

    for doc in documents:
        total_bytes += len(doc.encode("utf-8"))
        tokens = tokenizer.encode(doc)
        total_tokens += len(tokens)

    return total_bytes / total_tokens if total_tokens > 0 else 0


ts_docs = sample_documents("data/TinyStoriesV2-GPT4-train.txt", 10)
owt_docs = sample_documents("data/owt_train.txt", 10)

ts_ratio = calculate_compression_ratio(ts_tokenizer, ts_docs)
owt_ratio = calculate_compression_ratio(owt_tokenizer, owt_docs)
owt_on_ts_tokenizer_ratio = calculate_compression_ratio(ts_tokenizer, owt_docs)

print(f"TinyStories tokenizer on TinyStories: {ts_ratio:.2f} bytes/token")
print(f"OpenWebText tokenizer on OpenWebText: {owt_ratio:.2f} bytes/token")
print(f"TinyStories tokenizer on OpenWebText: {owt_on_ts_tokenizer_ratio:.2f} bytes/token")


def estimate_throughput(tokenizer, filepath, num_bytes=1_000_000):
    with open(filepath, "r", encoding="utf-8") as f:
        # Read a chunk of text
        text = f.read(num_bytes)

    start_time = time.time()
    _ = tokenizer.encode(text)
    end_time = time.time()

    duration = end_time - start_time
    bytes_processed = len(text.encode("utf-8"))
    throughput = bytes_processed / duration  # bytes/second

    return throughput, duration


# Using your OWT tokenizer
throughput, duration = estimate_throughput(owt_tokenizer, "data/owt_train.txt")
print(f"Throughput: {throughput:,.2f} bytes/s")
print(f"Throughput: {throughput / 1024**2:,.2f} MB/s")


def encode_to_file(tokenizer, input_path, output_path):
    # Using encode_iterable to process file line-by-line or chunk-by-chunk
    with open(input_path, "r", encoding="utf-8") as f:
        # This assumes your encode_iterable takes an iterable of strings
        ids = list(tokenizer.encode_iterable(f))

    # Save as uint16
    np_ids = np.array(ids, dtype=np.uint16)
    np.save(output_path, np_ids)
    print(f"Saved {len(ids)} tokens to {output_path}")


# Example usage
encode_to_file(ts_tokenizer, "data/TinyStoriesV2-GPT4-train.txt", "data/ts_train.npy")
encode_to_file(owt_tokenizer, "data/owt_train.txt", "data/owt_train.npy")
