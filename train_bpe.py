from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count

import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path, vocab_size, special_tokens, num_workers=None):
    vocab: dict[int, bytes] = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    pretoken_counts = pretokenize_parallel(input_path, special_tokens, num_workers)

    merges = compute_bpe_merges_indexed(pretoken_counts, vocab, num_merges)

    return vocab, merges


def _pretokenize_chunk(chunk_args, special_tokens):
    """Worker function to process a single chunk of the file."""
    path, start, end = chunk_args
    with open(path, "rb") as f:
        f.seek(start)
        # Read the exact chunk and decode
        chunk_bytes = f.read(end - start)
        text = chunk_bytes.decode("utf-8", errors="replace")
    return pretokenize(text, special_tokens)


def pretokenize_parallel(input_path, special_tokens, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")

    # Prepare args for workers
    chunks = [(input_path, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    worker_func = partial(_pretokenize_chunk, special_tokens=special_tokens)

    print(f"Pre-tokenizing using {len(chunks)} worker... ")
    with Pool(len(chunks)) as pool:
        results = pool.map(worker_func, chunks)

    total_counts = Counter()
    for count_dict in results:
        total_counts.update(count_dict)

    return total_counts


def pretokenize(text, special_tokens):
    """
    Pre-tokenize text and return frequency counts.
    Each pre-token is represented as a tuple of single bytes.
    """
    pretoken_counts = Counter()

    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pretoken = match.group()
            pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            pretoken_counts[pretoken_bytes] += 1

    return pretoken_counts


def merge_pair_in_pretoken(pretoken, pair, new_token):
    """Repleace all occurences of pair with new_token in pretoken"""
    result = []
    i = 0
    while i < len(pretoken):
        if i < len(pretoken) - 1 and pretoken[i] == pair[0] and pretoken[i + 1] == pair[1]:
            result.append(new_token)
            i += 2
        else:
            result.append(pretoken[i])
            i += 1

    return tuple(result)


def compute_bpe_merges(pretoken_counts, vocab, num_merges):
    """
    Compute BPE merge and update vocabulary

    Args:
        pretoken_counts: Freq of each pre-token (will be modified in place)
        vocab: Current vocabulary (will be modfied in place)
        num_merges: Numebr of merges to perform

    Return:
        List of merges in order they were performed
    """

    merges: list[tuple[bytes, bytes]] = []

    for _ in tqdm(range(num_merges), desc="Merging tokens"):
        # Count all adjacent pairs
        pair_counts: dict[tuple[bytes, bytes], int] = {}

        for pretoken, count in pretoken_counts.items():
            # pretoken is a tuple like (b't', b'h', b'e')
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count

        if not pair_counts:
            break

        # Find most frequent pair (tie-break: lexicorgraphically greater)
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))

        # Merge the pair in all pre-tokens
        new_token = best_pair[0] + best_pair[1]

        new_pretoken_counts = {}

        for pretoken, count in pretoken_counts.items():
            new_pretoken = merge_pair_in_pretoken(pretoken, best_pair, new_token)
            new_pretoken_counts[new_pretoken] = new_pretoken_counts.get(new_pretoken, 0) + count

        pretoken_counts.clear()
        pretoken_counts.update(new_pretoken_counts)

        vocab[len(vocab)] = new_token

        merges.append(best_pair)

    return merges


def get_pairs_from_pretoken(pretoken):
    return [(pretoken[i], pretoken[i + 1]) for i in range(len(pretoken) - 1)]


def compute_bpe_merges_indexed(pretoken_counts, vocab, num_merges):
    merges = []

    # Global pair counts across all pre-toakens
    pair_counts: dict[tuple[bytes, bytes], int] = Counter()

    # Index: which pre-tokens contain each pair
    # pair -> set of pre-tokens that contain this pair
    pair_to_pretokens = defaultdict(set)

    # Build initial count and Index
    for pretoken, count in pretoken_counts.items():
        for pair in get_pairs_from_pretoken(pretoken):
            pair_counts[pair] += count
            pair_to_pretokens[pair].add(pretoken)

    for _ in tqdm(range(num_merges), desc="Merging tokens"):
        if not pair_counts:
            break

        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))

        new_token = best_pair[0] + best_pair[1]

        affected_tokens = list(pair_to_pretokens[best_pair])

        for old_pretoken in affected_tokens:
            if old_pretoken not in pretoken_counts:
                continue

            count = pretoken_counts[old_pretoken]

            new_pretoken = merge_pair_in_pretoken(old_pretoken, best_pair, new_token)

            # Remove old pre-token's contribution to pair counts
            for pair in get_pairs_from_pretoken(old_pretoken):
                pair_counts[pair] -= count
                pair_to_pretokens[pair].discard(old_pretoken)
                # Clean up zero counts
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            # Add new pre-token's contribution to pair counts
            for pair in get_pairs_from_pretoken(new_pretoken):
                pair_counts[pair] += count
                pair_to_pretokens[pair].add(new_pretoken)

            # Update pretoken_counts
            del pretoken_counts[old_pretoken]
            pretoken_counts[new_pretoken] = pretoken_counts.get(new_pretoken, 0) + count

        # Clean up the merged pair from the Index
        if best_pair in pair_to_pretokens:
            del pair_to_pretokens[best_pair]
        if best_pair in pair_counts:
            del pair_counts[best_pair]

        vocab[len(vocab)] = new_token
        merges.append(best_pair)

    return merges
