from typing import Iterable, Iterator

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Create a rank map for fast merge priority lookup.
        # Smaller index = higher priority
        self.merges_ranks = {pair: i for i, pair in enumerate(merges)}

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # for token in self.special_tokens:
        #     token_bytes = token.encode("utf-8")
        #     if token_bytes in self.inverse_vocab:
        #         new_id = len(self.vocab)
        #         self.vocab[new_id] = token_bytes
        #         self.inverse_vocab[token_bytes] = new_id

        self.pat = re.compile(PAT)

        self.special_token_pattern = None
        if self.special_tokens:
            # Sort by length descending to match longest tokens first
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_special_token = (re.escape(t) for t in sorted_special_tokens)
            self.special_token_pattern = re.compile("|".join(escaped_special_token))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        import json
        import pickle

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        vocab = {int(token_id): bytes(byte_list) for token_id, byte_list in vocab_data.items()}

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _merge(self, tokens: list[bytes]) -> list[bytes]:
        """
        Iteratively merge the most frequent/earliest-ranked pairs
        within a single pre-token
        """
        if len(tokens) <= 1:
            return tokens

        while True:
            # Find all possible pairs in the current sequence
            pairs = []
            for i in range(len(tokens) - 1):
                pairs.append((tokens[i], tokens[i + 1]))

            if not pairs:
                break

            # Find which of these pairs has the highest priority (lowest rank)
            # according to our learned merges.
            bigram_to_merge = min(pairs, key=lambda p: self.merges_ranks.get(p, float("inf")))

            if bigram_to_merge not in self.merges_ranks:
                break

            # Apply the merge: replace all occurrences of (t1, t2) with t1+t2
            t1, t2 = bigram_to_merge
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == t1 and tokens[i + 1] == t2:
                    new_tokens.append(t1 + t2)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            if self.special_token_pattern:
                last_pos = 0
                for match in self.special_token_pattern.finditer(text):
                    pre_chunk = text[last_pos : match.start()]
                    if pre_chunk:
                        yield from self._encode_chunk(pre_chunk)

                    special_token = match.group()
                    yield self.inverse_vocab[special_token.encode("utf-8")]
                    last_pos = match.end()

                remaining_chunk = text[last_pos:]
                if remaining_chunk:
                    yield from self._encode_chunk(remaining_chunk)
            else:
                yield from self._encode_chunk(text)

    def _encode_chunk(self, text) -> Iterator[int]:
        for match in self.pat.finditer(text):
            pre_token = match.group()

            byte_tokens = [bytes([b]) for b in pre_token.encode("utf-8")]

            merged_tokens = self._merge(byte_tokens)

            for token in merged_tokens:
                yield self.inverse_vocab[token]

    def decode(self, ids: list[int]) -> str:
        all_bytes = b"".join(self.vocab[idx] for idx in ids)
        return all_bytes.decode("utf-8", errors="replace")
