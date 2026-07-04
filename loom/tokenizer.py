"""Byte-level byte-pair encoding, trained from raw text.

The tokenizer follows the GPT-2 recipe:

1. Text is split into *pre-tokens* with a regex so that merges never cross
   word boundaries (a space belongs to the word that follows it, contractions
   split off, digits and punctuation group separately).
2. Each pre-token becomes a sequence of UTF-8 bytes, so the base vocabulary
   is exactly the 256 byte values and any input round-trips losslessly. There
   is no unknown token by construction.
3. Training repeatedly merges the most frequent adjacent pair into a new
   token id. Ties break deterministically (lowest pair first), so training
   the same text twice produces the same tokenizer.

The pre-tokenization pattern approximates GPT-2's with the stdlib ``re``
module: ``\\p{L}``/``\\p{N}`` are not available, so Unicode letters are
matched with ``[^\\W\\d_]`` and numbers with ``\\d``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_PRETOKEN_PATTERN = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")

_NUM_BYTES = 256

Word = tuple[int, ...]
Pair = tuple[int, int]


def _merge_word(word: Word, pair: Pair, new_id: int) -> Word:
    """Replace every non-overlapping occurrence of ``pair`` in ``word`` with ``new_id``."""
    merged: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


class BPETokenizer:
    """A trainable byte-level BPE tokenizer with lossless round-tripping."""

    def __init__(
        self,
        merges: list[Pair] | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.merges: list[Pair] = list(merges or [])
        # rank = merge priority: earlier merges apply first during encoding
        self._ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        # id -> raw bytes for every non-special token
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(_NUM_BYTES)}
        for i, (a, b) in enumerate(self.merges):
            self._vocab[_NUM_BYTES + i] = self._vocab[a] + self._vocab[b]
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        self._special_by_id: dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self._chunk_cache: dict[bytes, list[int]] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """Learn a BPE vocabulary of ``vocab_size`` tokens from ``text``.

        ``vocab_size`` counts the 256 base bytes, the learned merges, and any
        special tokens. Training stops early if no pair occurs at least twice.
        """
        specials = special_tokens or []
        num_merges = vocab_size - _NUM_BYTES - len(specials)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size must be at least {_NUM_BYTES + len(specials)} "
                f"(256 bytes + {len(specials)} special tokens), got {vocab_size}"
            )

        # Unique pre-tokens with counts: pair statistics collapse across
        # repeated words, which is what makes training fast on natural text.
        pretoken_counts = Counter(_PRETOKEN_PATTERN.findall(text))
        words: dict[Word, int] = Counter()
        for pretoken, count in pretoken_counts.items():
            words[tuple(pretoken.encode("utf-8"))] += count

        merges: list[Pair] = []
        for new_id in range(_NUM_BYTES, _NUM_BYTES + num_merges):
            pair_counts: Counter[Pair] = Counter()
            for word, count in words.items():
                for pair in zip(word, word[1:], strict=False):
                    pair_counts[pair] += count
            if not pair_counts:
                break
            # Most frequent pair wins; ties break toward the lowest pair ids
            # so training is deterministic regardless of dict order.
            best_pair, best_count = max(
                pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1])
            )
            if best_count < 2:
                break
            merges.append(best_pair)
            updated: dict[Word, int] = Counter()
            for word, count in words.items():
                updated[_merge_word(word, best_pair, new_id)] += count
            words = updated

        special_ids = {token: _NUM_BYTES + len(merges) + i for i, token in enumerate(specials)}
        return cls(merges=merges, special_tokens=special_ids)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return _NUM_BYTES + len(self.merges) + len(self.special_tokens)

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        """Apply learned merges to one pre-token, lowest rank first."""
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            return cached
        word = [int(b) for b in chunk]
        while len(word) >= 2:
            pairs = list(zip(word, word[1:], strict=False))
            best = min(pairs, key=lambda p: self._ranks.get(p, len(self._ranks)))
            if best not in self._ranks:
                break
            word = list(_merge_word(tuple(word), best, _NUM_BYTES + self._ranks[best]))
        self._chunk_cache[chunk] = word
        return word

    def encode(self, text: str, allowed_special: bool = False) -> list[int]:
        """Encode ``text`` to token ids.

        Special tokens in the input are only recognised when
        ``allowed_special=True``; otherwise they are encoded as ordinary text,
        which prevents prompt injection of control tokens.
        """
        if allowed_special and self.special_tokens:
            pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
            ids: list[int] = []
            for part in re.split(f"({pattern})", text):
                if part in self.special_tokens:
                    ids.append(self.special_tokens[part])
                elif part:
                    ids.extend(self.encode(part))
            return ids

        ids = []
        for pretoken in _PRETOKEN_PATTERN.findall(text):
            ids.extend(self._encode_chunk(pretoken.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text. Invalid UTF-8 becomes U+FFFD."""
        parts: list[bytes] = []
        for token_id in ids:
            if token_id in self._special_by_id:
                parts.append(self._special_by_id[token_id].encode("utf-8"))
            elif token_id in self._vocab:
                parts.append(self._vocab[token_id])
            else:
                raise ValueError(f"token id {token_id} is out of vocabulary")
        return b"".join(parts).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        """The raw bytes a single token id stands for (debugging aid)."""
        if token_id in self._special_by_id:
            return self._special_by_id[token_id].encode("utf-8")
        if token_id in self._vocab:
            return self._vocab[token_id]
        raise ValueError(f"token id {token_id} is out of vocabulary")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported tokenizer file version: {payload.get('version')!r}")
        merges = [(int(a), int(b)) for a, b in payload["merges"]]
        return cls(merges=merges, special_tokens=payload.get("special_tokens") or {})
