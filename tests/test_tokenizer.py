"""Tokenizer suite: round-tripping, training determinism, persistence, specials."""

from pathlib import Path

import pytest

from loom.tokenizer import BPETokenizer, _merge_word

CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps again, and the dog sleeps. "
) * 20


def test_merge_word_replaces_non_overlapping_pairs() -> None:
    assert _merge_word((1, 2, 1, 2, 2), (1, 2), 9) == (9, 9, 2)
    assert _merge_word((1, 1, 1), (1, 1), 9) == (9, 1)  # no overlap reuse


def test_untrained_tokenizer_is_pure_bytes() -> None:
    tok = BPETokenizer()
    assert tok.vocab_size == 256
    assert tok.encode("abc") == [97, 98, 99]
    assert tok.decode([97, 98, 99]) == "abc"


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "Tabs\tand\nnewlines  and   runs of spaces",
        "unicode: cafe au lait, naive facade, Zurich",
        "emoji survive: \U0001f9f5\U0001f52e and CJK: 编程",
        "punctuation!!! (all) [of] {it} <works>; don't it?",
        "",
    ],
)
def test_round_trip_is_lossless(text: str) -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    assert tok.decode(tok.encode(text)) == text


def test_training_learns_frequent_merges() -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    assert 256 < tok.vocab_size <= 300
    # Common words in the corpus should compress well below one token per byte.
    ids = tok.encode("The quick brown fox")
    assert len(ids) < len(b"The quick brown fox")


def test_training_is_deterministic() -> None:
    tok_a = BPETokenizer.train(CORPUS, vocab_size=300)
    tok_b = BPETokenizer.train(CORPUS, vocab_size=300)
    assert tok_a.merges == tok_b.merges


def test_classic_bpe_example_merges_aa_first() -> None:
    # The canonical example from the BPE literature: "aa" is the most
    # frequent pair, so it becomes the first learned token (id 256).
    tok = BPETokenizer.train("aaabdaaabac", vocab_size=260)
    assert tok.merges[0] == (97, 97)  # ord("a") == 97


def test_training_stops_when_no_pair_repeats() -> None:
    tok = BPETokenizer.train("abcdefg", vocab_size=1000)
    assert tok.merges == []  # every pair occurs once; nothing worth merging


def test_vocab_size_below_minimum_raises() -> None:
    with pytest.raises(ValueError, match="vocab_size must be at least"):
        BPETokenizer.train("text", vocab_size=100)


def test_merges_never_cross_word_boundaries() -> None:
    # "sunset" and "set sun" share byte content across a space; the space
    # prefix convention must keep " sun" distinct from "sun".
    tok = BPETokenizer.train("sunset set sun sunset set sun " * 30, vocab_size=300)
    text = "sunset set sun"
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_ignored_by_default() -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=300, special_tokens=["<|endoftext|>"])
    special_id = tok.special_tokens["<|endoftext|>"]
    ids = tok.encode("hello <|endoftext|> world")
    assert special_id not in ids  # treated as plain text: injection-safe
    assert tok.decode(ids) == "hello <|endoftext|> world"


def test_special_tokens_recognised_when_allowed() -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=300, special_tokens=["<|endoftext|>"])
    special_id = tok.special_tokens["<|endoftext|>"]
    ids = tok.encode("a<|endoftext|>b", allowed_special=True)
    assert ids.count(special_id) == 1
    assert tok.decode(ids) == "a<|endoftext|>b"


def test_special_tokens_counted_in_vocab_size() -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=300, special_tokens=["<|endoftext|>"])
    assert tok.vocab_size <= 300
    assert tok.special_tokens["<|endoftext|>"] == tok.vocab_size - 1


def test_save_load_round_trip(tmp_path: Path) -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=300, special_tokens=["<|endoftext|>"])
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    sample = "The quick brown fox! <|endoftext|>"
    assert loaded.encode(sample, allowed_special=True) == tok.encode(sample, allowed_special=True)


def test_load_rejects_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"version": 99, "merges": []}')
    with pytest.raises(ValueError, match="unsupported tokenizer file version"):
        BPETokenizer.load(path)


def test_decode_out_of_vocab_raises() -> None:
    tok = BPETokenizer()
    with pytest.raises(ValueError, match="out of vocabulary"):
        tok.decode([9999])


def test_token_bytes_exposes_learned_tokens() -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    first_merge_bytes = tok.token_bytes(256)
    assert len(first_merge_bytes) == 2
    assert tok.token_bytes(97) == b"a"


def test_encode_cache_is_consistent() -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    assert tok.encode("the quick") == tok.encode("the quick")
