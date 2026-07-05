"""Watch a BPE tokenizer learn, merge by merge.

Run from the repo root:

    python examples/bpe_explorer.py

Trains a small vocabulary on the Shakespeare corpus and shows what the
tokenizer actually learned: the first merges are English's most frequent
bigrams, later ones are whole words and morphemes.
"""

from pathlib import Path

from loom import BPETokenizer

text = Path("data/shakespeare.txt").read_text(encoding="utf-8")

print("training a 400-token vocabulary on tiny Shakespeare ...")
tok = BPETokenizer.train(text, vocab_size=400)

print("\nfirst 12 merges (the engine discovers English bigram frequency):")
for i in range(12):
    print(f"   token {256 + i}: {tok.token_bytes(256 + i)!r}")

print("\nlast 12 merges (by now it is learning words and morphemes):")
for i in range(tok.vocab_size - 12, tok.vocab_size):
    print(f"   token {i}: {tok.token_bytes(i)!r}")

sample = "To be, or not to be, that is the question."
ids = tok.encode(sample)
print(f"\ntokenizing: {sample!r}")
print(f"   {len(sample.encode())} bytes -> {len(ids)} tokens")
print("   boundaries:", " | ".join(tok.token_bytes(i).decode("utf-8", "replace") for i in ids))

assert tok.decode(ids) == sample
print("\nround-trip exact: True")
