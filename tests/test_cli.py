"""End-to-end CLI tests: train, generate, and eval on a tiny corpus.

These run the real commands against a real (miniature) text file in a tmp
directory, so they cover the whole artifact lifecycle the README promises.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from loom.cli import main
from loom.evaluate import perplexity
from loom.model import GPT, GPTConfig
from loom.rng import set_seed
from loom.train import load_model

CORPUS = (
    "the cat sat on the mat. the dog sat on the log. the cat saw the dog and the dog saw the cat. "
) * 60


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(CORPUS)
    return path


def train_args(corpus_file: Path, out: Path, steps: int = 4) -> list[str]:
    return [
        "train",
        "--data",
        str(corpus_file),
        "--out",
        str(out),
        "--vocab-size",
        "280",
        "--block-size",
        "16",
        "--n-layer",
        "1",
        "--n-head",
        "2",
        "--n-embd",
        "16",
        "--steps",
        str(steps),
        "--batch-size",
        "4",
        "--warmup",
        "2",
        "--log-interval",
        "2",
        "--eval-interval",
        "100",
    ]


def test_train_writes_all_artifacts(
    corpus_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    assert main(train_args(corpus_file, out)) == 0
    assert (out / "tokenizer.json").exists()
    assert (out / "checkpoint.npz").exists()
    assert (out / "model.npz").exists()
    history = json.loads((out / "history.json").read_text())
    assert len(history) >= 2
    printed = capsys.readouterr().out
    assert "parameters" in printed
    assert "final:" in printed


def test_model_npz_is_weights_only(corpus_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    main(train_args(corpus_file, out))
    with np.load(out / "model.npz") as slim:
        assert not [k for k in slim.files if k.startswith("optim.")]
    with np.load(out / "checkpoint.npz") as full:
        assert [k for k in full.files if k.startswith("optim.")]


def test_train_resume_reuses_tokenizer_and_continues(
    corpus_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    main(train_args(corpus_file, out, steps=3))
    capsys.readouterr()
    assert main(train_args(corpus_file, out, steps=6) + ["--resume"]) == 0
    printed = capsys.readouterr().out
    assert "loaded tokenizer" in printed
    assert "resumed from" in printed
    model, meta = load_model(out / "checkpoint.npz")
    assert meta["step"] == 6


def test_generate_produces_text(
    corpus_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    main(train_args(corpus_file, out))
    capsys.readouterr()
    code = main(
        [
            "generate",
            "--checkpoint",
            str(out / "model.npz"),
            "--tokenizer",
            str(out / "tokenizer.json"),
            "--prompt",
            "the cat",
            "--tokens",
            "8",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.startswith("the cat")
    assert len(printed) > len("the cat")


def test_generate_is_seed_reproducible(
    corpus_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    main(train_args(corpus_file, out))
    capsys.readouterr()
    args = [
        "generate",
        "--checkpoint",
        str(out / "model.npz"),
        "--tokenizer",
        str(out / "tokenizer.json"),
        "--prompt",
        "the",
        "--tokens",
        "6",
        "--seed",
        "7",
    ]
    main(args)
    first = capsys.readouterr().out
    main(args)
    second = capsys.readouterr().out
    assert first == second


def test_eval_reports_perplexity(
    corpus_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    main(train_args(corpus_file, out))
    capsys.readouterr()
    code = main(
        [
            "eval",
            "--checkpoint",
            str(out / "model.npz"),
            "--tokenizer",
            str(out / "tokenizer.json"),
            "--data",
            str(corpus_file),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "perplexity:" in printed


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "loom" in capsys.readouterr().out


# ----------------------------------------------------------------------
# perplexity unit behaviour
# ----------------------------------------------------------------------


def test_perplexity_of_untrained_model_near_uniform() -> None:
    set_seed(0)
    config = GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    ids = np.arange(400) % 64
    ppl = perplexity(model, ids)
    assert 40 < ppl < 90  # untrained: close to the uniform baseline of 64


def test_perplexity_requires_enough_tokens() -> None:
    config = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)
    with pytest.raises(ValueError, match="at least"):
        perplexity(GPT(config), np.arange(4))


def test_perplexity_restores_training_mode() -> None:
    config = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(config)
    model.train()
    perplexity(model, np.arange(100) % 16)
    assert model.training
