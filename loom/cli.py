"""The ``loom`` command line: train, generate, evaluate.

Everything the CLI does is a thin composition of the library: it exists so
the full workflow (raw text file in, trained model and samples out) is one
command each, with no notebook or script required.

Artifacts live in one output directory per run:

    out/
      tokenizer.json    trained BPE tokenizer
      checkpoint.npz    weights + optimizer state (for resuming)
      model.npz         weights only (the shippable artifact)
      history.json      loss curve records for plotting
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from loom import __version__
from loom.evaluate import perplexity
from loom.model import GPT, GPTConfig
from loom.rng import set_seed
from loom.tokenizer import BPETokenizer
from loom.train import TrainConfig, Trainer, load_model


def _load_artifacts(checkpoint: Path, tokenizer: Path) -> tuple[GPT, BPETokenizer]:
    model, _ = load_model(checkpoint)
    model.eval()
    return model, BPETokenizer.load(tokenizer)


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = Path(args.data).read_text(encoding="utf-8")

    tokenizer_path = out / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"loaded tokenizer ({tokenizer.vocab_size} tokens) from {tokenizer_path}")
    else:
        print(f"training tokenizer (vocab {args.vocab_size}) on {len(text):,} chars ...")
        tokenizer = BPETokenizer.train(text, vocab_size=args.vocab_size)
        tokenizer.save(tokenizer_path)
        print(f"saved {tokenizer_path}")

    ids = np.array(tokenizer.encode(text), dtype=np.int64)
    print(f"corpus: {ids.size:,} tokens ({len(text.encode()) / ids.size:.2f} bytes/token)")

    train_config = TrainConfig(
        max_steps=args.steps,
        schedule_steps=args.schedule_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        min_lr=args.lr / 10.0,
        warmup_steps=args.warmup,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
    )

    checkpoint_path = out / "checkpoint.npz"
    if args.resume and checkpoint_path.exists():
        trainer = Trainer.resume(checkpoint_path, token_ids=ids, config=train_config)
        print(f"resumed from {checkpoint_path} at step {trainer.step}")
    else:
        model = GPT(
            GPTConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=args.block_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                dropout=args.dropout,
            )
        )
        trainer = Trainer(model=model, token_ids=ids, config=train_config)
        print(f"model: {trainer.model.num_parameters():,} parameters")

    started = time.time()
    last_save = started

    def on_step(record: dict[str, float]) -> None:
        nonlocal last_save
        parts = [f"step {int(record['step']):>5}", f"loss {record['loss']:.4f}"]
        if "val_loss" in record:
            parts.append(f"val {record['val_loss']:.4f}")
        parts.append(f"lr {record['lr']:.2e}")
        parts.append(f"{(time.time() - started) / record['step'] * 1000:.0f} ms/step")
        print("  ".join(parts), flush=True)
        if time.time() - last_save > args.save_every:
            trainer.save_checkpoint(checkpoint_path)
            last_save = time.time()

    trainer.train(on_step=on_step)

    trainer.save_checkpoint(checkpoint_path)
    trainer.save_checkpoint(out / "model.npz", include_optimizer=False)
    (out / "history.json").write_text(json.dumps(trainer.history, indent=2))
    print(f"saved {checkpoint_path}, {out / 'model.npz'}, {out / 'history.json'}")

    final = trainer.estimate_loss()
    print(f"final: train loss {final['train']:.4f}, val loss {final['val']:.4f}")
    return 0


# ----------------------------------------------------------------------
# generate
# ----------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    model, tokenizer = _load_artifacts(Path(args.checkpoint), Path(args.tokenizer))
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.encode("\n")[0]]
    out = model.generate(
        np.array([prompt_ids]),
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tokenizer.decode([int(i) for i in out[0]]))
    return 0


# ----------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------


def _cmd_eval(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    model, tokenizer = _load_artifacts(Path(args.checkpoint), Path(args.tokenizer))
    text = Path(args.data).read_text(encoding="utf-8")
    ids = np.array(tokenizer.encode(text), dtype=np.int64)
    holdout = ids[int(ids.size * (1.0 - args.val_fraction)) :]
    ppl = perplexity(model, holdout)
    print(f"tokens evaluated: {holdout.size:,}")
    print(f"perplexity: {ppl:.2f} (uniform baseline: {tokenizer.vocab_size})")
    return 0


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom",
        description="Train and sample a small language model built from scratch.",
    )
    parser.add_argument("--version", action="version", version=f"loom {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a tokenizer and model on a text file")
    train.add_argument("--data", required=True, help="path to a UTF-8 text file")
    train.add_argument("--out", required=True, help="output directory for artifacts")
    train.add_argument("--vocab-size", type=int, default=512)
    train.add_argument("--block-size", type=int, default=128)
    train.add_argument("--n-layer", type=int, default=4)
    train.add_argument("--n-head", type=int, default=4)
    train.add_argument("--n-embd", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument(
        "--schedule-steps",
        type=int,
        default=None,
        help="LR schedule horizon if training in phases (defaults to --steps)",
    )
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--grad-accum", type=int, default=1, help="micro-batches per step")
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--warmup", type=int, default=100)
    train.add_argument("--eval-interval", type=int, default=250)
    train.add_argument("--log-interval", type=int, default=50)
    train.add_argument("--save-every", type=float, default=300.0, help="checkpoint every N seconds")
    train.add_argument(
        "--resume", action="store_true", help="continue from checkpoint.npz if present"
    )
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=_cmd_train)

    generate = sub.add_parser("generate", help="sample text from a trained model")
    generate.add_argument("--checkpoint", required=True, help="model.npz or checkpoint.npz")
    generate.add_argument("--tokenizer", required=True, help="tokenizer.json")
    generate.add_argument("--prompt", default="\n")
    generate.add_argument("--tokens", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--top-p", type=float, default=None, help="nucleus sampling mass")
    generate.add_argument("--seed", type=int, default=42)
    generate.set_defaults(func=_cmd_generate)

    evaluate = sub.add_parser("eval", help="perplexity of a trained model on held-out text")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--tokenizer", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--val-fraction", type=float, default=0.1)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
