"""Training loop: batching, LR scheduling, evaluation, checkpointing.

A checkpoint is a single ``.npz`` holding the model weights, the AdamW
moment estimates (so resuming continues the same trajectory rather than
restarting Adam cold), and a JSON metadata blob with the model config,
step counter, and loss history. ``resume()`` reconstructs all of it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from loom.model import GPT, GPTConfig
from loom.optim import AdamW, clip_grad_norm, cosine_lr
from loom.rng import get_rng


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 32
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    eval_interval: int = 250
    eval_batches: int = 10
    log_interval: int = 50


@dataclass
class Trainer:
    """Drives training of a GPT on a flat array of token ids."""

    model: GPT
    token_ids: np.ndarray
    config: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        ids = np.asarray(self.token_ids, dtype=np.int64)
        min_needed = self.model.config.block_size + 2
        if ids.ndim != 1 or ids.size < min_needed:
            raise ValueError(f"token_ids must be a flat array of at least {min_needed} tokens")
        split = int(ids.size * (1.0 - self.config.val_fraction))
        self.train_ids = ids[:split]
        self.val_ids = ids[split:]
        self.optimizer = AdamW.for_model(
            self.model, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.step = 0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def get_batch(self, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        """Sample ``batch_size`` random windows; targets are inputs shifted by one."""
        data = self.train_ids if split == "train" else self.val_ids
        block = self.model.config.block_size
        starts = get_rng().integers(0, data.size - block - 1, size=self.config.batch_size)
        x = np.stack([data[s : s + block] for s in starts])
        y = np.stack([data[s + 1 : s + block + 1] for s in starts])
        return x, y

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def estimate_loss(self) -> dict[str, float]:
        """Mean loss over a few batches of each split, in eval mode."""
        self.model.eval()
        losses: dict[str, float] = {}
        for split in ("train", "val"):
            total = 0.0
            for _ in range(self.config.eval_batches):
                x, y = self.get_batch(split)
                _, loss = self.model.forward(x, targets=y)
                assert loss is not None
                total += loss.item()
            losses[split] = total / self.config.eval_batches
        self.model.train()
        return losses

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def train(
        self, on_step: Callable[[dict[str, float]], None] | None = None
    ) -> list[dict[str, float]]:
        """Run up to ``max_steps`` optimization steps; returns the loss history.

        ``on_step`` (if given) receives each logged record, which is how the
        CLI prints progress without the trainer knowing about terminals.
        """
        self.model.train()
        while self.step < self.config.max_steps:
            lr = cosine_lr(
                self.step,
                max_lr=self.config.lr,
                min_lr=self.config.min_lr,
                warmup_steps=self.config.warmup_steps,
                total_steps=self.config.max_steps,
            )
            self.optimizer.lr = lr

            x, y = self.get_batch("train")
            _, loss = self.model.forward(x, targets=y)
            assert loss is not None
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.step += 1

            if self.step % self.config.log_interval == 0 or self.step == 1:
                record: dict[str, float] = {
                    "step": float(self.step),
                    "lr": lr,
                    "loss": loss.item(),
                    "grad_norm": grad_norm,
                }
                if self.step % self.config.eval_interval == 0:
                    record.update({f"{k}_loss": v for k, v in self.estimate_loss().items()})
                self.history.append(record)
                if on_step is not None:
                    on_step(record)
        return self.history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, include_optimizer: bool = True) -> None:
        """Write a single-file checkpoint.

        With ``include_optimizer=False`` only the weights and metadata are
        saved (about a third of the size): right for shipping a trained
        model, wrong for resuming, since Adam would restart cold.
        """
        arrays: dict[str, np.ndarray] = {}
        named = self.model.named_parameters()
        for name, param in named:
            arrays[f"model.{name}"] = param.data
        if include_optimizer:
            for name, param in named:
                key = id(param)
                if key in self.optimizer._m:
                    arrays[f"optim.m.{name}"] = self.optimizer._m[key]
                    arrays[f"optim.v.{name}"] = self.optimizer._v[key]
        meta = {
            "version": 1,
            "model_config": self.model.config.to_dict(),
            "train_config": asdict(self.config),
            "step": self.step,
            "adam_step": self.optimizer._step_count,
            "history": self.history,
        }
        arrays["meta"] = np.array(json.dumps(meta))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)  # type: ignore[arg-type]  # numpy stubs mistype **kwds

    @classmethod
    def resume(
        cls, path: str | Path, token_ids: np.ndarray, config: TrainConfig | None = None
    ) -> Trainer:
        """Rebuild model, optimizer moments, and step counter from a checkpoint.

        ``config`` overrides the checkpointed train config, which is how a
        resumed run extends ``max_steps`` beyond the original target.
        """
        model, meta, arrays = load_model(path, return_arrays=True)
        trainer = cls(
            model=model,
            token_ids=token_ids,
            config=config if config is not None else TrainConfig(**meta["train_config"]),
        )
        trainer.step = int(meta["step"])
        trainer.optimizer._step_count = int(meta["adam_step"])
        trainer.history = list(meta["history"])
        for name, param in model.named_parameters():
            m_key, v_key = f"optim.m.{name}", f"optim.v.{name}"
            if m_key in arrays:
                trainer.optimizer._m[id(param)] = arrays[m_key]
                trainer.optimizer._v[id(param)] = arrays[v_key]
        return trainer


def load_model(path: str | Path, return_arrays: bool = False) -> Any:
    """Load a GPT (and optionally the raw arrays) from a checkpoint file."""
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    meta = json.loads(str(arrays.pop("meta")))
    if meta.get("version") != 1:
        raise ValueError(f"unsupported checkpoint version: {meta.get('version')!r}")
    model = GPT(GPTConfig(**meta["model_config"]))
    state = {
        name.removeprefix("model."): value
        for name, value in arrays.items()
        if name.startswith("model.")
    }
    model.load_state_dict(state)
    if return_arrays:
        return model, meta, arrays
    return model, meta
