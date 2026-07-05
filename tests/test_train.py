"""Trainer tests: batching, the overfit sanity check, checkpoint resume."""

from pathlib import Path

import numpy as np
import pytest

from loom.model import GPT, GPTConfig
from loom.rng import set_seed
from loom.train import TrainConfig, Trainer, load_model

TINY = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)


def make_trainer(max_steps: int = 5, **overrides: float) -> Trainer:
    set_seed(0)
    model = GPT(TINY)
    ids = np.tile(np.arange(32), 40)  # 1280 tokens of a repeating pattern
    config = TrainConfig(
        max_steps=max_steps,
        batch_size=8,
        warmup_steps=2,
        eval_interval=1000,
        log_interval=1,
        **overrides,  # type: ignore[arg-type]
    )
    return Trainer(model=model, token_ids=ids, config=config)


def test_rejects_too_little_data() -> None:
    with pytest.raises(ValueError, match="flat array"):
        Trainer(model=GPT(TINY), token_ids=np.arange(5))


def test_get_batch_shapes_and_shift() -> None:
    trainer = make_trainer()
    x, y = trainer.get_batch()
    assert x.shape == (8, TINY.block_size)
    assert y.shape == (8, TINY.block_size)
    # y is x shifted by one within the source sequence.
    assert np.array_equal(x[0, 1:], y[0, :-1])


def test_train_runs_and_logs_history() -> None:
    trainer = make_trainer(max_steps=3)
    seen: list[dict[str, float]] = []
    history = trainer.train(on_step=seen.append)
    assert trainer.step == 3
    assert len(history) == 3  # log_interval=1
    assert seen == history
    assert {"step", "lr", "loss", "grad_norm"} <= history[0].keys()


def test_model_overfits_repeating_pattern() -> None:
    """The end-to-end sanity check: on trivially predictable data the whole
    stack (tokenized batches -> forward -> backward -> AdamW) must drive the
    loss from ~log(V) down to near zero. If any gradient in the engine were
    wrong, this is the test that fails."""
    trainer = make_trainer(max_steps=150, lr=1e-2, min_lr=1e-3)
    initial = trainer.estimate_loss()["train"]
    trainer.train()
    final = trainer.estimate_loss()["train"]
    assert initial > 3.0  # ~log(32) = 3.47 before training
    assert final < 0.15, f"loss only reached {final}; training is broken"


def test_grad_accumulation_runs_and_learns() -> None:
    trainer = make_trainer(max_steps=60, lr=1e-2, grad_accum_steps=2)
    initial = trainer.estimate_loss()["train"]
    history = trainer.train()
    assert trainer.step == 60
    assert history[-1]["loss"] < initial  # effective batch 16 still learns


def test_grad_accumulation_validation() -> None:
    with pytest.raises(ValueError, match="grad_accum_steps"):
        make_trainer(grad_accum_steps=0)


def test_estimate_loss_reports_both_splits_and_restores_train_mode() -> None:
    trainer = make_trainer()
    losses = trainer.estimate_loss()
    assert set(losses) == {"train", "val"}
    assert trainer.model.training  # estimate_loss must put the model back


def test_checkpoint_round_trip_preserves_weights(tmp_path: Path) -> None:
    trainer = make_trainer(max_steps=3)
    trainer.train()
    path = tmp_path / "ckpt.npz"
    trainer.save_checkpoint(path)

    model, meta = load_model(path)
    assert meta["step"] == 3
    assert meta["model_config"] == TINY.to_dict()
    for (_, original), (_, restored) in zip(
        trainer.model.named_parameters(), model.named_parameters(), strict=True
    ):
        assert np.allclose(original.data, restored.data)


def test_resume_continues_step_count_and_optimizer_state(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.npz"
    trainer_a = make_trainer(max_steps=4)
    trainer_a.train()
    trainer_a.save_checkpoint(path)

    resumed = Trainer.resume(path, token_ids=trainer_a.train_ids)
    assert resumed.step == 4
    assert resumed.optimizer._step_count == 4
    assert len(resumed.optimizer._m) > 0  # Adam moments restored, not cold
    assert resumed.history == trainer_a.history


def test_resumed_training_continues_to_learn(tmp_path: Path) -> None:
    trainer = make_trainer(max_steps=40, lr=1e-2)
    trainer.train()
    mid_loss = trainer.estimate_loss()["train"]
    path = tmp_path / "ckpt.npz"
    trainer.save_checkpoint(path)

    resumed = Trainer.resume(path, token_ids=np.tile(np.arange(32), 40))
    resumed = Trainer(
        model=resumed.model,
        token_ids=np.tile(np.arange(32), 40),
        config=TrainConfig(**{**resumed.config.__dict__, "max_steps": 120, "lr": 1e-2}),
    )
    resumed.train()
    assert resumed.estimate_loss()["train"] < mid_loss


def test_load_model_rejects_unknown_version(tmp_path: Path) -> None:
    import json

    path = tmp_path / "bad.npz"
    np.savez(path, meta=np.array(json.dumps({"version": 99})))
    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        load_model(path)
