"""FastAPI training dashboard server with WebSocket streaming."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from loom import __version__
from loom.model import GPT, GPTConfig
from loom.rng import set_seed
from loom.server.schemas import (
    GenerateRequest,
    GenerateResponse,
    TrainConfig,
    TrainEvent,
    TrainRunResponse,
    TrainStatus,
)
from loom.tokenizer import BPETokenizer
from loom.train import TrainConfig as LoomTrainConfig
from loom.train import Trainer


class TrainingManager:
    """Manages a single active training run."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.status: Literal["idle", "running", "completed", "failed", "stopped"] = "idle"
        self.step: int = 0
        self.total_steps: int = 0
        self.loss: float = 0.0
        self.val_loss: float = 0.0
        self.eta_seconds: float = 0.0
        self.event_queue: asyncio.Queue[TrainEvent] = asyncio.Queue()
        self.cancel_event: asyncio.Event = asyncio.Event()
        self.model: GPT | None = None
        self.tokenizer: BPETokenizer | None = None
        self.checkpoint_dir: Path | None = None
        self.start_time: float = 0.0

    def is_active(self) -> bool:
        """Check if a training run is active."""
        return self.status == "running"

    async def emit(self, event: TrainEvent) -> None:
        """Queue an event for WebSocket broadcast."""
        await self.event_queue.put(event)

    async def start_training(self, config: TrainConfig) -> str:
        """Start a new training run (non-blocking)."""
        if self.is_active():
            raise RuntimeError("Training already in progress")

        self.run_id = str(uuid.uuid4())[:8]
        self.status = "running"
        self.step = 0
        self.total_steps = config.steps
        self.loss = 0.0
        self.val_loss = 0.0
        self.start_time = time.time()
        self.cancel_event.clear()

        self.checkpoint_dir = Path("checkpoints") / self.run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        asyncio.create_task(self._train_loop(config))
        return self.run_id

    async def _train_loop(self, config: TrainConfig) -> None:
        """Run training in a background thread."""
        try:
            await asyncio.to_thread(self._sync_train, config)
        except Exception as e:
            self.status = "failed"
            await self.emit(TrainEvent(event_type="failed", message=f"Training failed: {str(e)}"))

    def _sync_train(self, config: TrainConfig) -> None:
        """Synchronous training loop (runs in thread pool)."""
        assert self.checkpoint_dir is not None
        set_seed(config.seed)
        data_path = Path(config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        text = data_path.read_text(encoding="utf-8")

        tokenizer_path = self.checkpoint_dir / "tokenizer.json"
        if tokenizer_path.exists():
            self.tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            self.tokenizer = BPETokenizer.train(text, vocab_size=config.vocab_size)
            self.tokenizer.save(tokenizer_path)

        ids = np.array(self.tokenizer.encode(text), dtype=np.int64)

        train_config = LoomTrainConfig(
            max_steps=config.steps,
            batch_size=config.batch_size,
            grad_accum_steps=config.grad_accum,
            lr=config.lr,
            min_lr=config.lr / 10.0,
            warmup_steps=config.warmup_steps,
            eval_interval=config.eval_interval,
            log_interval=config.log_interval,
        )

        checkpoint_path = self.checkpoint_dir / "checkpoint.npz"
        if checkpoint_path.exists():
            trainer = Trainer.resume(checkpoint_path, token_ids=ids, config=train_config)
        else:
            model = GPT(
                GPTConfig(
                    vocab_size=self.tokenizer.vocab_size,
                    block_size=config.block_size,
                    n_layer=config.n_layer,
                    n_head=config.n_head,
                    n_embd=config.n_embd,
                    dropout=config.dropout,
                )
            )
            trainer = Trainer(model=model, token_ids=ids, config=train_config)

        self.model = trainer.model
        self.model.eval()

        def on_step(record: dict[str, Any]) -> None:
            if self.cancel_event.is_set():
                raise KeyboardInterrupt("Training cancelled")

            self.step = int(record["step"])
            self.loss = float(record["loss"])
            self.val_loss = float(record.get("val_loss", 0.0))
            elapsed = time.time() - self.start_time
            ms_per_step = elapsed / self.step * 1000 if self.step > 0 else 0
            self.eta_seconds = (self.total_steps - self.step) * ms_per_step / 1000

            asyncio.run_coroutine_threadsafe(
                self.emit(
                    TrainEvent(
                        event_type="step",
                        step=self.step,
                        loss=self.loss,
                        val_loss=self.val_loss if "val_loss" in record else None,
                        lr=float(record.get("lr", 0.0)),
                        eta_seconds=self.eta_seconds,
                    )
                ),
                asyncio.get_event_loop(),
            )

        trainer.train(on_step=on_step)

        trainer.save_checkpoint(checkpoint_path)
        trainer.save_checkpoint(self.checkpoint_dir / "model.npz", include_optimizer=False)
        (self.checkpoint_dir / "history.json").write_text(json.dumps(trainer.history, indent=2))

        self.status = "completed"
        asyncio.run_coroutine_threadsafe(
            self.emit(
                TrainEvent(
                    event_type="completed",
                    step=self.step,
                    loss=self.loss,
                    val_loss=self.val_loss,
                    message="Training completed successfully",
                )
            ),
            asyncio.get_event_loop(),
        )

    def stop_training(self) -> None:
        """Stop the current training run."""
        if self.is_active():
            self.cancel_event.set()
            self.status = "stopped"


app = FastAPI(title="loom training dashboard", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = TrainingManager()

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/train/start")
async def train_start(config: TrainConfig) -> TrainRunResponse:
    """Start a new training run."""
    if manager.is_active():
        raise HTTPException(status_code=409, detail="Training already in progress. Stop it first.")

    try:
        run_id = await manager.start_training(config)
        return TrainRunResponse(
            run_id=run_id,
            message=f"Training started (run {run_id})",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/train/status")
async def train_status() -> TrainStatus:
    """Get current training status."""
    return TrainStatus(
        run_id=manager.run_id or "none",
        status=manager.status,
        step=manager.step,
        total_steps=manager.total_steps,
        loss=manager.loss if manager.loss > 0 else None,
        val_loss=manager.val_loss if manager.val_loss > 0 else None,
        eta_seconds=manager.eta_seconds if manager.eta_seconds > 0 else None,
    )


@app.post("/api/train/stop")
async def train_stop() -> dict[str, str]:
    """Stop the current training run."""
    if not manager.is_active():
        raise HTTPException(status_code=400, detail="No training in progress")
    manager.stop_training()
    return {"message": "Training stopped"}


@app.websocket("/api/train/stream")
async def train_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming training events."""
    await websocket.accept()
    try:
        while True:
            event = await manager.event_queue.get()
            await websocket.send_json(event.model_dump())
            if event.event_type in ("completed", "failed"):
                break
    except Exception:
        await websocket.close(code=1000)


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate text from the current model."""
    if manager.model is None or manager.tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. Start a training run first.")

    prompt_ids = manager.tokenizer.encode(request.prompt)
    if not prompt_ids:
        prompt_ids = [manager.tokenizer.encode("\n")[0]]

    output = manager.model.generate(
        np.array([prompt_ids]),
        max_new_tokens=request.tokens,
        temperature=request.temperature,
        top_k=request.top_k,
    )

    text = manager.tokenizer.decode([int(i) for i in output[0]])
    return GenerateResponse(text=text)


@app.get("/api/defaults")
async def get_defaults() -> dict[str, Any]:
    """Get default training configuration."""
    defaults = TrainConfig()
    return defaults.model_dump()


def serve() -> None:
    """Run the training dashboard server."""
    import uvicorn

    uvicorn.run(
        "loom.server.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    serve()
