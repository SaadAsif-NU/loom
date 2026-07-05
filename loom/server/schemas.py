"""Request/response schemas for the training dashboard API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Training configuration."""

    data_path: str = Field(default="data/shakespeare.txt", description="Path to training data")
    vocab_size: int = Field(default=512, ge=256, le=2048)
    block_size: int = Field(default=128, ge=32, le=512)
    n_layer: int = Field(default=4, ge=1, le=12)
    n_head: int = Field(default=4, ge=1, le=16)
    n_embd: int = Field(default=128, ge=64, le=512)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    steps: int = Field(default=2000, ge=100, le=10000)
    batch_size: int = Field(default=32, ge=1, le=128)
    grad_accum: int = Field(default=1, ge=1, le=8)
    lr: float = Field(default=1e-3, ge=1e-5, le=1e-1)
    warmup_steps: int = Field(default=100, ge=0, le=1000)
    eval_interval: int = Field(default=250, ge=50)
    log_interval: int = Field(default=50, ge=1)
    seed: int = Field(default=42, ge=0)


class TrainRunResponse(BaseModel):
    """Response from starting a training run."""

    run_id: str
    message: str


class TrainStatus(BaseModel):
    """Current training status."""

    run_id: str
    status: Literal["idle", "running", "completed", "failed", "stopped"]
    step: int
    total_steps: int
    loss: float | None
    val_loss: float | None
    eta_seconds: float | None


class TrainEvent(BaseModel):
    """A training event streamed via WebSocket."""

    event_type: Literal["step", "completed", "failed", "info"]
    step: int | None = None
    loss: float | None = None
    val_loss: float | None = None
    lr: float | None = None
    eta_seconds: float | None = None
    message: str | None = None


class GenerateRequest(BaseModel):
    """Request to generate samples from current model."""

    prompt: str = Field(default="\n")
    tokens: int = Field(default=200, ge=50, le=1000)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=40, ge=0, le=100)


class GenerateResponse(BaseModel):
    """Generated text response."""

    text: str
