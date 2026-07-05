"""Render training history to SVG loss curves (light + dark), no dependencies.

Usage:

    python scripts/plot_history.py checkpoints/shakespeare/history.json docs/

Writes ``loss-curve.svg`` and ``loss-curve-dark.svg`` for the README's
``<picture>`` element. Hand-rolled SVG keeps the repo free of plotting
dependencies and the output byte-for-byte reproducible.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

WIDTH, HEIGHT = 720, 340
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 52, 20, 46, 40

THEMES = {
    "loss-curve.svg": {
        "surface": "#fcfcfb",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "muted": "#898781",
        "ink": "#0b0b0b",
        "train": "#2a78d6",
        "val": "#1baf7a",
    },
    "loss-curve-dark.svg": {
        "surface": "#1a1a19",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "muted": "#898781",
        "ink": "#ffffff",
        "train": "#3987e5",
        "val": "#199e70",
    },
}

FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'


def render(history: list[dict[str, float]], theme: dict[str, str]) -> str:
    train = [(r["step"], r["loss"]) for r in history]
    val = [(r["step"], r["val_loss"]) for r in history if "val_loss" in r]

    max_step = max(s for s, _ in train)
    max_loss = math.ceil(max(loss for _, loss in train))
    plot_w = WIDTH - MARGIN_L - MARGIN_R
    plot_h = HEIGHT - MARGIN_T - MARGIN_B

    def sx(step: float) -> float:
        return MARGIN_L + step / max_step * plot_w

    def sy(loss: float) -> float:
        return MARGIN_T + (1.0 - loss / max_loss) * plot_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-label="Training and validation loss over {int(max_step)} steps">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="8" fill="{theme["surface"]}"/>',
    ]

    # horizontal gridlines + y ticks at whole-loss values
    for tick in range(1, max_loss + 1):
        y = sy(tick)
        parts.append(
            f'<line x1="{MARGIN_L}" y1="{y:.1f}" x2="{WIDTH - MARGIN_R}" y2="{y:.1f}" '
            f'stroke="{theme["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{MARGIN_L - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'{FONT} fill="{theme["muted"]}">{tick}</text>'
        )

    # baseline + x ticks every 500 steps
    base_y = sy(0)
    parts.append(
        f'<line x1="{MARGIN_L}" y1="{base_y:.1f}" x2="{WIDTH - MARGIN_R}" y2="{base_y:.1f}" '
        f'stroke="{theme["baseline"]}" stroke-width="1"/>'
    )
    for step in range(0, int(max_step) + 1, 500):
        parts.append(
            f'<text x="{sx(step):.1f}" y="{base_y + 18:.1f}" text-anchor="middle" '
            f'font-size="11" {FONT} fill="{theme["muted"]}">{step:,}</text>'
        )

    # axis captions
    parts.append(
        f'<text x="{MARGIN_L}" y="{HEIGHT - 6}" font-size="11" {FONT} '
        f'fill="{theme["muted"]}">step</text>'
    )
    parts.append(
        f'<text x="{MARGIN_L - 8}" y="{MARGIN_T - 10}" text-anchor="end" font-size="11" '
        f'{FONT} fill="{theme["muted"]}">loss</text>'
    )

    # train series: 2px line
    points = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in train)
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="{theme["train"]}" '
        f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    # val series: line + ringed markers (sparse: evaluated every 250 steps)
    if val:
        vpoints = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in val)
        parts.append(
            f'<polyline points="{vpoints}" fill="none" stroke="{theme["val"]}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for s, v in val:
            parts.append(
                f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="4" fill="{theme["val"]}" '
                f'stroke="{theme["surface"]}" stroke-width="2"/>'
            )

    # legend (identity never rides color alone: chip + ink label)
    legend = [
        (theme["train"], "train loss"),
        (theme["val"], "val loss (every 250 steps)"),
    ]
    x = MARGIN_L
    for color, label in legend:
        parts.append(
            f'<line x1="{x}" y1="20" x2="{x + 16}" y2="20" stroke="{color}" '
            f'stroke-width="3" stroke-linecap="round"/>'
        )
        x += 22
        parts.append(
            f'<text x="{x}" y="24" font-size="12" {FONT} fill="{theme["ink"]}">{label}</text>'
        )
        x += 7.2 * len(label) + 24

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    history_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    history = json.loads(history_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, theme in THEMES.items():
        (out_dir / filename).write_text(render(history, theme))
        print(f"wrote {out_dir / filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
