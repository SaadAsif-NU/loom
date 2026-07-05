"""Render a trained model's attention maps as SVG heatmaps, no dependencies.

Usage:

    python scripts/attention_map.py checkpoints/shakespeare/model.npz \
        checkpoints/shakespeare/tokenizer.json "The king is dead" docs/

Writes ``attention.svg`` and ``attention-dark.svg``: one row of heatmaps,
one per head of the last block. Cell (row t, column s) is how much
attention position t pays to position s; the empty upper triangle is
causality made visible.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from loom.tensor import no_grad
from loom.tokenizer import BPETokenizer
from loom.train import load_model

CELL = 18
GAP = 30
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'

# Sequential blue ramp (steps 100 -> 700). On the light surface low values
# recede toward near-white; on the dark surface the ramp is reversed so low
# values recede toward the dark surface instead.
RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

THEMES = {
    "attention.svg": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#898781", "ramp": RAMP},
    "attention-dark.svg": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#898781",
        "ramp": list(reversed(RAMP)),
    },
}


def _label(token: bytes) -> str:
    text = token.decode("utf-8", "replace").replace("\n", "\\n").replace(" ", "·")
    return text[:6]


def render(maps: np.ndarray, labels: list[str], theme: dict) -> str:
    n_head, seq, _ = maps.shape
    label_w = 58
    head_w = seq * CELL
    width = label_w + n_head * (head_w + GAP)
    height = 46 + seq * CELL + 26

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="Attention maps for {n_head} heads over {seq} tokens">',
        f'<rect width="{width}" height="{height}" rx="8" fill="{theme["surface"]}"/>',
    ]
    for h in range(n_head):
        x0 = label_w + h * (head_w + GAP)
        parts.append(
            f'<text x="{x0 + head_w / 2:.0f}" y="24" text-anchor="middle" font-size="12" '
            f'{FONT} fill="{theme["ink"]}">head {h + 1}</text>'
        )
        for t in range(seq):
            for s in range(seq):
                # sqrt scaling: attention rows are distributions, so late
                # rows are diffuse; linear mapping would wash them out.
                level = int(round(math.sqrt(float(maps[h, t, s])) * (len(theme["ramp"]) - 1)))
                color = theme["ramp"][level]
                parts.append(
                    f'<rect x="{x0 + s * CELL}" y="{40 + t * CELL}" '
                    f'width="{CELL - 1}" height="{CELL - 1}" rx="2" fill="{color}"/>'
                )
        if h == 0:
            for t, label in enumerate(labels):
                parts.append(
                    f'<text x="{label_w - 6}" y="{40 + t * CELL + CELL / 2 + 4:.0f}" '
                    f'text-anchor="end" font-size="10" {FONT} '
                    f'fill="{theme["muted"]}">{label}</text>'
                )
    parts.append(
        f'<text x="{label_w}" y="{height - 8}" font-size="11" {FONT} fill="{theme["muted"]}">'
        f"rows attend to columns; the empty upper triangle is the causal mask</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    checkpoint, tokenizer_path, prompt, out_dir = sys.argv[1:5]
    model, _ = load_model(checkpoint)
    model.eval()
    tokenizer = BPETokenizer.load(tokenizer_path)

    ids = tokenizer.encode(prompt)
    attn = model.blocks[-1].attn
    attn.store_weights = True
    with no_grad():
        model.forward(np.array([ids]))
    assert attn.last_weights is not None
    maps = attn.last_weights[0]  # (H, T, T)
    labels = [_label(tokenizer.token_bytes(i)) for i in ids]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, theme in THEMES.items():
        (out / filename).write_text(render(maps, labels, theme))
        print(f"wrote {out / filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
