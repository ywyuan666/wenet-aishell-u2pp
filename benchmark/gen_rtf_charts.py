#!/usr/bin/env python3
"""Generate individual metric cards for each streaming config — one image per chunk."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Data ──
configs = [
    ("Non-Streaming",  "∞",  4.61, 0.0088, "#1a76c4"),
    ("Chunk 32",       "1280", 4.90, 0.0090, "#2ecc71"),
    ("Chunk 16",       "640",  5.21, 0.0079, "#e67e22"),
    ("Chunk 8",        "320",  6.45, 0.0081, "#9b59b6"),
    ("Chunk 4",        "160",  7.52, 0.0080, "#e74c3c"),
]

def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    print(f"[OK] {path} ({path.stat().st_size // 1024} KB)")
    plt.close(fig)

# ── One compact card per config ──
for name, lat, cer, rtf, color in configs:
    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title bar
    ax.add_patch(plt.Rectangle((0, 0.82), 1, 0.18, color=color, alpha=0.85,
                                transform=ax.transData, zorder=2))
    ax.text(0.5, 0.91, name, ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", transform=ax.transData)

    # Metrics
    lines = [
        ("CER", f"{cer:.2f}%", "#2c3e50"),
        ("Latency", lat + " ms", "#2c3e50"),
        ("RTF", f"{rtf:.4f}", "#2c3e50"),
    ]
    for i, (label, value, c) in enumerate(lines):
        y_pos = 0.60 - i * 0.22
        ax.text(0.12, y_pos, label, ha="left", va="center", fontsize=9,
                color="#7f8c8d", transform=ax.transData)
        ax.text(0.88, y_pos, value, ha="right", va="center", fontsize=11,
                fontweight="bold", color=c, transform=ax.transData)
        # Divider
        if i < len(lines) - 1:
            ax.axhline(y=y_pos - 0.09, xmin=0.08, xmax=0.92,
                       color="#e0e0e0", lw=0.5)

    # RTF bar (small indicator at bottom)
    bar_max = 0.010
    bar_width = 0.76
    bar_x = 0.12
    bar_y = 0.04
    ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, 0.08,
                                color="#e8e8e8", ec="none", lw=0,
                                transform=ax.transData, zorder=1))
    ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * rtf / bar_max, 0.08,
                                color=color, alpha=0.8, ec="none", lw=0,
                                transform=ax.transData, zorder=2))
    ax.text(bar_x + bar_width / 2, bar_y + 0.04, f"RTF bar (max={bar_max})",
            ha="center", va="center", fontsize=6.5, color="#95a5a6",
            transform=ax.transData)

    # Tag
    ax.text(0.5, -0.08, "All RTF ≪ 1.0 — real-time capable",
            ha="center", va="top", fontsize=7, color="#b0b0b0",
            fontstyle="italic", transform=ax.transData)

    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    save(fig, f"card_{safe_name}.png")

print("Done — 5 metric cards generated.")
