#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-quality WeNet Conformer U2++ architecture diagram.
Output: figures/architecture_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Larger canvas, more breathing room ──
fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")
ax.set_facecolor("white")

C = {
    "input":   "#43a047",
    "fbank":   "#66bb6a",
    "sub":     "#81c784",
    "encoder": "#1976d2",
    "block":   "#42a5f5",
    "ctc":     "#ef6c00",
    "decoder": "#8e24aa",
    "logits":  "#e53935",
    "rerank":  "#00897b",
    "final":   "#2e7d32",
    "arrow":   "#90a4ae",
    "text":    "#37474f",
}


def box(ax, x, y, w, h, color, label, sub="",
        fs=13, subfs=9, alpha=0.92, lw=2):
    """Draw a prettier rounded rectangle."""
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.18",
                        facecolor=color, edgecolor="white",
                        linewidth=lw, alpha=alpha)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2 + (0.08 if sub else 0),
            label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.38, sub,
                ha="center", va="center",
                fontsize=subfs, color="white", alpha=0.9)
    return (x + w / 2, y + h)


def arrow(ax, x1, y1, x2, y2, rad=0, lw=2.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=C["arrow"],
                                lw=lw, connectionstyle=f"arc3,rad={rad}"))


# ── Title ──
ax.text(9, 10.6, "WeNet Conformer U2++  Architecture",
        ha="center", va="center", fontsize=28, fontweight="bold", color=C["text"])
ax.text(9, 10.1, "Two-Pass Streaming + Non-Streaming Unified End-to-End ASR on AISHELL-1",
        ha="center", va="center", fontsize=15, color="#90a4ae", fontstyle="italic")

# ── Layer 1: Audio Input ──
box(ax, 7.2, 9.2, 3.8, 0.65, C["input"],
    "Audio Input", "16 kHz WAV", fs=16)
arrow(ax, 9.1, 9.2, 9.1, 8.7)

# ── Layer 2: Feature Extraction ──
box(ax, 4.8, 8.0, 8.6, 0.6, C["fbank"],
    "Feature Extraction", "80-dim Fbank + CMVN + SpecAugment", fs=15)
arrow(ax, 9.1, 8.0, 9.1, 7.5)

# ── Layer 3: Subsampling ──
box(ax, 5.8, 7.0, 6.6, 0.45, C["sub"],
    "Conv2D Subsampling (4x downsampling)", "", fs=14)

# ── Encoder label ──
ax.text(1.0, 6.0, "Encoder", fontsize=20, fontweight="bold",
        color=C["encoder"], va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd",
                  edgecolor=C["encoder"], lw=1.5))
ax.text(1.0, 5.2, "12 layers of:", fontsize=13, color=C["text"], va="center")
ax.text(1.0, 4.7, "CNN +", fontsize=13, color=C["block"], va="center", fontweight="bold")
ax.text(1.0, 4.3, "Self-Attn", fontsize=13, color=C["block"], va="center", fontweight="bold")
ax.text(1.0, 3.9, "+ FFN", fontsize=13, color=C["block"], va="center", fontweight="bold")
ax.text(1.0, 3.3, "Relative Pos.", fontsize=11, color="#78909c", va="center", fontstyle="italic")

# ── Layer 4: Conformer Encoder ──
e_y, e_h, e_w = 4.4, 1.8, 10.5
e_x = 2.5
box(ax, e_x, e_y, e_w, e_h, C["encoder"],
    "Conformer Encoder  —  12 Blocks", "", fs=16, alpha=0.88)

# Draw 12 individual blocks with more spacing
for i in range(12):
    bx = e_x + 0.3 + i * (e_w - 0.6) / 12
    bw = (e_w - 0.6) / 12 - 0.08
    b = FancyBboxPatch((bx, e_y + 0.25), bw, e_h - 0.5,
                        boxstyle="round,pad=0.06",
                        facecolor=C["block"], edgecolor="white",
                        linewidth=0.8, alpha=0.85)
    ax.add_patch(b)
    ax.text(bx + bw / 2, e_y + e_h / 2, str(i + 1),
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")

arrow(ax, 9.1, e_y + e_h - 0.1, 9.1, 4.5)

# ── Split labels ──
ax.text(4.5, 3.9, "1st Pass (Streaming)", ha="center", fontsize=14,
        fontweight="bold", color=C["ctc"])
ax.text(13.5, 3.9, "2nd Pass (High-Precision)", ha="center", fontsize=14,
        fontweight="bold", color=C["decoder"])

# ── Layer 5a: CTC Greedy ──
box(ax, 2.8, 3.1, 3.4, 0.65, C["ctc"],
    "1st Pass: CTC Greedy", "Low-latency streaming decode", fs=14)

# ── Layer 5b: Bi-Transformer Decoder ──
box(ax, 11.8, 3.1, 3.4, 0.65, C["decoder"],
    "Bi-Transformer Decoder", "Cross-Attention  x3+3", fs=14)

# Arrows split
arrow(ax, 9.1, 4.5, 4.5, 3.75, rad=-0.25)
arrow(ax, 9.1, 4.5, 13.5, 3.75, rad=0.25)

# ── Layer 6: CTC Logits ──
box(ax, 2.8, 1.8, 3.4, 0.55, C["logits"],
    "CTC Logits (4234 tokens)", "blank-aware, vocabulary", fs=13, subfs=10)
arrow(ax, 4.5, 3.1, 4.5, 2.35)

# ── Arrow from Decoder to Rescoring ──
arrow(ax, 13.5, 3.1, 13.5, 2.4, rad=0)

# ── Layer 7: Attention Rescoring ──
box(ax, 6.2, 1.2, 6.0, 0.75, C["rerank"],
    "Attention Rescoring  (2nd Pass)", "Re-rank CTC candidates for best result",
    fs=15, subfs=10)
arrow(ax, 4.5, 1.8, 9.2, 1.95, rad=0.3)
arrow(ax, 13.5, 2.4, 9.2, 1.95, rad=-0.3)

# ── Final Output ──
final = FancyBboxPatch((6.5, 0.35), 5.2, 0.65,
                        boxstyle="round,pad=0.12",
                        facecolor=C["final"], edgecolor="white", linewidth=3)
ax.add_patch(final)
ax.text(9.1, 0.675, "Final Recognition Text",
        ha="center", va="center", fontsize=18, fontweight="bold", color="white")
arrow(ax, 9.2, 1.9, 9.1, 1.0)

# ── Legend ──
legend_items = [
    ("#43a047", "Input"),
    ("#66bb6a", "Feature"),
    ("#1976d2", "Encoder"),
    ("#ef6c00", "CTC/Stream"),
    ("#8e24aa", "Decoder"),
    ("#e53935", "Logits"),
    ("#00897b", "Rescoring"),
    ("#2e7d32", "Output"),
]
for i, (c, l) in enumerate(legend_items):
    x0 = 1.2 + i * 1.85
    b = FancyBboxPatch((x0, 0.05), 0.35, 0.25,
                        boxstyle="round,pad=0.03",
                        facecolor=c, edgecolor="white", lw=1)
    ax.add_patch(b)
    ax.text(x0 + 0.45, 0.175, l, ha="left", va="center", fontsize=10,
            fontweight="bold", color=c)

# ── Dataset badge ──
ax.text(17.2, 0.25, "AISHELL-1", ha="right", va="center",
        fontsize=12, fontweight="bold", color="#78909c",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eceff1",
                  edgecolor="#cfd8dc", lw=1))
ax.text(17.2, 0.05, "CER 4.61%", ha="right", va="center",
        fontsize=11, color="#43a047", fontweight="bold")

plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.06)
fig.savefig("figures/architecture_diagram.png", dpi=180, facecolor="white")
print("Generated: figures/architecture_diagram.png  (18x11, DPI=180)")
plt.close(fig)
