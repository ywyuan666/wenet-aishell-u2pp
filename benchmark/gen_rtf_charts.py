#!/usr/bin/env python3
"""Generate lightweight RTF & CER comparison charts (multi-image approach)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Data ──
chunks = ["non-streaming", "chunk_32", "chunk_16", "chunk_8", "chunk_4"]
rtfs   = [0.0088, 0.0090, 0.0079, 0.0081, 0.0080]
cers   = [4.61,   4.90,   5.21,   6.45,   7.52]
colors_rtf = ["#1a76c4", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
})

def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"[OK] {path} ({path.stat().st_size // 1024} KB)")
    plt.close(fig)

# ═══════════════════════════════════════════
#  Chart 1: RTF Bar Chart (simple, clean)
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafbfc")
x = np.arange(len(chunks))
bars = ax.bar(x, rtfs, 0.5, color=colors_rtf, alpha=0.85,
              edgecolor="white", lw=1.2, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(["Non-Streaming", "Chunk 32", "Chunk 16", "Chunk 8", "Chunk 4"],
                   fontsize=12, fontweight="bold")
ax.set_ylabel("RTF", fontweight="bold")
ax.set_title("Real-Time Factor per Chunk Size", fontweight="bold", pad=12)
ax.grid(axis="y", alpha=0.3)
ax.grid(axis="x", alpha=0)
ax.set_ylim(0, max(rtfs) * 1.4)
for bar, rtf in zip(bars, rtfs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rtfs)*0.02,
            f"{rtf:.4f}", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="#2c3e50")
ax.axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.3, lw=1.5)
ax.text(x[-1]+0.3, 1.02, "RTF=1.0", fontsize=9, color="#e74c3c", alpha=0.5, fontstyle="italic")
save(fig, "rtf_comparison_new.png")

# ═══════════════════════════════════════════
#  Chart 2: CER Bar Chart (per chunk)
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafbfc")
bars = ax.bar(x, cers, 0.5, color=colors_rtf, alpha=0.85,
              edgecolor="white", lw=1.2, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(["Non-Streaming", "Chunk 32", "Chunk 16", "Chunk 8", "Chunk 4"],
                   fontsize=12, fontweight="bold")
ax.set_ylabel("CER (%)", fontweight="bold")
ax.set_title("CER per Chunk Size (Non-Streaming baseline: 4.61%)", fontweight="bold", pad=12)
ax.grid(axis="y", alpha=0.3)
ax.grid(axis="x", alpha=0)
ax.set_ylim(0, max(cers) * 1.3)
for bar, cer in zip(bars, cers):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cers)*0.02,
            f"{cer:.2f}%", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="#2c3e50")
save(fig, "cer_per_chunk.png")

# ═══════════════════════════════════════════
#  Chart 3: CER vs Latency dual-axis (clean standalone)
# ═══════════════════════════════════════════
latencies = [float("inf"), 1280, 640, 320, 160]
fig, ax1 = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("white")
ax1.set_facecolor("#fafbfc")
x = np.arange(len(chunks))
bars = ax1.bar(x, cers, 0.5, color="#1a76c4", alpha=0.8,
               edgecolor="white", lw=1.2, zorder=3, label="CER (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(["Non-Streaming", "Chunk 32", "Chunk 16", "Chunk 8", "Chunk 4"],
                    fontsize=11, fontweight="bold")
ax1.set_ylabel("CER (%)", fontweight="bold", color="#1a76c4")
ax1.tick_params(axis="y", labelcolor="#1a76c4")
ax1.set_ylim(0, max(cers) * 1.35)
ax1.grid(axis="y", alpha=0.3)
ax1.grid(axis="x", alpha=0)

for bar, cer in zip(bars, cers):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cers)*0.025,
             f"{cer:.2f}%", ha="center", va="bottom", fontsize=10,
             fontweight="bold", color="#1a76c4")

ax2 = ax1.twinx()
ax2.set_facecolor("#fafbfc")
finite_lat = [l for l in latencies if l != float("inf")]
finite_idx = [i for i, l in enumerate(latencies) if l != float("inf")]
ax2.plot(finite_idx, finite_lat, "D-", color="#e74c3c", ms=8, lw=2.5,
         zorder=5, label="Latency (ms)")
ax2.set_ylabel("Latency (ms)", fontweight="bold", color="#e74c3c")
ax2.tick_params(axis="y", labelcolor="#e74c3c")

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left",
           framealpha=0.9, edgecolor="#d0d7de", fontsize=11)
ax1.set_title("CER vs Latency per Chunk Size", fontweight="bold", pad=12)
save(fig, "cer_vs_latency.png")

print("Done — 3 charts generated.")
