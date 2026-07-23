#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark visualization script — Publication-quality plots.

Produces three charts suitable for README and interview presentation:
  1. streaming_tradeoff.png     — CER vs Latency dual-axis chart
  2. architecture_comparison.png — Decoding mode comparison (two panels)
  3. rtf_comparison.png          — RTF per chunk-size

Usage:
    python benchmark/plot_results.py                          # default output: figures/
    python benchmark/plot_results.py --output my_charts        # custom output dir
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Premium publication style ──
plt.rcParams.update({
    "figure.dpi": 180,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "axes.titlepad": 14,
    "axes.labelpad": 10,
    "axes.facecolor": "#fafbfc",
    "axes.edgecolor": "#d0d7de",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": "#d0d7de",
    "grid.linestyle": "-",
    "legend.fontsize": 13,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#d0d7de",
    "legend.fancybox": True,
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})


def load_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_val(val, default=None):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ── Color palette (professional, colorblind-friendly) ──
C_PALETTE = ["#1a76c4", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6", "#1abc9c"]
C_CER = "#1a76c4"
C_LATENCY = "#e74c3c"
C_RTF = "#2ecc71"
C_GRID = "#e1e4e8"
C_BG = "#fafbfc"


# ═══════════════════════════════════════════════════════════════
#  Chart 1: Streaming Tradeoff — CER vs Latency
# ═══════════════════════════════════════════════════════════════

def plot_streaming_tradeoff(save_dir: Path):
    csv_path = Path("results") / "streaming_tradeoff.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found"); return

    rows = load_csv(str(csv_path))
    labels, cers, latencies = [], [], []
    for r in rows:
        cer = parse_val(r.get("CER(%)"))
        if cer is None:
            continue
        labels.append(r.get("chunk_size", "unknown"))
        cers.append(cer)
        lat = r.get("latency(ms)", "inf")
        latencies.append(float("inf") if lat in ("inf", "Inf", "∞") else float(lat))

    if len(labels) < 2:
        print("[SKIP] Streaming tradeoff: insufficient data"); return

    # ── Layout: extra-wide for dual-axis ──
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor(C_BG)

    x = np.arange(len(labels))
    bar_width = 0.55

    # CER bars
    bars = ax1.bar(x, cers, bar_width, color=C_CER, alpha=0.85,
                   edgecolor="white", linewidth=1.2, zorder=3, label="CER (%)")

    # Value labels on bars
    for i, (lbl, cer) in enumerate(zip(labels, cers)):
        ax1.text(i, cer + max(cers) * 0.02, f"{cer:.2f}%",
                 ha="center", va="bottom", fontsize=13, fontweight="bold",
                 color=C_CER)

    ax1.set_xlabel("Chunk Size", fontweight="bold")
    ax1.set_ylabel("CER (%)", fontweight="bold", color=C_CER)
    ax1.tick_params(axis="y", labelcolor=C_CER)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(cers) * 1.35 + 0.5)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Latency line (secondary axis)
    ax2 = ax1.twinx()
    ax2.set_facecolor(C_BG)
    finite_lat = [l for l in latencies if l != float("inf")]
    finite_idx = [i for i, l in enumerate(latencies) if l != float("inf")]

    if finite_lat:
        ax2.plot(finite_idx, finite_lat, "D-", color=C_LATENCY,
                 ms=10, lw=3, zorder=5, label="Latency (ms)")
    ax2.set_ylabel("Latency (ms)", fontweight="bold", color=C_LATENCY)
    ax2.tick_params(axis="y", labelcolor=C_LATENCY)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # Real-time threshold line
    if finite_lat:
        ax2.axhline(y=1000, color=C_LATENCY, linestyle="--", alpha=0.4, lw=1.5)
        ax2.text(len(labels) - 0.1, 1020, "Real-time threshold (1s)",
                 ha="right", va="bottom", fontsize=10, color=C_LATENCY, alpha=0.6,
                 fontstyle="italic")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               framealpha=0.9, edgecolor="#d0d7de")

    ax1.set_title("Streaming Tradeoff: CER vs Latency per Chunk Size",
                  fontweight="bold", pad=16)
    ax1.grid(axis="y", alpha=0.3)
    ax1.grid(axis="x", alpha=0)

    fig.subplots_adjust(left=0.1, right=0.88, top=0.92, bottom=0.12)
    save_path = save_dir / "streaming_tradeoff.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=180, facecolor="white")
    print(f"[SAVED] {save_path} ({save_path.stat().st_size // 1024} KB)")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Chart 2: Architecture Comparison — Decoding Modes
# ═══════════════════════════════════════════════════════════════

def plot_architecture_comparison(save_dir: Path):
    csv_path = Path("results") / "architecture_comparison.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found"); return

    rows = load_csv(str(csv_path))

    non_streaming = [r for r in rows
                     if r.get("category") == "decode_mode"
                     or ("nstream" not in r.get("variant", "").lower()
                         and "chunk" not in r.get("variant", "").lower())]
    if not non_streaming:
        non_streaming = [r for r in rows if "method" in r]

    labels = [r.get("variant", r.get("method", "unknown")) for r in non_streaming]
    cers = [parse_val(r.get("CER(%)", "0")) for r in non_streaming]
    rtfs = [parse_val(r.get("RTF", "0")) for r in non_streaming]
    valid = [(l, c, r) for l, c, r in zip(labels, cers, rtfs) if c is not None]
    if len(valid) < 2:
        print("[SKIP] Architecture comparison: insufficient data"); return
    labels, cers, rtfs = zip(*valid)

    # ── Two panels: side-by-side, each wide enough ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    bar_w = 0.5

    # Left panel: CER
    ax1.set_facecolor(C_BG)
    bars1 = ax1.bar(x, cers, bar_w, color=C_PALETTE[:len(labels)],
                    alpha=0.85, edgecolor="white", lw=1.2, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12, rotation=20, ha="right")
    ax1.set_ylabel("CER (%)", fontweight="bold", fontsize=15)
    ax1.set_title("Word Error Rate (lower is better)", fontweight="bold", pad=12)
    ax1.grid(axis="y", alpha=0.3)
    ax1.grid(axis="x", alpha=0)
    ax1.set_ylim(0, max(cers) * 1.25 + 0.5)
    for bar, cer in zip(bars1, cers):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(cers) * 0.015,
                 f"{cer:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold",
                 color=C_PALETTE[0])

    # Right panel: RTF
    ax2.set_facecolor(C_BG)
    bars2 = ax2.bar(x, rtfs, bar_w, color=C_PALETTE[:len(labels)],
                    alpha=0.85, edgecolor="white", lw=1.2, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12, rotation=20, ha="right")
    ax2.set_ylabel("RTF (Real-Time Factor)", fontweight="bold", fontsize=15)
    ax2.set_title("Inference Speed (lower is better)", fontweight="bold", pad=12)
    ax2.grid(axis="y", alpha=0.3)
    ax2.grid(axis="x", alpha=0)
    if any(r is not None for r in rtfs):
        max_rtf = max([r for r in rtfs if r is not None])
        ax2.set_ylim(0, max_rtf * 1.35 + 0.002)
    for bar, rtf in zip(bars2, rtfs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rtfs) * 0.015,
                 f"{rtf:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
                 color=C_PALETTE[1])

    fig.suptitle("Decoding Mode Comparison — AISHELL-1 Evaluation",
                 fontsize=19, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.15, wspace=0.3)
    save_path = save_dir / "architecture_comparison.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=180, facecolor="white")
    print(f"[SAVED] {save_path} ({save_path.stat().st_size // 1024} KB)")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Chart 3: RTF Comparison per Chunk Size
# ═══════════════════════════════════════════════════════════════

def plot_rtf_comparison(save_dir: Path):
    csv_path = Path("results") / "streaming_tradeoff.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found"); return

    rows = load_csv(str(csv_path))
    labels = [r.get("chunk_size", "unknown") for r in rows]
    rtfs = [parse_val(r.get("RTF", "0")) for r in rows]
    cers = [parse_val(r.get("CER(%)", "0")) for r in rows]
    valid = [(l, r, c) for l, r, c in zip(labels, rtfs, cers) if r is not None and r > 0]
    if len(valid) < 2:
        print("[SKIP] RTF comparison: insufficient data (need RTF values > 0)"); return
    labels, rtfs, cers = zip(*valid)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(C_BG)

    x = np.arange(len(labels))
    bar_w = 0.5
    bars = ax.bar(x, rtfs, bar_w, color=C_PALETTE[:len(labels)],
                  alpha=0.85, edgecolor="white", lw=1.2, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold")
    ax.set_xlabel("Chunk Size", fontweight="bold", fontsize=15)
    ax.set_ylabel("RTF (Real-Time Factor)", fontweight="bold", fontsize=15)
    ax.set_title("Real-Time Factor per Chunk Size (lower is faster)",
                 fontweight="bold", pad=14)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0)

    max_rtf = max(rtfs)
    ax.set_ylim(0, max_rtf * 1.5 + 0.002)

    for bar, rtf, cer in zip(bars, rtfs, cers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_rtf * 0.02,
                f"RTF = {rtf:.4f}\nCER = {cer:.2f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color="#2c3e50", linespacing=1.4)

    # Annotate real-time line
    ax.axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.3, lw=1.5)
    ax.text(x[-1] + 0.3, 1.02, "RTF=1.0 (real-time boundary)",
            ha="left", va="bottom", fontsize=10, color="#e74c3c", alpha=0.6,
            fontstyle="italic")

    fig.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.12)
    save_path = save_dir / "rtf_comparison.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=180, facecolor="white")
    print(f"[SAVED] {save_path} ({save_path.stat().st_size // 1024} KB)")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualization plots")
    parser.add_argument("--output", default="figures", help="Output directory")
    parser.add_argument("--no-show", action="store_true", help="Save only, no display")
    args = parser.parse_args()

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating publication-quality benchmark plots...")
    print("=" * 50)
    plot_streaming_tradeoff(save_dir)
    plot_architecture_comparison(save_dir)
    plot_rtf_comparison(save_dir)
    print("=" * 50)
    print(f"All plots saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
