#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark visualization script.
Generates publication-quality plots from evaluation results.

Usage:
    python benchmark/plot_results.py                          # plot from results/*.csv
    python benchmark/plot_results.py --output figures         # custom output dir
    python benchmark/plot_results.py --no-show                # save only, no display

Output:
    figures/architecture_comparison.png   - bar chart: decode modes
    figures/streaming_tradeoff.png        - line chart: CER vs latency
    figures/rtf_comparison.png           - bar chart: RTF per config
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe in CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Chart style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})


def load_csv(path: str) -> list[dict]:
    """Load a CSV file and return list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_cer(val):
    """Parse CER value (handle inf, None, empty)."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ─── Streaming Tradeoff: CER vs Latency ───

def plot_streaming_tradeoff(save_dir: Path):
    """CER vs Latency tradeoff curve for different chunk sizes."""
    csv_path = Path("results") / "streaming_tradeoff.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return

    rows = load_csv(str(csv_path))
    labels, cers, latencies = [], [], []
    for r in rows:
        cer = parse_cer(r.get("CER(%)"))
        if cer is None:
            continue
        labels.append(r.get("chunk_size", "unknown"))
        cers.append(cer)
        lat = r.get("latency(ms)", "inf")
        latencies.append(float("inf") if lat in ("inf", "Inf", "∞") else float(lat))

    if len(labels) < 2:
        print("[SKIP] Streaming tradeoff: insufficient data")
        return

    fig, ax1 = plt.subplots()

    # Primary axis: CER vs chunk (bar chart)
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]
    ax1.bar(labels, cers, color=colors[:len(labels)], alpha=0.8, label="CER(%)", zorder=3)
    ax1.set_xlabel("Chunk Size")
    ax1.set_ylabel("CER (%)", color="#2c3e50")
    ax1.set_ylim(0, max(cers) * 1.3 + 1)

    # Annotate values on bars
    for i, (lbl, cer) in enumerate(zip(labels, cers)):
        ax1.annotate(f"{cer:.2f}%", (i, cer), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=10, fontweight="bold",
                     color=colors[i % len(colors)])

    # Secondary axis: latency (line)
    ax2 = ax1.twinx()
    finite_lat = [l for l in latencies if l != float("inf")]
    finite_idx = [i for i, l in enumerate(latencies) if l != float("inf")]
    if finite_lat:
        ax2.plot(finite_idx, finite_lat, "D-", color="#e74c3c", ms=6,
                 label="Latency (ms)", zorder=4)
    ax2.set_ylabel("Latency (ms)", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    # Annotation line: real-time threshold (1000ms)
    if finite_lat:
        ax2.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax2.annotate("Real-time threshold (1s)", xy=(0.5, 1000),
                     xytext=(0.5, 1050), ha="center", fontsize=9, color="gray",
                     textcoords=ax2.get_yaxis_transform())

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Streaming Tradeoff: CER vs Latency per Chunk Size")
    ax1.grid(axis="y", alpha=0.3)

    save_path = save_dir / "streaming_tradeoff.png"
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[SAVED] {save_path}")
    plt.close(fig)


# ─── Architecture Comparison: decoding modes ───

def plot_architecture_comparison(save_dir: Path):
    """Bar chart comparing different decoding modes."""
    csv_path = Path("results") / "architecture_comparison.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return

    rows = load_csv(str(csv_path))

    # Filter non-streaming decode modes
    non_streaming = [r for r in rows
                     if r.get("category") == "decode_mode"
                     or ("nstream" not in r.get("variant", "").lower()
                         and "chunk" not in r.get("variant", "").lower())]

    if not non_streaming:
        # Try alternative: use method column if available
        non_streaming = [r for r in rows if "method" in r]

    labels = [r.get("variant", r.get("method", "unknown")) for r in non_streaming]
    cers = [parse_cer(r.get("CER(%)", r.get("CER(%)", "0"))) for r in non_streaming]
    rtfs = [parse_cer(r.get("RTF", "0")) for r in non_streaming]

    # Filter None
    valid = [(l, c, r) for l, c, r in zip(labels, cers, rtfs) if c is not None]
    if len(valid) < 2:
        print("[SKIP] Architecture comparison: insufficient data")
        return
    labels, cers, rtfs = zip(*valid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: CER comparison
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
    bars = ax1.bar(labels, cers, color=colors[:len(labels)], alpha=0.8, zorder=3)
    ax1.set_ylabel("CER (%)")
    ax1.set_title("Decoding Mode: Word Error Rate")
    ax1.grid(axis="y", alpha=0.3)

    for bar, cer in zip(bars, cers):
        ax1.annotate(f"{cer:.2f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10,
                     fontweight="bold")

    # Right: RTF comparison
    bars2 = ax2.bar(labels, rtfs, color=colors[:len(labels)], alpha=0.8, zorder=3)
    ax2.set_ylabel("RTF (Real-Time Factor)")
    ax2.set_title("Decoding Mode: Inference Speed")
    ax2.grid(axis="y", alpha=0.3)

    for bar, rtf in zip(bars2, rtfs):
        ax2.annotate(f"{rtf:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10,
                     fontweight="bold")

    fig.suptitle("Architecture Comparison: AISHELL-1 Evaluation", fontsize=15, y=1.02)

    save_path = save_dir / "architecture_comparison.png"
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[SAVED] {save_path}")
    plt.close(fig)


# ─── Chunk-size RTF comparison ───

def plot_rtf_comparison(save_dir: Path):
    """Bar chart comparing RTF across chunk sizes."""
    csv_path = Path("results") / "streaming_tradeoff.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return

    rows = load_csv(str(csv_path))
    labels = [r.get("chunk_size", "unknown") for r in rows]
    rtfs = [parse_cer(r.get("RTF", "0")) for r in rows]
    cers = [parse_cer(r.get("CER(%)", "0")) for r in rows]

    valid = [(l, r, c) for l, r, c in zip(labels, rtfs, cers) if r is not None]
    if len(valid) < 2:
        print("[SKIP] RTF comparison: insufficient data")
        return
    labels, rtfs, cers = zip(*valid)

    fig, ax = plt.subplots()
    colors = ["#1abc9c", "#3498db", "#f39c12", "#e74c3c", "#9b59b6"]
    bars = ax.bar(labels, rtfs, color=colors[:len(labels)], alpha=0.8, zorder=3)
    ax.set_ylabel("RTF")
    ax.set_xlabel("Chunk Size")
    ax.set_title("Real-Time Factor per Chunk Size")
    ax.grid(axis="y", alpha=0.3)

    for bar, rtf, cer in zip(bars, rtfs, cers):
        ax.annotate(f"RTF={rtf:.4f}\nCER={cer:.2f}%",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=9, fontweight="bold")

    save_path = save_dir / "rtf_comparison.png"
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[SAVED] {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualization plots")
    parser.add_argument("--output", default="figures", help="Output directory for charts")
    parser.add_argument("--no-show", action="store_true", help="Save only, no display")
    args = parser.parse_args()

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating benchmark visualization plots...")
    print("=" * 50)

    plot_streaming_tradeoff(save_dir)
    plot_architecture_comparison(save_dir)
    plot_rtf_comparison(save_dir)

    print("=" * 50)
    print(f"All plots saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
