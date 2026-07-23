#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone evaluation script for WeNet Conformer U2++ model.
Runs on the full AISHELL-1 test set (or a subset) and computes CER for
multiple decoding strategies: greedy, prefix beam, attention rescoring.

Usage:
    python run_eval.py                    # full test set eval
    python run_eval.py --subset 20        # eval on first N utterances
    python run_eval.py --chunk 16         # streaming eval with chunk=16
    python run_eval.py --output results/  # custom output dir

Requirements:
    - A trained model checkpoint or JIT model
    - AISHELL-1 test data
    - torch, soundfile, torchaudio
"""
import os
import sys
import json
import time
import argparse
import csv
import traceback
from pathlib import Path

# Paths - auto-detect or env var override
WENET_DIR = os.environ.get("WENET_DIR", r"D:/wenet/wenet")
S0_DIR = os.environ.get("WENET_S0_DIR", os.path.join(WENET_DIR, "examples/aishell/s0"))
CKPT_PATH = os.path.join(S0_DIR, "exp/u2pp_conformer_course/epoch_4.pt")

# ── Compatibility patches for WeNet on Python 3.14 / torch 2.9 ──
# Must be done before any WeNet import
def apply_wenet_patches():
    import torch
    import torchaudio
    import types

    # Patch torchaudio.sox_effects (removed in newer torchaudio)
    if not hasattr(torchaudio, "sox_effects"):
        torchaudio.sox_effects = types.ModuleType("sox_effects")
        torchaudio.sox_effects.apply_effects_tensor = lambda w, sr, e: (w, sr)

    # Patch torch.jit._check for Python 3.14 compatibility
    if not hasattr(torch.jit._check, "AttributeTypeIsSupportedChecker"):
        torch.nn.Module.__annotations__ = {}

    # Fix prefetch_factor when num_workers=0
    import torch.utils.data.dataloader as dl
    _orig_init = dl.DataLoader.__init__
    def _patched_init(self, dataset, **kwargs):
        if kwargs.get("num_workers", 0) == 0 and "prefetch_factor" in kwargs:
            kwargs.pop("prefetch_factor", None)
        _orig_init(self, dataset, **kwargs)
    dl.DataLoader.__init__ = _patched_init


def load_model(s0_dir: str, ckpt_path: str | None = None):
    apply_wenet_patches()

    import torch
    import yaml
    sys.path.insert(0, WENET_DIR)
    os.chdir(s0_dir)

    if ckpt_path is None:
        ckpt_path = CKPT_PATH

    from wenet.utils.init_model import init_model

    class Args:
        pass
    a = Args()
    a.checkpoint = ckpt_path
    a.config = os.path.join(s0_dir, "exp/u2pp_conformer_course/train.yaml")

    with open(a.config, encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    model, configs = init_model(a, configs)
    model.eval()
    return model


def extract_fbank(wav_path: str) -> tuple:
    import torch
    import soundfile
    from torchaudio.compliance import kaldi

    wav, sr = soundfile.read(wav_path, dtype="float32")
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    if sr != 16000:
        import torchaudio.transforms as T
        wav_t = T.Resample(sr, 16000)(wav_t)
    fb = kaldi.fbank(wav_t, num_mel_bins=80, sample_frequency=16000,
                     frame_shift=10, frame_length=25, dither=0.1)
    return fb.unsqueeze(0), torch.tensor([fb.shape[0]], dtype=torch.long)


def load_test_items(s0_dir: str, subset: int | None = None) -> list:
    data_list = os.path.join(s0_dir, "data/test/data.list")
    if not os.path.exists(data_list):
        print(f"ERROR: test data list not found: {data_list}")
        sys.exit(1)

    items = []
    with open(data_list, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    if subset and subset < len(items):
        items = items[:subset]

    print(f"Loaded {len(items)} test utterances")
    return items


def sum_compute_chars(items):
    """Sum total character count across all test items."""
    return sum(len(item.get("txt", "")) for item in items)


def precompute_fbanks(items: list) -> list:
    """Pre-extract fbank features for all test items to avoid repeated WAV reading."""
    print(f"Pre-extracting fbank features for {len(items)} utterances...", end=" ", flush=True)
    fb_cache = []
    for i, item in enumerate(items):
        fb, fb_len = extract_fbank(item["wav"])
        fb_cache.append((fb, fb_len, item["txt"]))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}...", end="", flush=True)
    print(" done")
    return fb_cache


def compute_cer(hyp_tokens: list[str], ref_text: str) -> tuple[int, int]:
    """Compute character error rate between hypothesis and reference."""
    # Token-level comparison using edit distance
    m, n = len(ref_text), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_text[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n], m


def eval_model(items: list, model, method: str, beam_size: int = 1, chunk_size: int = -1,
               ctc_weight: float = 0.3, reverse_weight: float = 0.3, verbose: bool = False,
               fb_cache: list | None = None):
    """
    Run evaluation with a specific decoding configuration.
    Returns average CER and total time.

    If fb_cache is provided (pre-extracted fbank features), it's used instead
    of calling extract_fbank() for each utterance.
    """
    import torch

    total_errors = 0
    total_chars = 0
    total_frames = 0
    total_time = 0.0
    skip_count = 0

    source = fb_cache if fb_cache else items
    for i, entry in enumerate(source):
        try:
            if fb_cache:
                fb, fb_len, ref_txt = entry
            else:
                fb, fb_len = extract_fbank(entry["wav"])
                ref_txt = entry.get("txt", "") or entry.get("key", "")

            t0 = time.time()
            with torch.no_grad():
                result = model.decode(
                    speech=fb,
                    speech_lengths=fb_len,
                    decoding_chunk_size=chunk_size,
                    beam_size=beam_size,
                    ctc_weight=ctc_weight,
                    reverse_weight=reverse_weight,
                    methods=[method],
                )
            total_frames += fb.shape[1]
            elapsed = time.time() - t0
            total_time += elapsed

            # Get hypothesis
            hyp = result[method][0].tokens

            errors, n = compute_cer(hyp, ref_txt)
            total_errors += min(errors, n)
            total_chars += n

            if verbose and i < 5:
                cer_i = errors / n * 100 if n > 0 else 0
                print(f"  [{i}] ref='{ref_txt[:30]}' hyp='{hyp[:30]}' CER={cer_i:.1f}%")

            if (i + 1) % 10 == 0:
                cer_sofar = total_errors / total_chars * 100 if total_chars > 0 else 0
                print(f"  [{i + 1}/{len(items)}] CER so far: {cer_sofar:.2f}%")

        except Exception as e:
            skip_count += 1
            if skip_count <= 3:
                print(f"  [{i}] SKIP: {e}")

    if skip_count:
        print(f"  Skipped {skip_count}/{len(items)} utterances due to errors")

    avg_cer = total_errors / total_chars * 100 if total_chars > 0 else 0
    audio_duration = total_frames * 0.01  # frame_shift=10ms
    avg_rtf = total_time / audio_duration if audio_duration > 0 else 0

    return avg_cer, total_time, avg_rtf


def main():
    parser = argparse.ArgumentParser(description="Evaluate WeNet Conformer U2++ on AISHELL-1")
    parser.add_argument("--subset", type=int, default=None, help="Number of test utterances")
    parser.add_argument("--s0_dir", default=S0_DIR, help="AISHELL s0 directory")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print per-utterance results")
    parser.add_argument("--checkpoint", default=None,
                        help="Use checkpoint instead of JIT model. Default: JIT first, fallback to checkpoint.")
    parser.add_argument("--chunk", type=int, default=None,
                        help="Override decoding_chunk_size for all configs. E.g. 16 for streaming.")
    args = parser.parse_args()

    # Load model
    ckpt_to_load = args.checkpoint or CKPT_PATH
    if not os.path.exists(ckpt_to_load):
        print(f"ERROR: No model found at {ckpt_to_load}")
        print("Train a model first: scripts/03_train_full.sh or scripts/04_train_course_fast.sh")
        sys.exit(1)

    print(f"Loading model from checkpoint: {ckpt_to_load}")
    model = load_model(args.s0_dir, ckpt_to_load)
    print("Model loaded successfully")

    # Load test data
    items = load_test_items(args.s0_dir, args.subset)

    # Pre-extract fbank features once (avoids 6× repeated WAV reading across configs)
    fb_cache = precompute_fbanks(items)

    # Evaluation configs
    eval_configs = [
        # (method, beam_size, chunk_size, ctc_weight, reverse_weight, label)
        ("ctc_greedy_search", 1, -1, 0.3, 0.3, "CTC Greedy (non-streaming)"),
        ("ctc_prefix_beam_search", 5, -1, 0.3, 0.3, "CTC Prefix Beam (non-streaming)"),
        ("attention_rescoring", 5, -1, 0.3, 0.3, "Attention Rescoring (non-streaming)"),
        ("ctc_greedy_search", 1, 16, 0.3, 0.3, "CTC Greedy (chunk=16)"),
        ("ctc_greedy_search", 1, 8, 0.3, 0.3, "CTC Greedy (chunk=8)"),
        ("ctc_greedy_search", 1, 4, 0.3, 0.3, "CTC Greedy (chunk=4)"),
    ]

    print(f"\n{'=' * 70}")
    print(f"Decoding Evaluation: {len(items)} utterances")
    print(f"{'=' * 70}")

    if args.chunk is not None:
        print(f"  Overriding chunk_size to {args.chunk} for all configs")
        eval_configs = [
            (m, b, args.chunk, cw, rw, f"{lbl} (chunk={args.chunk})")
            for m, b, _, cw, rw, lbl in eval_configs
        ]

    results = []
    print(f"\n{'Config':>35}  {'CER(%)':>8}  {'RTF':>8}  {'Time(s)':>8}  {'#Chars':>8}")
    print(f"{'-' * 35}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for method, beam, chunk, ctc_w, rev_w, label in eval_configs:
        print(f"  Running {label}...", end=" ", flush=True)
        try:
            cer, total_time, avg_rtf = eval_model(
                items, model, method, beam, chunk, ctc_w, rev_w,
                verbose=args.verbose, fb_cache=fb_cache
            )
            print(f"CER={cer:.2f}%")
            results.append({
                "method": label,
                "config": f"method={method}, beam={beam}, chunk={chunk}, ctc_weight={ctc_w}",
                "CER(%)": round(cer, 2),
                "RTF": round(avg_rtf, 4),
                "total_time(s)": round(total_time, 2),
                "utterances": len(items),
                "total_chars": sum_compute_chars(items),
            })
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"Results Summary")
    print(f"{'=' * 70}")
    print(f"{'Method':>35}  {'CER(%)':>8}  {'RTF':>8}")
    print(f"{'-' * 35}  {'-' * 8}  {'-' * 8}")
    for r in results:
        print(f"{r['method']:>35}  {r['CER(%)']:>8.2f}  {r['RTF']:>8.4f}")

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    # CSV
    csv_path = out_dir / "architecture_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "config", "CER(%)", "RTF", "total_time(s)", "utterances", "total_chars"])
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Markdown
    md_path = out_dir / "architecture_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Architecture Comparison Results\n\n")
        f.write(f"*Evaluated on {len(items)} AISHELL-1 test utterances*\n\n")
        f.write("| Method | CER(%) | RTF | Config |\n")
        f.write("|--------|--------|-----|--------|\n")
        for r in results:
            f.write(f"| {r['method']} | {r['CER(%)']} | {r['RTF']} | {r['config']} |\n")
    print(f"Results saved to {md_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
