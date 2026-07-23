#!/usr/bin/env python3
"""
Standalone training script for WeNet Conformer U2++ model.
Supports full training, fast training (subset), and checkpoint resumption.

Usage:
    python run_train.py                              # default: CPU fast training
    python run_train.py --epochs 50 --device cpu     # training from scratch
    python run_train.py --resume ./checkpoint.pt     # resume from checkpoint
    python run_train.py --full                       # full training (GPU recommended)

Requirements:
    - AISHELL-1 data prepared (data/train/data.list, data/dev/data.list)
    - torch, torchaudio, soundfile, yaml
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


# ── Paths - adjust for your environment ──
PROJECT_ROOT = Path(__file__).resolve().parent
S0_DIR = PROJECT_ROOT / "wenet" / "examples" / "aishell" / "s0"
WENET_DIR = PROJECT_ROOT / "wenet"
CONFIG_FILE = S0_DIR / "conf" / "train_u2++_conformer.yaml"
DEFAULT_EXP_DIR = S0_DIR / "exp" / "u2pp_conformer_course"


def check_data_exists():
    """Verify required data files exist."""
    train_data = S0_DIR / "data" / "train" / "data.list"
    cv_data = S0_DIR / "data" / "dev" / "data.list"
    missing = []
    if not train_data.exists():
        missing.append(str(train_data))
    if not cv_data.exists():
        missing.append(str(cv_data))
    if missing:
        print("ERROR: Required data files not found:")
        for p in missing:
            print(f"  {p}")
        print("\nRun setup first:")
        print("  bash scripts/02_prepare_aishell.sh  (Linux)")
        print("  or prepare data manually per WeNet docs.")
        sys.exit(1)
    return train_data, cv_data


def main():
    parser = argparse.ArgumentParser(
        description="Train WeNet Conformer U2++ on AISHELL-1"
    )
    parser.add_argument("--config", default=str(CONFIG_FILE),
                        help="Training config YAML path")
    parser.add_argument("--train_data", default=None,
                        help="Train data.list path (default: data/train/data.list)")
    parser.add_argument("--cv_data", default=None,
                        help="CV data.list path (default: data/dev/data.list)")
    parser.add_argument("--model_dir", default=str(DEFAULT_EXP_DIR),
                        help="Output model directory")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint to resume from (epoch_X.pt)")
    parser.add_argument("--device", default="cpu",
                        help='Device: "cpu" or "cuda" (default: cpu)')
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--full", action="store_true",
                        help="Full training mode (360 epochs, GPU recommended)")
    parser.add_argument("--subset", type=int, default=0,
                        help="Use N utterances for fast training (0 = use full train set)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (default: 0 for CPU, 4 for CUDA)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size per GPU (default: config value)")
    args = parser.parse_args()

    # ── Resolve paths ──
    if args.full:
        args.epochs = 360
        if args.device == "cpu":
            print("WARNING: Full training on CPU will be extremely slow. Use --device cuda.")
    if args.train_data is None:
        if args.subset > 0:
            # Use subset
            args.train_data = str(S0_DIR / "data" / "train_subset" / "data.list")
        else:
            check_data_exists()
            args.train_data = str(S0_DIR / "data" / "train" / "data.list")
    if args.cv_data is None:
        args.cv_data = str(S0_DIR / "data" / "dev" / "data.list")

    for f in [args.config, args.train_data, args.cv_data]:
        if not os.path.exists(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)

    num_workers = args.num_workers
    if num_workers == 0 and args.device == "cuda":
        num_workers = 4
        print(f"Auto-set num_workers={num_workers} for CUDA")

    # ── Build command ──
    cmd = [
        sys.executable,
        str(WENET_DIR / "wenet" / "bin" / "train.py"),
        "--config", args.config,
        "--train_data", args.train_data,
        "--cv_data", args.cv_data,
        "--model_dir", args.model_dir,
        "--device", args.device,
        "--num_workers", str(num_workers),
    ]

    if args.resume:
        cmd += ["--checkpoint", args.resume]
    if args.batch_size > 0:
        cmd += ["--batch_size", str(args.batch_size)]

    # Override num_epochs in config
    cmd += ["--override_config", f"num_epochs={args.epochs}"]

    # ── Print info ──
    print("=" * 60)
    print("WeNet Conformer U2++ Training")
    print("=" * 60)
    print(f"  Config:     {args.config}")
    print(f"  Train data: {args.train_data}")
    print(f"  CV data:    {args.cv_data}")
    print(f"  Model dir:  {args.model_dir}")
    print(f"  Device:     {args.device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Workers:    {num_workers}")
    if args.resume:
        print(f"  Resume:     {args.resume}")
    print(f"  Full cmd:   {' '.join(cmd)}")
    print("=" * 60)

    # ── Run training ──
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WENET_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"

    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=str(S0_DIR), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
