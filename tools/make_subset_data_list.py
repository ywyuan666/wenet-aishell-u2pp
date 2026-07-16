#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source WeNet data.list")
    parser.add_argument("--output", required=True, help="Output subset data.list")
    parser.add_argument("--num", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.num > 0 and len(lines) > args.num:
        rng = random.Random(args.seed)
        lines = rng.sample(lines, args.num)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} utterances to {dst}")


if __name__ == "__main__":
    main()