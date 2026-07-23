#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse WeNet decoder output (text + cer_result.txt) into a structured summary.

Usage:
    python tools/summarize_wenet_results.py --exp exp/u2pp_conformer_course/dec_test --out summary.md
"""
import argparse
import re
from pathlib import Path


SCORE_RE = re.compile(r"(?:%WER|Overall|CER|wer|WER)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def read_small(path: Path, limit: int = 8000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    exp = Path(args.exp)
    lines = ["# WeNet Result Summary", "", f"exp: `{exp}`", ""]

    if not exp.exists():
        lines.append("Experiment directory does not exist.")
    else:
        lines.append("## Candidate score files")
        score_files = []
        for pat in ["**/wer", "**/cer", "**/*wer*", "**/*cer*", "**/result.txt", "**/scores.txt"]:
            score_files.extend(exp.glob(pat))
        seen = set()
        count = 0
        for path in sorted(score_files):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            text = read_small(path)
            if not text.strip():
                continue
            match = SCORE_RE.search(text)
            value = f" -> `{match.group(1)}`" if match else ""
            try:
                rel = path.relative_to(exp)
            except Exception:
                rel = path
            lines.append(f"- `{rel}`{value}")
            count += 1
            if count >= 80:
                lines.append("- ... more score files omitted")
                break
        if count == 0:
            lines.append("- No score files found yet.")

        lines.extend(["", "## Exported models"])
        exported = []
        for pat in ["**/final.zip", "**/*.jit", "**/avg_*.pt", "**/final.pt"]:
            exported.extend(exp.glob(pat))
        exported = sorted({p for p in exported if p.is_file()})
        for path in exported:
            size_mb = path.stat().st_size / 1024 / 1024
            try:
                rel = path.relative_to(exp)
            except Exception:
                rel = path
            lines.append(f"- `{rel}` ({size_mb:.1f} MB)")
        if not exported:
            lines.append("- No exported model found yet.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()