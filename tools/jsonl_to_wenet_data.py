#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert JSONL format to WeNet data.list format.

Input JSONL format: {"audio_filepath": "...", "text": "..."}
Output format: {"key": "...", "wav": "...", "txt": "..."}

Usage:
    python tools/jsonl_to_wenet_data.py --jsonl data.jsonl --out-dir data/
"""
import argparse
import json
from pathlib import Path


def get_value(row, names):
    for name in names:
        value = row.get(name)
        if value:
            return str(value)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_scp = []
    text = []
    data_list = []

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = get_value(row, ["key", "utt_id", "id"]) or f"utt_{idx:08d}"
            wav = get_value(row, ["wav_path_abs", "wav_path", "audio", "path"])
            txt = get_value(row, ["text", "ref", "sentence", "transcript"])
            if not wav or not txt:
                raise ValueError(f"missing wav/text at line {idx + 1}: {line[:200]}")
            wav_scp.append(f"{key} {wav}")
            text.append(f"{key} {txt}")
            data_list.append(json.dumps({"key": key, "wav": wav, "txt": txt}, ensure_ascii=False))

    (out_dir / "wav.scp").write_text("\n".join(wav_scp) + "\n", encoding="utf-8")
    (out_dir / "text").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out_dir / "data.list").write_text("\n".join(data_list) + "\n", encoding="utf-8")
    print(f"wrote {len(data_list)} rows to {out_dir}")


if __name__ == "__main__":
    main()