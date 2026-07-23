#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a trained WeNet checkpoint to HuggingFace-compatible format.

This script:
  1. Loads a trained WeNet model from checkpoint (or JIT zip)
  2. Extracts model configuration from train.yaml
  3. Wraps it in WenetForASR with save_pretrained() API
  4. Saves config.json + pytorch_model.bin + model card

Usage:
    # From checkpoint
    python scripts/convert_to_hf.py \\
        --checkpoint exp/u2pp_conformer_course/epoch_4.pt \\
        --config exp/u2pp_conformer_course/train.yaml \\
        --output saved_models/wenet-aishell-u2pp

    # From JIT export
    python scripts/convert_to_hf.py \\
        --checkpoint exp/u2pp_conformer_course/final.zip \\
        --output saved_models/wenet-aishell-u2pp-jit

    # Push to HuggingFace Hub (requires huggingface_hub)
    python scripts/convert_to_hf.py \\
        --checkpoint epoch_4.pt --output ./model \\
        --push-to-hub your-username/wenet-aishell-u2pp

    # Verify the saved model
    python scripts/convert_to_hf.py --verify saved_models/wenet-aishell-u2pp
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Default paths (auto-detected from project root) ──
_DEFAULT_S0_DIR = str(_PROJECT_ROOT / "wenet" / "examples" / "aishell" / "s0")
_DEFAULT_CKPT = str(_PROJECT_ROOT / "wenet" / "examples" / "aishell" / "s0" / "exp" / "u2pp_conformer_course" / "epoch_4.pt")
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "wenet" / "examples" / "aishell" / "s0" / "exp" / "u2pp_conformer_course" / "train.yaml")


def load_checkpoint(checkpoint_path: str, config_path: str, wenet_dir: str, s0_dir: str):
    """Load a WeNet model from checkpoint."""
    sys.path.insert(0, wenet_dir)
    os.chdir(s0_dir)

    import torch
    from wenet.utils.init_model import init_model

    # Apply compatibility patches
    if not hasattr(torch.nn.Module, "__annotations__"):
        torch.nn.Module.__annotations__ = {}
    import torchaudio
    import types
    if not hasattr(torchaudio, "sox_effects"):
        torchaudio.sox_effects = types.ModuleType("sox_effects")
        torchaudio.sox_effects.apply_effects_tensor = lambda w, sr, e: (w, sr)

    with open(config_path, encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    class Args:
        pass
    a = Args()
    a.checkpoint = checkpoint_path
    a.config = config_path

    model, _ = init_model(a, configs)
    model.eval()
    return model, configs


def extract_hf_config(configs: dict) -> dict:
    """Extract HuggingFace-compatible config from WeNet YAML config."""
    encoder_conf = configs.get("encoder_conf", {})
    decoder_conf = configs.get("decoder_conf", {})
    model_conf = configs.get("model_conf", {})

    return {
        "model_type": configs.get("encoder", "conformer"),
        "num_blocks": encoder_conf.get("num_blocks", 12),
        "attention_heads": encoder_conf.get("attention_heads", 8),
        "attention_dim": encoder_conf.get("output_size", 512),
        "linear_units": encoder_conf.get("linear_units", 2048),
        "vocab_size": configs.get("output_dim", 4234),
        "num_mel_bins": configs.get("input_dim", 80),
        "subsampling_factor": 4,
        "ctc_weight": model_conf.get("ctc_weight", 0.3),
        "reverse_weight": model_conf.get("reverse_weight", 0.3),
    }


def verify_saved_model(hf_path: str):
    """Verify a saved HuggingFace model loads correctly."""
    sys.path.insert(0, str(_PROJECT_ROOT))

    from wenet_hf_model import WenetForASR

    print(f"\n🔍 Verifying model at: {hf_path}")
    try:
        model = WenetForASR.from_pretrained(hf_path)
        print(f"✅ Model loaded successfully")
        print(f"   Architecture: Conformer U2++")
        print(f"   Vocab size: {model.config.vocab_size}")
        print(f"   Attention dim: {model.config.attention_dim}")
        print(f"   Device: {model.device}")

        # Quick forward pass with dummy input
        import torch
        dummy_fbank = torch.randn(1, 50, model.config.num_mel_bins)
        dummy_len = torch.tensor([50], dtype=torch.long)
        with torch.no_grad():
            out = model._model.encoder(dummy_fbank, dummy_len)
        print(f"✅ Forward pass OK (encoder output shape: {out[0].shape})")

        # Check files
        for fname in ["config.json", "pytorch_model.bin", "README.md"]:
            fpath = Path(hf_path) / fname
            status = "✅" if fpath.exists() else "❌"
            size = fpath.stat().st_size if fpath.exists() else 0
            print(f"   {status} {fname}: {size:,} bytes")

        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def push_to_hub(hf_path: str, repo_id: str, token: Optional[str] = None):
    """Push saved model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub not installed. Install: pip install huggingface_hub")
        return False

    api = HfApi(token=token)
    try:
        create_repo(repo_id, exist_ok=True)
    except Exception:
        pass

    api.upload_folder(
        folder_path=hf_path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[".gitkeep", "__pycache__"],
    )
    print(f"✅ Model pushed to https://huggingface.co/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert WeNet checkpoint to HuggingFace-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument("--checkpoint", default=None,
                        help=f"Checkpoint path (default: {_DEFAULT_CKPT})")
    parser.add_argument("--config", default=None,
                        help=f"train.yaml path (default: auto-detected from checkpoint)")
    parser.add_argument("--wenet-dir", default=None,
                        help="WeNet source directory (auto-detected)")
    parser.add_argument("--s0-dir", default=None,
                        help="AISHELL s0 directory (auto-detected)")

    # Output
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "hf_model"),
                        help="Output directory for HuggingFace model")
    parser.add_argument("--safetensors", action="store_true",
                        help="Use safetensors format (instead of .bin)")

    # Actions
    parser.add_argument("--verify", default=None, nargs="?",
                        const=str(_PROJECT_ROOT / "hf_model"),
                        help="Verify a saved model (path optional, defaults to --output)")
    parser.add_argument("--push-to-hub", default=None,
                        help="Push to HuggingFace Hub (e.g., 'username/wenet-aishell-u2pp')")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")

    args = parser.parse_args()

    # ── Verify mode ──
    if args.verify:
        verify_path = args.verify
        success = verify_saved_model(verify_path)
        sys.exit(0 if success else 1)

    # ── Convert mode ──
    # Resolve paths
    wenet_dir = args.wenet_dir or os.environ.get(
        "WENET_DIR", str(_PROJECT_ROOT / "wenet")
    )
    s0_dir = args.s0_dir or os.environ.get(
        "WENET_S0_DIR", _DEFAULT_S0_DIR
    )

    checkpoint_path = args.checkpoint or os.environ.get(
        "CKPT_PATH", _DEFAULT_CKPT
    )
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Auto-detect config path
    if args.config:
        config_path = args.config
    else:
        # Derive from checkpoint: replace .pt with train.yaml in exp/ dir
        ckpt_dir = Path(checkpoint_path).parent
        config_path = str(ckpt_dir / "train.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(s0_dir, "exp/u2pp_conformer_course/train.yaml")

    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    print(f"📦 Loading checkpoint: {checkpoint_path}")
    print(f"📄 Config: {config_path}")
    print(f"📁 WeNet dir: {wenet_dir}")

    # Load model
    model, configs = load_checkpoint(checkpoint_path, config_path, wenet_dir, s0_dir)

    # Extract config
    hf_config_dict = extract_hf_config(configs)
    print(f"\n📊 Model Configuration:")
    for k, v in hf_config_dict.items():
        print(f"   {k}: {v}")

    # Create HF wrapper
    sys.path.insert(0, str(_PROJECT_ROOT))
    from wenet_hf_model import WenetForASR, WenetModelConfig

    hf_config = WenetModelConfig(**hf_config_dict)
    hf_model = WenetForASR(model, hf_config)

    # Save
    output_path = args.output
    saved_path = hf_model.save_pretrained(output_path, safetensors=args.safetensors)
    print(f"\n✅ Model saved to: {saved_path}")

    # Verify
    verify_saved_model(str(saved_path))

    # Push to Hub
    if args.push_to_hub:
        token = args.token or os.environ.get("HF_TOKEN")
        push_to_hub(str(saved_path), args.push_to_hub, token)


if __name__ == "__main__":
    main()
