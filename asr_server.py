#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming ASR inference server with FastAPI + WebSocket.
Supports real-time chunk-by-chunk decoding with U2++ dynamic chunk mechanism.

Usage:
    # Start server (requires checkpoint + AISHELL test data)
    python asr_server.py --port 8765

    # Test with WebSocket client
    python asr_server.py --mode client --wav test.wav

    # Quick health check
    curl http://localhost:8765/health

Requirements:
    pip install fastapi uvicorn websockets
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── Path setup ──
_PROJECT_ROOT = Path(__file__).resolve().parent
WENET_DIR = os.environ.get("WENET_DIR", str(_PROJECT_ROOT / "wenet"))
S0_DIR = os.environ.get("WENET_S0_DIR", str(_PROJECT_ROOT / "wenet" / "examples" / "aishell" / "s0"))
CKPT_PATH = os.environ.get("CKPT_PATH", str(_PROJECT_ROOT / "wenet" / "examples" / "aishell" / "s0" / "exp" / "u2pp_conformer_course" / "epoch_4.pt"))


@dataclass
class ASRConfig:
    """Inference configuration, all fields have sensible defaults."""
    model_path: str = CKPT_PATH
    s0_dir: str = S0_DIR
    wenet_dir: str = WENET_DIR
    chunk_size: int = 16  # streaming chunk size
    beam_size: int = 5
    ctc_weight: float = 0.3
    reverse_weight: float = 0.3
    sample_rate: int = 16000
    num_mel_bins: int = 80
    subsampling_factor: int = 4  # 4x encoder subsampling
    frame_shift_ms: int = 10

    @property
    def latency_per_chunk_ms(self) -> int:
        """Theoretical minimum latency per chunk."""
        return self.chunk_size * self.subsampling_factor * self.frame_shift_ms


class StreamingASR:
    """
    Streaming ASR engine wrapping WeNet model.
    Handles model loading, fbank extraction, and chunk-by-chunk decoding.
    """

    def __init__(self, config: ASRConfig):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load WeNet model from checkpoint."""
        sys.path.insert(0, self.config.wenet_dir)
        os.chdir(self.config.s0_dir)

        import torch
        import yaml

        # Compatibility patches
        if not hasattr(torch.nn.Module, "__annotations__"):
            torch.nn.Module.__annotations__ = {}
        import torchaudio
        import types
        if not hasattr(torchaudio, "sox_effects"):
            torchaudio.sox_effects = types.ModuleType("sox_effects")
            torchaudio.sox_effects.apply_effects_tensor = lambda w, sr, e: (w, sr)

        from wenet.utils.init_model import init_model
        from torchaudio.compliance import kaldi

        self.torch = torch
        self.kaldi = kaldi
        self.init_model = init_model

        config_path = self.config.model_path.replace("epoch_4.pt", "train.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(self.config.s0_dir, "exp/u2pp_conformer_course/train.yaml")

        with open(config_path, encoding="utf-8") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        class Args:
            pass
        a = Args()
        a.checkpoint = self.config.model_path
        a.config = config_path
        a.jit = True

        self.model, self.configs = init_model(a, configs)
        self.model.eval()
        print(f"[ASR] Model loaded from {self.config.model_path}")

    def extract_fbank(self, audio: np.ndarray, sample_rate: int):
        """Extract 80-dim fbank features from raw audio."""
        import torchaudio.transforms as T

        wav_t = torch.from_numpy(audio).unsqueeze(0).float()
        if sample_rate != 16000:
            wav_t = T.Resample(sample_rate, 16000)(wav_t)

        fb = self.kaldi.fbank(
            wav_t, num_mel_bins=80,
            sample_frequency=16000,
            frame_shift=10, frame_length=25, dither=0.1,
        )
        return fb.unsqueeze(0), torch.tensor([fb.shape[0]], dtype=torch.long)

    def decode(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Full-utterance decode (non-streaming mode for comparison baseline).

        Returns:
            dict with keys: text, cer, rtf, confidence
        """
        fb, fb_len = self.extract_fbank(audio, sample_rate)

        t0 = time.time()
        with torch.no_grad():
            result = self.model.decode(
                methods=["attention_rescoring", "ctc_greedy_search"],
                speech=fb, speech_lengths=fb_len,
                beam_size=self.config.beam_size,
                decoding_chunk_size=-1,
                ctc_weight=self.config.ctc_weight,
                reverse_weight=self.config.reverse_weight,
            )
        elapsed = time.time() - t0

        hyp_tokens = result["attention_rescoring"][0].tokens
        text = "".join(hyp_tokens)
        audio_duration = fb.shape[1] * 0.01
        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        return {
            "text": text,
            "tokens": hyp_tokens,
            "rtf": round(rtf, 4),
            "time_ms": round(elapsed * 1000, 1),
            "method": "attention_rescoring",
        }

    def decode_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Streaming decode with fixed chunk size.

        Returns:
            dict with partial transcription and timing info.
        """
        fb, fb_len = self.extract_fbank(audio_chunk, 16000)

        t0 = time.time()
        with torch.no_grad():
            result = self.model.decode(
                methods=["ctc_greedy_search"],
                speech=fb, speech_lengths=fb_len,
                beam_size=1,
                decoding_chunk_size=self.config.chunk_size,
            )
        elapsed = time.time() - t0

        hyp_tokens = result["ctc_greedy_search"][0].tokens
        text = "".join(hyp_tokens)
        audio_duration = fb.shape[1] * 0.01
        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        return {
            "text": text,
            "tokens": hyp_tokens,
            "rtf": round(rtf, 4),
            "time_ms": round(elapsed * 1000, 1),
            "chunk_size": self.config.chunk_size,
        }


# ── FastAPI WebSocket Server ──

server_app = None
asr_engine: Optional[StreamingASR] = None


def create_app(config: ASRConfig):
    """
    Create FastAPI application with:
      - GET /health — health check
      - GET /transcribe — full-utterance transcription (query param ?wav=...)
      - WS /asr/stream — real-time streaming via WebSocket
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="WeNet U2++ Streaming ASR",
        description="Real-time streaming ASR server with U2++ dynamic chunk mechanism.\n"
                    "Supports chunk-by-chunk WebSocket streaming and full-utterance REST.",
        version="0.1.0",
    )

    engine = StreamingASR(config)

    @app.get("/health")
    async def health():
        return JSONResponse({
            "status": "ok",
            "model": os.path.basename(config.model_path),
            "chunk_size": config.chunk_size,
            "latency_per_chunk_ms": config.latency_per_chunk_ms,
        })

    @app.get("/transcribe")
    async def transcribe(wav_path: str = Query(..., description="Path to WAV file")):
        """Transcribe a full WAV file (non-streaming baseline)."""
        import soundfile
        try:
            audio, sr = soundfile.read(wav_path, dtype="float32")
        except Exception as e:
            return JSONResponse({"error": f"Cannot read WAV: {e}"}, status_code=400)

        result = engine.decode(audio, sr)
        return JSONResponse({
            "text": result["text"],
            "rtf": result["rtf"],
            "time_ms": result["time_ms"],
            "method": result["method"],
            "wav": wav_path,
        })

    @app.websocket("/asr/stream")
    async def stream_asr(websocket: WebSocket):
        """
        WebSocket streaming endpoint.
        Protocol:
          1. Client sends:  {"config": {"chunk_size": 16, "beam_size": 5}}
          2. Server sends:  {"status": "ready", "chunk_size": 16}
          3. Client sends audio bytes (raw PCM S16LE, 16kHz, mono)
          4. Server responds: {"text": "...", "finished": false, "rtf": 0.01}
          5. Repeat 3-4 until client sends {"eof": true}
          6. Server responds: {"finished": true, "text": "..."}
        """
        await websocket.accept()
        print(f"[WS] Client connected")

        import soundfile

        # 1. Receive configuration
        config_msg = await websocket.receive_text()
        try:
            client_config = json.loads(config_msg)
            if "config" in client_config:
                cfg = client_config["config"]
                if "chunk_size" in cfg:
                    config.chunk_size = int(cfg["chunk_size"])
                if "beam_size" in cfg:
                    config.beam_size = int(cfg["beam_size"])
        except json.JSONDecodeError:
            pass

        await websocket.send_json({
            "status": "ready",
            "chunk_size": config.chunk_size,
            "sample_rate": 16000,
        })
        print(f"[WS] Ready with chunk_size={config.chunk_size}")

        # 2. Streaming loop
        audio_buffer = b""
        full_text = []
        try:
            while True:
                data = await websocket.receive_bytes()

                # Check for EOF signal
                if data == b"EOF":
                    break

                # Append to buffer
                audio_buffer += data

                # Process if we have enough data
                # (at least 1 chunk = chunk_size * 640 samples with subsampling)
                min_samples = config.chunk_size * config.subsampling_factor * config.frame_shift_ms * 16
                while len(audio_buffer) >= min_samples:
                    chunk_bytes = audio_buffer[:min_samples]
                    audio_buffer = audio_buffer[min_samples:]

                    # Decode PCM S16LE → float32
                    chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    result = engine.decode_chunk(chunk_np)
                    full_text.append(result["text"])

                    await websocket.send_json({
                        "text": result["text"],
                        "finished": False,
                        "rtf": result["rtf"],
                        "time_ms": result["time_ms"],
                    })

            # Process remaining buffer (partial chunk)
            if audio_buffer:
                chunk_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                if len(chunk_np) > 0:
                    result = engine.decode_chunk(chunk_np)
                    full_text.append(result["text"])
                    await websocket.send_json({
                        "text": result["text"],
                        "finished": False,
                        "rtf": result["rtf"],
                        "time_ms": result["time_ms"],
                    })

            # Final result
            await websocket.send_json({
                "finished": True,
                "text": "".join(full_text),
                "total_text": "".join(full_text),
            })

        except WebSocketDisconnect:
            print(f"[WS] Client disconnected")
        finally:
            print(f"[WS] Session done: {''.join(full_text)[:60]}")

    return app


# ── WebSocket Client (for testing) ──

async def ws_client(server_url: str, wav_path: str, chunk_size: int = 16):
    """Connect to WebSocket server and stream audio chunk by chunk."""
    import soundfile

    try:
        import websockets
    except ImportError:
        print("Please install websockets: pip install websockets")
        sys.exit(1)

    audio, sr = soundfile.read(wav_path, dtype="float32")
    if sr != 16000:
        from scipy import signal
        import numpy as np
        old_len = len(audio)
        new_len = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, new_len)

    # Resample to S16LE PCM bytes
    pcm_bytes = (audio * 32768).astype(np.int16).tobytes()

    # Chunk size in samples
    chunk_samples = chunk_size * 4 * 10 * 16  # chunk * subsampling * frame_shift * 16kHz

    uri = server_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{uri}/asr/stream"

    async with websockets.connect(ws_url) as ws:
        # Send config
        await ws.send(json.dumps({"config": {"chunk_size": chunk_size}}))
        resp = await ws.recv()
        print(f"[Client] Server: {resp}")

        # Stream audio
        offset = 0
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset:offset + chunk_samples]
            await ws.send(chunk)
            resp = await ws.recv()
            data = json.loads(resp)
            if not data.get("finished"):
                print(f"[Client] Partial: {data['text']}  (rtf={data.get('rtf', 'N/A')})")
            offset += chunk_samples

        # Send EOF
        await ws.send(b"EOF")
        resp = await ws.recv()
        data = json.loads(resp)
        print(f"[Client] Final: {data.get('total_text', data.get('text', 'N/A'))}")


def main():
    parser = argparse.ArgumentParser(description="Streaming ASR Server & Client")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--chunk", type=int, default=16, help="Decoding chunk size")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--mode", choices=["server", "client"], default="server",
                        help="Run as server or client")
    parser.add_argument("--wav", default=None, help="WAV path (client mode only)")
    parser.add_argument("--server-url", default="http://localhost:8765",
                        help="Server URL (client mode)")
    parser.add_argument("--no-reload", action="store_true",
                        help="Disable hot-reload (server mode)")
    args = parser.parse_args()

    config = ASRConfig()
    if args.chunk:
        config.chunk_size = args.chunk
    if args.checkpoint:
        config.model_path = args.checkpoint

    if args.mode == "client":
        if not args.wav:
            print("ERROR: --wav required in client mode")
            sys.exit(1)
        asyncio.run(ws_client(args.server_url, args.wav, args.chunk))
        return

    # Server mode
    app = create_app(config)
    import uvicorn
    print(f"\n{'=' * 60}")
    print(f"  WeNet U2++ Streaming ASR Server")
    print(f"  Chunk size: {config.chunk_size} | Port: {args.port}")
    print(f"  Latency per chunk: ~{config.latency_per_chunk_ms}ms")
    print(f"  WebSocket: ws://{args.host}:{args.port}/asr/stream")
    print(f"  Health:    http://{args.host}:{args.port}/health")
    print(f"{'=' * 60}\n")
    uvicorn.run(app, host=args.host, port=args.port,
                reload=not args.no_reload)


if __name__ == "__main__":
    main()
