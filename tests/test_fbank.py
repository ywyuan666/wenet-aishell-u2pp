#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for fbank extraction (requires audio file)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_eval import extract_fbank


@pytest.fixture
def sample_wav(tmp_path):
    """Generate a small synthetic WAV file (sine wave, 16kHz)."""
    sr = 16000
    duration = 0.3  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    wav_path = tmp_path / "test_sine.wav"
    soundfile.write(str(wav_path), wav, sr)
    return str(wav_path)


class TestExtractFbank:
    """Tests for fbank feature extraction."""

    def test_extract_output_shape(self, sample_wav):
        """Fbank output should have 80 mel bins."""
        fb, fb_len = extract_fbank(sample_wav)
        # fb shape: (1, time_frames, 80)
        assert fb.dim() == 3
        assert fb.shape[0] == 1  # batch
        assert fb.shape[2] == 80  # mel bins
        assert fb_len.shape == (1,)  # 1-element tensor

    def test_extract_fbank_dtype(self, sample_wav):
        """Fbank output should be float32."""
        fb, _ = extract_fbank(sample_wav)
        assert fb.dtype == torch.float32

    def test_extract_fbank_nonzero(self, sample_wav):
        """Valid audio should produce non-zero features."""
        fb, _ = extract_fbank(sample_wav)
        assert fb.abs().sum() > 0

    def test_non_existent_file(self):
        """Invalid file path should raise an exception."""
        with pytest.raises(Exception):
            extract_fbank("/nonexistent/file.wav")
