# -*- coding: utf-8 -*-
"""
Approximate waveform from log-mel (Kaldi-style) using InverseMelScale + GriffinLim.
Quality is limited; prefer ``waveform_cuts_from_frames`` when possible.
"""

from __future__ import annotations

import torch
import torchaudio

SAMPLE_RATE = 16000
N_MELS = 128
HOP = 160


def log_mel_to_waveform_approx(log_mel: torch.Tensor) -> torch.Tensor:
    """
    log_mel: [T_frames, n_mels] natural-log domain (denormalized, not AST-normalized).
    Returns waveform [1, samples] float32 (or near-silent placeholder if inversion fails).
    """
    log_mel = log_mel.float().clamp(min=-35.0, max=15.0)
    # GriffinLim on thousands of frames is very slow; use silent placeholder of matching length.
    if log_mel.shape[0] > 2500:
        return torch.zeros(1, int(log_mel.shape[0] * HOP), dtype=torch.float32)
    power = torch.exp(log_mel).T.unsqueeze(0) + 1e-4  # [1, n_mels, T]

    n_fft = 1024
    n_stft = n_fft // 2 + 1

    try:
        inv = torchaudio.transforms.InverseMelScale(
            n_stft=n_stft,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="htk",
        )
        linear = inv(power)
        gl = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=HOP,
            n_iter=32,
            power=1.0,
        )
        return gl(linear)
    except Exception:
        # lstsq can fail on rank-deficient mel frames; return short silence
        return torch.zeros(1, max(1, power.shape[-1] * HOP), dtype=torch.float32)
