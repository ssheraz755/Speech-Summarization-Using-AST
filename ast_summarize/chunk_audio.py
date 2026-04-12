# -*- coding: utf-8 -*-
"""Concatenate ordered audio chunks (e.g. LibriSpeech utterances) into one waveform."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import torch

from .mel_ast import SAMPLE_RATE, load_waveform

SAMPLE_RATE = SAMPLE_RATE


def _load_audio(path: Path) -> tuple:
    """Load mono [1, samples] at SAMPLE_RATE (MP3 via ffmpeg, WAV/FLAC via soundfile)."""
    w = load_waveform(path, target_sr=SAMPLE_RATE)
    return w, SAMPLE_RATE


def discover_flac_chunks(folder: str | Path) -> List[Path]:
    """All *.flac under folder, sorted by numeric suffix (0000, 0001, ...)."""
    folder = Path(folder)
    files = list(folder.glob("*.flac"))
    if not files:
        return []

    def sort_key(p: Path) -> tuple:
        m = re.search(r"-(\d+)\.flac$", p.name, re.I)
        return (int(m.group(1)), p.name) if m else (99999, p.name)

    return sorted(files, key=sort_key)


def concat_audio_files(
    paths: List[Path],
    out_path: str | Path,
    target_sr: int = SAMPLE_RATE,
    silence_between_sec: float = 0.15,
) -> Path:
    """
    Load each file, convert to mono, resample to target_sr, optional short silence between.
    Saves WAV (or format implied by out_path suffix).
    """
    if not paths:
        raise FileNotFoundError("No audio files to concatenate.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    silence_samples = int(target_sr * silence_between_sec) if silence_between_sec > 0 else 0
    silence = torch.zeros(1, silence_samples)

    pieces: List[torch.Tensor] = []
    for i, p in enumerate(paths):
        wav, _sr = _load_audio(p)
        pieces.append(wav)
        if silence_samples > 0 and i < len(paths) - 1:
            pieces.append(silence)

    combined = torch.cat(pieces, dim=1)
    out_path = Path(out_path)
    try:
        import soundfile as sf

        arr = combined.squeeze(0).numpy()
        if combined.shape[0] > 1:
            arr = combined.T.numpy()
        sf.write(str(out_path), arr, target_sr, subtype="PCM_16", format="WAV")
    except ImportError:
        import torchaudio

        torchaudio.save(str(out_path), combined, target_sr)
    return out_path
