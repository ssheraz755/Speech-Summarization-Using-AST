# -*- coding: utf-8 -*-
"""Load waveform and build AST-style log-mel (Kaldi fbank + AudioSet normalization)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio

# Defaults from AST AudioSet recipe (see ast/src/dataloader.py, ast/src/run.py)
AUDIOMEAN = -4.2677393
AUDIOSTD = 4.5689974
NUM_MELS = 128
SAMPLE_RATE = 16000

# soundfile/libsndfile often cannot read MP3; torchaudio 2.x may require TorchCodec.
_FFMPEG_EXTENSIONS = frozenset({".mp3", ".m4a", ".aac", ".opus", ".wma"})


def _load_waveform_ffmpeg(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Decode to mono float32 at ``target_sr`` via ffmpeg (stdout pipe)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg is not on PATH. Install ffmpeg (https://ffmpeg.org) to use MP3/M4A, "
            "or convert your file to WAV/FLAC."
        )
    proc = subprocess.run(
        [
            exe,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path.resolve()),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[:800]
        raise RuntimeError(f"ffmpeg could not decode the file: {err}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    wav = torch.from_numpy(raw.copy()).float().unsqueeze(0)
    return wav, target_sr


def load_waveform(path: str | Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Mono float tensor [1, num_samples] at ``target_sr``.

    - WAV/FLAC: soundfile when possible.
    - MP3/M4A/...: ffmpeg (must be on PATH).
    """
    path = Path(path).expanduser()
    if not path.is_file():
        msg = f"Audio file not found: {path.resolve()}"
        try:
            if path.parent.is_dir():
                names = [p.name for p in path.parent.iterdir() if p.is_file()]
                if names:
                    msg += "\nFiles in that folder: " + ", ".join(sorted(names)[:25])
        except OSError:
            pass
        raise FileNotFoundError(msg)

    path = path.resolve()
    ext = path.suffix.lower()

    wav: torch.Tensor
    sr: int

    if ext in _FFMPEG_EXTENSIONS:
        wav, sr = _load_waveform_ffmpeg(path, target_sr)
    else:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), always_2d=True, dtype="float32")
            wav = torch.from_numpy(data).float().T
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
        except Exception as e1:
            try:
                wav, sr = _load_waveform_ffmpeg(path, target_sr)
            except Exception as e2:
                raise RuntimeError(
                    f"Could not read {path.name!r} (soundfile: {e1}; ffmpeg: {e2})"
                ) from e2

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def save_waveform(path: str | Path, wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono/stereo float tensor [C, T] as WAV (PCM16)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = wav.detach().cpu()
    try:
        import soundfile as sf

        if w.dim() == 1:
            arr = w.numpy()
        elif w.shape[0] == 1:
            arr = w.squeeze(0).numpy()
        else:
            arr = w.T.numpy()
        sf.write(str(path), arr, int(sample_rate), subtype="PCM_16", format="WAV")
    except ImportError:
        torchaudio.save(str(path), w, int(sample_rate))


def waveform_to_mel_fbank(
    waveform: torch.Tensor,
    target_length: int,
    mean: float = AUDIOMEAN,
    std: float = AUDIOSTD,
) -> torch.Tensor:
    """
    waveform: [1, samples]
    Returns mel normalized for AST: shape [target_length, NUM_MELS] (time x freq).
    """
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=SAMPLE_RATE,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=NUM_MELS,
        dither=0.0,
        frame_shift=10,
    )
    n_frames = fbank.shape[0]
    if n_frames < target_length:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - n_frames))
    else:
        fbank = fbank[:target_length, :]
    fbank = (fbank - mean) / (std * 2.0)
    return fbank


def waveform_to_mel_fbank_full(
    waveform: torch.Tensor,
    mean: float = AUDIOMEAN,
    std: float = AUDIOSTD,
    max_frames: int = 200_000,
) -> torch.Tensor:
    """
    Full-length log-mel for long audio (no truncation). Caps at ``max_frames`` for safety.
    Shape [T, NUM_MELS] with ~10 ms per frame.
    """
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=SAMPLE_RATE,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=NUM_MELS,
        dither=0.0,
        frame_shift=10,
    )
    if fbank.shape[0] > max_frames:
        fbank = fbank[:max_frames, :]
    return (fbank - mean) / (std * 2.0)


def denormalize_mel(mel_norm: torch.Tensor, mean: float = AUDIOMEAN, std: float = AUDIOSTD) -> torch.Tensor:
    """Invert AST normalization (log-mel domain)."""
    return mel_norm * (std * 2.0) + mean


def prepare_batch(mel_tf: torch.Tensor) -> torch.Tensor:
    """mel [T, F] -> [1, T, F] for ASTModel."""
    return mel_tf.unsqueeze(0)
