# -*- coding: utf-8 -*-
"""AST attention -> importance map -> frame selection -> mel + waveform reconstruction."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .attention_extract import (
    attention_to_time_patch_importance,
    collect_attention_matrices,
    frame_importance_from_patches,
    _patch_grid_dims,
)
from .mel_ast import (
    SAMPLE_RATE,
    denormalize_mel,
    load_waveform,
    prepare_batch,
    waveform_to_mel_fbank,
    waveform_to_mel_fbank_full,
)
from .vocoder import log_mel_to_waveform_approx


def _best_contiguous_frame_indices(per_frame_score: torch.Tensor, window_len: int) -> List[int]:
    """
    One time-contiguous excerpt of ``window_len`` frames with maximum total AST importance.
    Avoids scattered 10 ms hops (which jump in time and destroy speech).
    """
    T = int(per_frame_score.shape[0])
    k = min(max(1, window_len), T)
    if T <= k:
        return list(range(T))
    s = per_frame_score.float()
    roll = s.unfold(0, k, 1)
    start = int(torch.argmax(roll.sum(dim=1)).item())
    return list(range(start, start + k))


def _ensure_ast_src_on_path(repo_root: Path) -> Path:
    src = repo_root / "ast" / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Expected AST src at {src}")
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return src


def load_ast_model(
    repo_root: Path,
    input_tdim: int,
    *,
    label_dim: int = 527,
    audioset_pretrain: bool = False,
    imagenet_pretrain: bool = True,
    fstride: int = 10,
    tstride: int = 10,
    device: Optional[str] = None,
):
    """
    Instantiate ASTModel with cwd temporarily set to ast/src so pretrained paths resolve.
    """
    src = _ensure_ast_src_on_path(repo_root)
    prev = os.getcwd()
    os.chdir(src)
    try:
        torch_home = (repo_root / "ast" / "pretrained_models").resolve()
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)
        from models.ast_models import ASTModel  # noqa: WPS433

        m = ASTModel(
            label_dim=label_dim,
            fstride=fstride,
            tstride=tstride,
            input_fdim=128,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size="base384",
            verbose=False,
        )
    finally:
        os.chdir(prev)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(dev)
    m.eval()
    return m


@dataclass
class ASTSummarizeResult:
    mel_full: torch.Tensor
    mel_summary: torch.Tensor
    waveform_summary: torch.Tensor
    selected_frame_indices: List[int]
    time_patch_importance: torch.Tensor
    f_dim: int
    t_dim: int


def _write_flow_verification(
    debug_dir: Path,
    *,
    audio_path: str | Path,
    wav_samples: int,
    sample_rate: int,
    mel_full: torch.Tensor,
    mel_summary: torch.Tensor,
    f_dim: int,
    t_dim: int,
    tstride: int,
    kernel_t: int,
    num_attn_layers: int,
    time_patch_importance: torch.Tensor,
    frame_importance: torch.Tensor,
    selected_indices: List[int],
    waveform_out_samples: int,
) -> None:
    """Write artifacts proving each step of the Audio->...->Summary pipeline."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    np.save(debug_dir / "01_mel_log_normalized_ast.npy", mel_full.numpy())
    np.save(debug_dir / "02_time_patch_importance.npy", time_patch_importance.numpy())
    np.save(debug_dir / "03_frame_importance_upsampled.npy", frame_importance.numpy())
    np.save(debug_dir / "04_mel_summary_reconstructed.npy", mel_summary.numpy())

    (debug_dir / "05_selected_frame_indices.txt").write_text(
        ",".join(map(str, selected_indices)), encoding="utf-8"
    )

    report = f"""AST summarization flow check (this run)
=====================================
Input audio file: {audio_path}

1) Audio
   Loaded mono waveform, {wav_samples} samples @ {sample_rate} Hz (~{wav_samples / sample_rate:.2f} s).

2) Log-Mel spectrogram (Kaldi fbank, AST normalization)
   Tensor shape [time_frames, mel_bins] = {tuple(mel_full.shape)}
   Implementation: mel_ast.waveform_to_mel_fbank()

3) Patch embedding + 4) Audio Spectrogram Transformer
   Inside ASTModel.forward (ast/src/models/ast_models.py):
   Conv2d patch embed kernel (16,16), stride (fstride, tstride) from AST.
   Patch grid (freq x time) = {f_dim} x {t_dim}
   Time stride between patches: {tstride}, patch covers ~{kernel_t} mel time bins.

5) Attention extraction
   Hooks on last {num_attn_layers} blocks' attention softmax (before dropout).
   Implementation: attention_extract.collect_attention_matrices()

6) Importance map
   CLS+dist token attention to patch tokens, max over frequency -> per time-patch scores.
   Shape [t_dim] = {tuple(time_patch_importance.shape)}
   Upsampled to per-frame scores [T] = {tuple(frame_importance.shape)}
   Implementation: attention_to_time_patch_importance, frame_importance_from_patches

7) Patch / frame selection
   One contiguous block of {len(selected_indices)} frames with highest total importance
   (sliding window over time; avoids scattered micro-hops that break speech).
   Indices written to 05_selected_frame_indices.txt

8) Spectrogram reconstruction
   Rows mel[selected_frames, :] -> shape {tuple(mel_summary.shape)}
   File: 04_mel_summary_reconstructed.npy

9) Waveform reconstruction
   Denormalize mel -> InverseMelScale + GriffinLim (approximate).
   Output waveform length: {waveform_out_samples} samples (~{waveform_out_samples / sample_rate:.2f} s)

10) Summarized audio
   Written by CLI as --out-wav (GriffinLim) and optionally --out-wav-cuts (original waveform hops).

Note: For long files, only the first target_length mel frames (~10.24 s at default 1024) are
fed to AST in one forward; extend with sliding windows if you need the full {wav_samples / sample_rate:.0f}s clip.
"""
    (debug_dir / "FLOW_IMPLEMENTATION_MAP.txt").write_text(report, encoding="utf-8")


def _ast_speed_summarize_timed(
    audio_path: str | Path,
    repo_root: Path,
    *,
    summary_duration_sec: float = 60.0,
    ast_window: int = 1024,
    ast_hop: int = 1024,
    num_last_blocks: int = 6,
    device: Optional[str] = None,
    audioset_pretrain: bool = False,
    imagenet_pretrain: bool = True,
    debug_dir: Optional[str | Path] = None,
) -> ASTSummarizeResult:
    """
    Sliding-window AST over the full mel, aggregate frame importance, then pick **one
    contiguous** stretch of ``summary_duration_sec`` (100 mel frames per second) with
    the highest total importance—listenable speech, not scattered 10 ms hops.
    """
    wav = load_waveform(audio_path)
    wav_num_samples = int(wav.shape[1])
    mel_cpu = waveform_to_mel_fbank_full(wav)
    T_full = int(mel_cpu.shape[0])
    k_frames = min(max(1, int(round(summary_duration_sec * 100.0))), T_full)

    model = load_ast_model(
        repo_root,
        input_tdim=ast_window,
        device=device,
        audioset_pretrain=audioset_pretrain,
        imagenet_pretrain=imagenet_pretrain,
    )
    tstride = int(model.v.patch_embed.proj.stride[1])
    f_dim, t_dim = _patch_grid_dims(model, ast_window)

    global_scores = torch.zeros(T_full, dtype=torch.float32)
    global_counts = torch.zeros(T_full, dtype=torch.float32)
    last_imp_t: Optional[torch.Tensor] = None

    for s in range(0, T_full, ast_hop):
        valid = min(ast_window, T_full - s)
        if valid <= 0:
            break
        chunk = mel_cpu[s : s + ast_window, :].clone()
        if chunk.shape[0] < ast_window:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, ast_window - chunk.shape[0]))
        x = prepare_batch(chunk)
        attn = collect_attention_matrices(model, x, num_last_blocks=num_last_blocks)
        imp_t = attention_to_time_patch_importance(attn, f_dim, t_dim)
        last_imp_t = imp_t[0].detach().cpu()
        imp_frames = frame_importance_from_patches(imp_t, ast_window, tstride)
        scores = imp_frames[0, :valid].detach().float().cpu()
        global_scores[s : s + valid] += scores
        global_counts[s : s + valid] += 1.0

    agg = global_scores / global_counts.clamp(min=1.0)
    idx = _best_contiguous_frame_indices(agg, k_frames)

    mel_s = mel_cpu[idx[0] : idx[0] + len(idx), :]
    log_mel_denorm = denormalize_mel(mel_s)
    wav_out = log_mel_to_waveform_approx(log_mel_denorm.cpu())

    if debug_dir is not None and last_imp_t is not None:
        _write_flow_verification(
            Path(debug_dir),
            audio_path=audio_path,
            wav_samples=wav_num_samples,
            sample_rate=SAMPLE_RATE,
            mel_full=mel_cpu,
            mel_summary=mel_s,
            f_dim=f_dim,
            t_dim=t_dim,
            tstride=tstride,
            kernel_t=16,
            num_attn_layers=num_last_blocks,
            time_patch_importance=last_imp_t,
            frame_importance=agg,
            selected_indices=idx,
            waveform_out_samples=int(wav_out.shape[-1]),
        )
        p = Path(debug_dir) / "FLOW_IMPLEMENTATION_MAP.txt"
        extra = (
            f"\n\n[Timed / sliding-window mode]\n"
            f"Full mel frames T_full={T_full}, AST window={ast_window}, hop={ast_hop}.\n"
            f"Target summary duration: {summary_duration_sec}s -> {k_frames} contiguous mel frames "
            f"starting at index {idx[0]} (max total importance).\n"
        )
        p.write_text(p.read_text(encoding="utf-8") + extra, encoding="utf-8")

    return ASTSummarizeResult(
        mel_full=mel_cpu,
        mel_summary=mel_s,
        waveform_summary=wav_out.cpu(),
        selected_frame_indices=idx,
        time_patch_importance=last_imp_t if last_imp_t is not None else torch.zeros(t_dim),
        f_dim=f_dim,
        t_dim=t_dim,
    )


def ast_speed_summarize(
    audio_path: str | Path,
    repo_root: Path,
    *,
    target_length: int = 1024,
    keep_ratio: float = 0.35,
    num_last_blocks: int = 6,
    device: Optional[str] = None,
    audioset_pretrain: bool = False,
    imagenet_pretrain: bool = True,
    debug_dir: Optional[str | Path] = None,
    summary_duration_sec: Optional[float] = None,
) -> ASTSummarizeResult:
    """
    Full pipeline:
    Audio -> log-mel -> AST -> attention importance map -> frame selection ->
    shortened mel -> GriffinLim waveform.

    If ``summary_duration_sec`` > 0, uses sliding AST windows over the **full** mel and keeps
    about that many seconds of frames (100 frames per second at 10 ms hop).

    Otherwise ``keep_ratio`` applies to the first ``target_length`` mel frames only
    (also as **one contiguous** sub-window of that length with max importance).
    """
    if summary_duration_sec is not None and float(summary_duration_sec) > 0:
        return _ast_speed_summarize_timed(
            audio_path,
            repo_root,
            summary_duration_sec=float(summary_duration_sec),
            ast_window=target_length,
            ast_hop=target_length,
            num_last_blocks=num_last_blocks,
            device=device,
            audioset_pretrain=audioset_pretrain,
            imagenet_pretrain=imagenet_pretrain,
            debug_dir=debug_dir,
        )

    wav = load_waveform(audio_path)
    wav_num_samples = int(wav.shape[1])
    mel = waveform_to_mel_fbank(wav, target_length=target_length)
    T = mel.shape[0]
    model = load_ast_model(
        repo_root,
        input_tdim=T,
        device=device,
        audioset_pretrain=audioset_pretrain,
        imagenet_pretrain=imagenet_pretrain,
    )

    x = prepare_batch(mel)
    attn = collect_attention_matrices(model, x, num_last_blocks=num_last_blocks)
    f_dim, t_dim = _patch_grid_dims(model, T)
    tstride = int(model.v.patch_embed.proj.stride[1])

    imp_t = attention_to_time_patch_importance(attn, f_dim, t_dim)
    imp_frames = frame_importance_from_patches(imp_t, T, tstride)

    k = max(1, int(round(T * keep_ratio)))
    scores = imp_frames[0]
    idx = _best_contiguous_frame_indices(scores.cpu(), k)

    mel_s = mel[idx[0] : idx[0] + len(idx), :]

    log_mel_denorm = denormalize_mel(mel_s)
    wav_out = log_mel_to_waveform_approx(log_mel_denorm.cpu())

    if debug_dir is not None:
        _write_flow_verification(
            Path(debug_dir),
            audio_path=audio_path,
            wav_samples=wav_num_samples,
            sample_rate=SAMPLE_RATE,
            mel_full=mel,
            mel_summary=mel_s,
            f_dim=f_dim,
            t_dim=t_dim,
            tstride=tstride,
            kernel_t=16,
            num_attn_layers=num_last_blocks,
            time_patch_importance=imp_t[0].detach().cpu(),
            frame_importance=scores.detach().cpu(),
            selected_indices=idx,
            waveform_out_samples=int(wav_out.shape[-1]),
        )

    return ASTSummarizeResult(
        mel_full=mel.cpu(),
        mel_summary=mel_s.cpu(),
        waveform_summary=wav_out.cpu(),
        selected_frame_indices=idx,
        time_patch_importance=imp_t[0].cpu(),
        f_dim=f_dim,
        t_dim=t_dim,
    )


def waveform_cuts_from_frames(
    audio_path: str | Path,
    frame_indices: List[int],
    hop_ms: float = 10.0,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """
    Concatenate original waveform segments for each selected frame (one hop of samples per frame).
    More faithful than GriffinLim when frames are contiguous runs.
    """
    wav = load_waveform(audio_path)
    hop = int(sample_rate * hop_ms / 1000.0)
    indices = sorted(set(frame_indices))
    if not indices:
        return torch.zeros(1, 0)
    # One contiguous block: single slice (no per-hop stitching)
    if indices[-1] - indices[0] + 1 == len(indices):
        s = indices[0] * hop
        e = min((indices[-1] + 1) * hop, wav.shape[1])
        return wav[:, s:e]
    pieces = []
    for fi in indices:
        s = fi * hop
        e = min(s + hop, wav.shape[1])
        if s < e:
            pieces.append(wav[:, s:e])
    return torch.cat(pieces, dim=1)
