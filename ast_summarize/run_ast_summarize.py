# -*- coding: utf-8 -*-
"""
CLI: AST attention pipeline for shorter "summary" audio.

From project root (see README.md):
  python ast_summarize/run_ast_summarize.py --audio examples/audio/your.wav --out-wav examples/output/summary.wav --out-wav-cuts examples/output/summary_cuts.wav

Flow: Audio -> log-mel (Kaldi) -> AST patch embed + blocks -> attention extraction ->
importance map -> frame selection -> mel reconstruction -> GriffinLim waveform.

Requires: torch, torchaudio, timm (see ast/requirements.txt; timm 0.4.5 recommended).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description="AST attention-based speed summarization")
    parser.add_argument("--audio", type=str, required=True, help="Input wav/mp3/flac (resampled to 16 kHz)")
    parser.add_argument("--out-wav", type=str, required=True, help="Output summarized wav")
    parser.add_argument(
        "--out-wav-cuts",
        type=str,
        default="",
        help="Optional: concatenate original 10 ms hops for selected frames (cleaner than GriffinLim)",
    )
    parser.add_argument(
        "--summary-duration-sec",
        type=float,
        default=60.0,
        help="Target summary length in seconds (~100 mel frames/s). Use 0 with --keep-ratio for short-window mode.",
    )
    parser.add_argument("--keep-ratio", type=float, default=0.35, help="Only if summary-duration-sec is 0: fraction of first window to keep")
    parser.add_argument("--target-length", type=int, default=1024, help="AST window size (mel frames) and hop in timed mode; first window size in legacy mode")
    parser.add_argument("--last-blocks", type=int, default=6, help="Transformer blocks to average for attention")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu")
    parser.add_argument(
        "--audioset-pretrain",
        action="store_true",
        help="Use AudioSet weights if ast/pretrained_models/audioset_10_10_0.4593.pth exists",
    )
    parser.add_argument(
        "--no-imagenet-pretrain",
        action="store_true",
        help="Skip DeiT ImageNet weights (no download; random init). Use when offline or DNS fails; summary quality drops.",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="",
        help="Write mel/importance .npy files and FLOW_IMPLEMENTATION_MAP.txt to verify each pipeline step",
    )
    args = parser.parse_args()

    from ast_summarize.mel_ast import SAMPLE_RATE, save_waveform
    from ast_summarize.pipeline import ast_speed_summarize, waveform_cuts_from_frames

    device = args.device if args.device else None
    dbg = Path(args.debug_dir) if args.debug_dir else None
    sd = None if args.summary_duration_sec <= 0 else float(args.summary_duration_sec)
    # AudioSet AST path requires ImageNet init in upstream ASTModel; keep pretrain on in that case.
    imagenet_pretrain = True if args.audioset_pretrain else not args.no_imagenet_pretrain
    result = ast_speed_summarize(
        args.audio,
        _REPO,
        target_length=args.target_length,
        keep_ratio=args.keep_ratio,
        num_last_blocks=args.last_blocks,
        device=device,
        audioset_pretrain=args.audioset_pretrain,
        imagenet_pretrain=imagenet_pretrain,
        debug_dir=dbg,
        summary_duration_sec=sd,
    )

    out = Path(args.out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_waveform(out, result.waveform_summary, SAMPLE_RATE)

    print("Saved:", out.resolve())
    print("Frames kept:", len(result.selected_frame_indices), "/", result.mel_full.shape[0])
    hop_sec = 0.01
    if result.selected_frame_indices:
        a, b = result.selected_frame_indices[0], result.selected_frame_indices[-1]
        print(
            "Contiguous excerpt in original:",
            f"{a * hop_sec:.2f}s to {(b + 1) * hop_sec:.2f}s",
            f"(mel frames {a}..{b})",
        )
    print(
        "Approx summary audio length (cuts):",
        f"{len(result.selected_frame_indices) * hop_sec:.1f}s",
        f"({hop_sec*1000:.0f} ms hop)",
    )
    print("Patch grid (freq x time):", result.f_dim, "x", result.t_dim)
    if dbg:
        print("Flow verification written to:", dbg.resolve())

    if sd is not None and len(result.selected_frame_indices) > 2500:
        print(
            "Note: For long summaries, GriffinLim (--out-wav) is skipped internally (silent/short); "
            "listen to --out-wav-cuts for the real ~60s excerpt."
        )

    if args.out_wav_cuts:
        w = waveform_cuts_from_frames(args.audio, result.selected_frame_indices)
        cuts_path = Path(args.out_wav_cuts)
        cuts_path.parent.mkdir(parents=True, exist_ok=True)
        save_waveform(cuts_path, w, SAMPLE_RATE)
        print("Saved (original-hop cuts):", cuts_path.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
