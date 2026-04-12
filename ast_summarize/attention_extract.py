# -*- coding: utf-8 -*-
"""
Extract patch-level importance from AST (DeiT) self-attention:
CLS + distillation tokens -> patch tokens, averaged over heads and last blocks.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def _patch_grid_dims(model, input_tdim: int) -> Tuple[int, int]:
    """Frequency x time patch counts for current patch embed."""
    proj = model.v.patch_embed.proj
    fstride = proj.stride[0]
    tstride = proj.stride[1]
    f_dim, t_dim = model.get_shape(fstride, tstride, 128, input_tdim)
    return f_dim, t_dim


def collect_attention_matrices(model, x: torch.Tensor, num_last_blocks: int = 6) -> torch.Tensor:
    """
    Run one forward, capture pre-dropout attention softmax from last ``num_last_blocks`` blocks.

    x: [B, T, F] mel batch
    Returns tensor [num_layers, B, heads, N, N] on CPU float32.
    """
    storage: List[torch.Tensor] = []
    hooks = []

    def make_hook():
        def hook(module, inp, _out):
            if inp and isinstance(inp[0], torch.Tensor):
                storage.append(inp[0].detach().float().cpu())

        return hook

    blocks = model.v.blocks
    for blk in blocks[-num_last_blocks:]:
        attn = getattr(blk, "attn", None)
        if attn is None:
            continue
        drop = getattr(attn, "attn_drop", None)
        if drop is None:
            continue
        hooks.append(drop.register_forward_hook(make_hook()))

    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    if not storage:
        raise RuntimeError(
            "Could not capture attention (no attn_drop hooks). "
            "Try timm==0.4.5 as in ast/requirements.txt."
        )
    return torch.stack(storage, dim=0)


def attention_to_time_patch_importance(
    attn_stack: torch.Tensor,
    f_dim: int,
    t_dim: int,
) -> torch.Tensor:
    """
    attn_stack: [L, B, H, N, N]
    Average layers and heads; pool CLS (0) and dist (1) attention onto patch tokens.
    Returns [B, t_dim] importance per time-patch column (max over frequency rows).
    """
    # [L, B, H, N, N] -> [B, H, N, N]
    a = attn_stack.mean(0)
    b, h, n, _ = a.shape
    # DeiT distilled: tokens 0=cls, 1=dist, 2..=patches
    cls_p = a[:, :, 0, 2:].mean(dim=1)
    dist_p = a[:, :, 1, 2:].mean(dim=1)
    patch = (cls_p + dist_p) / 2.0
    p = patch.shape[1]
    expected = f_dim * t_dim
    if p != expected:
        raise ValueError(f"Patch count mismatch: attention has {p}, grid {f_dim}x{t_dim}={expected}")
    grid = patch.reshape(b, f_dim, t_dim)
    # emphasize salient time regions
    imp_t, _ = grid.max(dim=1)  # [B, t_dim]
    return imp_t


def frame_importance_from_patches(
    imp_t: torch.Tensor,
    input_tdim: int,
    tstride: int,
    kernel_t: int = 16,
) -> torch.Tensor:
    """
    Upsample time-patch scores to per-frame importance [B, T].
    """
    b, t_dim = imp_t.shape
    out = torch.zeros(b, input_tdim, device=imp_t.device, dtype=imp_t.dtype)
    for tp in range(t_dim):
        s = tp * tstride
        e = min(s + kernel_t, input_tdim)
        if s >= input_tdim:
            break
        out[:, s:e] = torch.maximum(out[:, s:e], imp_t[:, tp : tp + 1])
    return out
