from __future__ import annotations
from typing import Optional
import torch

EPS = 1e-6

def to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())

def ensure_bhw(x: torch.Tensor) -> torch.Tensor:
    """Accept (B,H,W) or (H,W); return (B,H,W)."""
    if x.dim() == 2:  # (H,W)
        return x.unsqueeze(0)
    if x.dim() == 3:  # (B,H,W)
        return x
    raise ValueError(f"Expected 2D or 3D tensor, got {tuple(x.shape)}")

def valid_mask(
    pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None,
    gt_min: float = 0.0, gt_max: float = float("inf")
) -> torch.Tensor:
    """Combine user mask with finite & value range checks on GT."""
    m = torch.isfinite(gt) & (gt > gt_min) & (gt < gt_max)
    if mask is not None:
        m = m & mask.bool()
    return ensure_bhw(m.float()) > 0.5

def safe_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.float().sum().clamp_min(1.0)
    return (x * mask.float()).sum() / denom
