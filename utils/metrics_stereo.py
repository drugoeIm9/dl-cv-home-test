from __future__ import annotations
from typing import Dict, Optional
import torch

from .common import ensure_bhw, valid_mask, safe_mean, to_float

# -------- Stereo metrics (disparity in pixels) --------
def epe_disparity(pred: torch.Tensor, gt: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> float:
    """Mean absolute disparity error (px)."""
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=0.0)
    e = (pred - gt).abs()
    return to_float(safe_mean(e, m))

def bad_px(pred: torch.Tensor, gt: torch.Tensor, thr_px: float = 1.0,
           mask: Optional[torch.Tensor] = None) -> float:
    """Percent of pixels with |error| > thr_px."""
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=0.0)
    err = (pred - gt).abs()
    bad = (err > thr_px) & m
    pct = bad.float().sum() / m.float().sum().clamp_min(1.0)
    return to_float(pct * 100.0)

def d1_kitti(pred: torch.Tensor, gt: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> float:
    """KITTI D1-all: % with |err| > max(3 px, 5% of gt)."""
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=0.0)
    err = (pred - gt).abs()
    thr = torch.maximum(torch.tensor(3.0, device=gt.device, dtype=gt.dtype), 0.05 * gt)
    bad = (err > thr) & m
    pct = bad.float().sum() / m.float().sum().clamp_min(1.0)
    return to_float(pct * 100.0)

def compute_stereo_metrics(pred_disp: torch.Tensor, gt_disp: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Convenience wrapper."""
    return {
        "EPE": epe_disparity(pred_disp, gt_disp, mask),
        "D1(%)": d1_kitti(pred_disp, gt_disp, mask),
        "Bad1px(%)": bad_px(pred_disp, gt_disp, 1.0, mask),
        "Bad3px(%)": bad_px(pred_disp, gt_disp, 3.0, mask),
    }
