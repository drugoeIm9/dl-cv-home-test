from __future__ import annotations
from typing import Dict, Optional
import torch

from .common import ensure_bhw, valid_mask, safe_mean, to_float, EPS

# -------- Monocular depth metrics (meters) --------
def abs_rel(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=EPS)
    rel = (pred - gt).abs() / gt.clamp_min(EPS)
    return to_float(safe_mean(rel, m))

def sq_rel(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=EPS)
    srel = (pred - gt).pow(2) / gt.clamp_min(EPS)
    return to_float(safe_mean(srel, m))

def rmse(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=EPS)
    e2 = (pred - gt).pow(2)
    return to_float(torch.sqrt(safe_mean(e2, m)))

def rmse_log(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=EPS)
    e2 = (torch.log(pred.clamp_min(EPS)) - torch.log(gt.clamp_min(EPS))).pow(2)
    return to_float(torch.sqrt(safe_mean(e2, m)))

def delta_acc(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 1.25,
              mask: Optional[torch.Tensor] = None) -> float:
    """Percent where max(pred/gt, gt/pred) < thresh."""
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=EPS)
    ratio = torch.maximum(pred / gt.clamp_min(EPS), gt / pred.clamp_min(EPS))
    good = (ratio < thresh) & m
    pct = good.float().sum() / m.float().sum().clamp_min(1.0)
    return to_float(pct * 100.0)

def silog(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None,
          lam: float = 1.0) -> float:
    """
    SILog = sqrt( mean(e^2) - lam*(mean(e))^2 ), e=log(pred)-log(gt)
    """
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=EPS)
    e = torch.log(pred.clamp_min(EPS)) - torch.log(gt.clamp_min(EPS))
    mu = safe_mean(e, m)
    msq = safe_mean(e ** 2, m)
    return to_float(torch.sqrt((msq - lam * mu ** 2).clamp_min(0.0)))

def median_scale(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scale pred so median(pred)=median(gt) on valid region (for mono scale ambiguity)."""
    pred, gt = ensure_bhw(pred), ensure_bhw(gt)
    m = valid_mask(pred, gt, mask, gt_min=EPS)
    if m.sum() == 0:
        return pred
    s = torch.median(gt[m]).clamp_min(EPS) / torch.median(pred[m]).clamp_min(EPS)
    return pred * s

def compute_mono_metrics(pred_depth: torch.Tensor, gt_depth: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         apply_median_scale: bool = False) -> Dict[str, float]:
    if apply_median_scale:
        pred_depth = median_scale(pred_depth, gt_depth, mask)
    return {
        "AbsRel": abs_rel(pred_depth, gt_depth, mask),
        "SqRel": sq_rel(pred_depth, gt_depth, mask),
        "RMSE": rmse(pred_depth, gt_depth, mask),
        "RMSE_log": rmse_log(pred_depth, gt_depth, mask),
        "δ<1.25(%)": delta_acc(pred_depth, gt_depth, 1.25, mask),
        "δ<1.25^2(%)": delta_acc(pred_depth, gt_depth, 1.25 ** 2, mask),
        "δ<1.25^3(%)": delta_acc(pred_depth, gt_depth, 1.25 ** 3, mask),
        "SILog": silog(pred_depth, gt_depth, mask),
    }
