# instereo2k.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL (H,W,3) uint8 -> torch.float32 (3,H,W) in [0,1]."""
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.transpose(2, 0, 1)  # (C,H,W)
    return torch.from_numpy(arr).float().div_(255.0)


def load_disp_png(path: Path, divisor: float = 1.0) -> np.ndarray:
    """
    Load disparity PNG to float32 (H,W).
    If your PNGs are stored as uint16 with x100 scaling, pass divisor=100.0.
    """
    im = Image.open(path)
    arr = np.array(im)  # keep whatever dtype, usually uint16 or float
    disp = arr.astype(np.float32)
    if divisor != 1.0:
        disp /= float(divisor)
    return disp


def resize_pair_and_disp(
    left: Image.Image,
    right: Image.Image,
    disp: Optional[np.ndarray],
    size_hw: Optional[Tuple[int, int]],
) -> Tuple[Image.Image, Image.Image, Optional[np.ndarray]]:
    """
    Resize images to (H,W). For disparity: **NEAREST** resampling and then
    multiply by horizontal scale factor (W_new/W_old). This avoids mixing
    invalid pixels and preserves discrete disparity values.
    """
    if size_hw is None:
        return left, right, disp

    H_new, W_new = size_hw
    W_old, H_old = left.size  # PIL stores size as (W, H)

    # Images can be bilinear; they are continuous-valued.
    left_r = left.resize((W_new, H_new), resample=Image.BILINEAR)
    right_r = right.resize((W_new, H_new), resample=Image.BILINEAR)

    if disp is not None:
        sx = W_new / float(W_old)
        # Use NEAREST to avoid mixing invalid pixels and to keep disparities discrete.
        # Work in float -> PIL 'F' image to keep range, then NEAREST sample.
        disp_img = Image.fromarray(disp, mode="F")
        disp_resized = disp_img.resize((W_new, H_new), resample=Image.NEAREST)
        disp = np.array(disp_resized, dtype=np.float32) * sx

    return left_r, right_r, disp


class InStereo2KDataset(Dataset):
    """
    InStereo2K reader for per-scene folders with fixed filenames:
      scene/
        left.png, right.png, left_disp.png, right_disp.png

    Returns dict:
      {
        'left':  float32 tensor (3,H,W) in [0,1]
        'right': float32 tensor (3,H,W) in [0,1]
        'disp':  float32 tensor (H,W)   [optional, if load_disp=True]
        'mask':  bool tensor   (H,W)    valid = disp > 0  [if disp present]
        'id':    str           scene id (e.g., '000401')
      }
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        val_ratio: float = 0.1,
        random_seed: int = 42,
        load_disp: bool = True,
        disp_side: str = "left",         # 'left' or 'right'
        disp_divisor: float = 1.0,       # e.g., 100.0 if stored as uint16*100
        resize_hw: Optional[Tuple[int, int]] = None,  # (H, W) or None
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.load_disp = load_disp
        self.disp_side = disp_side.lower()
        assert self.disp_side in ("left", "right"), "disp_side must be 'left' or 'right'"
        self.disp_divisor = float(disp_divisor)
        self.resize_hw = resize_hw
        self.transform = transform

        # Gather scenes
        scenes = []
        for p in sorted(self.root_dir.iterdir()):
            if not p.is_dir():
                continue
            left = p / "left.png"
            right = p / "right.png"
            if left.exists() and right.exists():
                rec = {"id": p.name, "left": left, "right": right}
                if self.load_disp:
                    dpath = p / f"{self.disp_side}_disp.png"
                    if not dpath.exists():
                        raise FileNotFoundError(f"Missing disparity file: {dpath}")
                    rec["disp"] = dpath
                scenes.append(rec)
        if not scenes:
            raise RuntimeError(f"No scenes found under {self.root_dir}")

        # Deterministic split
        g = torch.Generator().manual_seed(random_seed)
        perm = torch.randperm(len(scenes), generator=g).tolist()
        cut = int(len(perm) * (1.0 - val_ratio))
        if split == "train":
            keep = perm[:cut]
        elif split == "val":
            keep = perm[cut:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.samples = [scenes[i] for i in keep]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.samples[idx]
        left_img = Image.open(rec["left"]).convert("RGB")
        right_img = Image.open(rec["right"]).convert("RGB")

        disp_np = None
        if self.load_disp:
            disp_np = load_disp_png(rec["disp"], divisor=self.disp_divisor)

        left_img, right_img, disp_np = resize_pair_and_disp(left_img, right_img, disp_np, self.resize_hw)

        left_t = pil_to_tensor(left_img)
        right_t = pil_to_tensor(right_img)

        sample: Dict[str, torch.Tensor | str] = {
            "left": left_t,
            "right": right_t,
            "id": rec["id"],
        }
        if disp_np is not None:
            disp_t = torch.from_numpy(disp_np).float()
            mask = disp_t > 0
            sample["disp"] = disp_t
            sample["mask"] = mask

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ---------- Quick smoke test ----------
if __name__ == "__main__":
    root = "data/instereo2k_sample"
    ds = InStereo2KDataset(root, split="train", val_ratio=0.1, load_disp=True,
                           disp_side="left", disp_divisor=100.0, resize_hw=(540, 960))
    s = ds[0]
    print("ID:", s["id"], "| left:", tuple(s["left"].shape), "| right:", tuple(s["right"].shape),
          "| disp:", None if "disp" not in s else tuple(s["disp"].shape))
