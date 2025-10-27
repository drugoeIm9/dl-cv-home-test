import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

import imageio.v2 as imageio
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from torchvision import transforms as T


import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

def ssim(img1, img2, C1=0.01**2, C2=0.03**2, window_size=3):
    mu1 = F.avg_pool2d(img1, kernel_size=window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, kernel_size=window_size, stride=1, padding=window_size//2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()

def warp_right_to_left(img_right, disp):
    """
    Warp right image to left view using disparity (pixel shift).
    img_right: [B,C,H,W] (C can be 1 or 3)
    disp: [B,1,H,W] disparity in pixels (positive means pixel shift to left)
    returns warped_right [B,C,H,W]
    """
    B, C, H, W = img_right.shape

    xs = torch.linspace(-1, 1, W, device=img_right.device)
    ys = torch.linspace(-1, 1, H, device=img_right.device)
    grid_x = xs.unsqueeze(0).repeat(H, 1)
    grid_y = ys.unsqueeze(1).repeat(1, W)
    grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1)

    disp_norm = disp.squeeze(1) * 2.0 / (W - 1)
    grid = grid.to(img_right.device)
    grid[:, :, :, 0] = grid[:, :, :, 0] - disp_norm

    warped = F.grid_sample(img_right, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped

def edge_aware_smoothness(disp, img):
    dx_disp = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1])
    dy_disp = torch.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])
    dx_img = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), dim=1, keepdim=True)
    dy_img = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), dim=1, keepdim=True)
    wx = torch.exp(-dx_img)
    wy = torch.exp(-dy_img)
    s = (dx_disp * wx).mean() + (dy_disp * wy).mean()
    return s

class InStereoSample(Dataset):
    def __init__(self, root, transform=None, resize=(512,256), disp_scale=1.0, pad_to=16, augment=False, sample_list=None):
        """
        root: data root with many subfolders each containing left.png,right.png,left_disp.png
        resize: (W,H) target resize
        pad_to: pad spatial dims to multiple of pad_to (typically 16)
        augment: apply data augmentation (for training)
        sample_list: (optional) pre-filtered list of (lp, rp, dp) tuples
        """
        self.samples = []
        if sample_list:
            self.samples = sample_list
        else:
            for folder in sorted(glob(os.path.join(root, '*'))):
                if os.path.isdir(folder):
                    lp = os.path.join(folder, 'left.png')
                    rp = os.path.join(folder, 'right.png')
                    dp = os.path.join(folder, 'left_disp.png')
                    if os.path.exists(lp) and os.path.exists(rp) and os.path.exists(dp):
                        self.samples.append((lp, rp, dp))
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {root} with left.png/right.png/left_disp.png")
            
        self.transform = transform
        self.resize = resize
        self.disp_scale = disp_scale
        self.pad_to = pad_to
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _load_image(self, p, resize):
        im = Image.open(p).convert('L')
        if resize is not None:
            im = im.resize((resize[0], resize[1]), Image.BILINEAR)
        arr = np.array(im).astype(np.float32) / 255.0
        return arr

    def _load_disp(self, p, resize):
        d = imageio.imread(p)
        d = np.array(d).astype(np.float32)
        if resize is not None:
            orig_w = d.shape[1]
            scale_w = resize[0] / float(orig_w)
            d = cv2.resize(d, dsize=(resize[0], resize[1]), interpolation=cv2.INTER_LINEAR)
            d = d * scale_w
            
        d = d * float(self.disp_scale)
        if d.ndim == 3:
            d = d[:, :, 0]
        return d

    def __getitem__(self, idx):
        lp, rp, dp = self.samples[idx]
        left = self._load_image(lp, self.resize)
        right = self._load_image(rp, self.resize)
        disp = self._load_disp(dp, self.resize)

        if self.augment:
            if torch.rand(1) > 0.5:
                left = np.ascontiguousarray(np.flipud(left))
                right = np.ascontiguousarray(np.flipud(right))
                disp = np.ascontiguousarray(np.flipud(disp))

            if torch.rand(1) > 0.5:
                gamma = torch.rand(1).item() * 1.0 + 0.5
                left = left ** gamma
            if torch.rand(1) > 0.5:
                brightness = torch.rand(1).item() * 0.4 - 0.2
                contrast = torch.rand(1).item() * 0.4 + 0.8
                left = np.clip(left * contrast + brightness, 0, 1)
            
            if torch.rand(1) > 0.5:
                gamma = torch.rand(1).item() * 1.0 + 0.5
                right = right ** gamma
            if torch.rand(1) > 0.5:
                brightness = torch.rand(1).item() * 0.4 - 0.2
                contrast = torch.rand(1).item() * 0.4 + 0.8
                right = np.clip(right * contrast + brightness, 0, 1)

        H0, W0 = left.shape[:2]
        mult = self.pad_to
        pad_h = (mult - (H0 % mult)) % mult
        pad_w = (mult - (W0 % mult)) % mult

        if pad_h > 0 or pad_w > 0:
            left_p = cv2.copyMakeBorder(left, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT)
            right_p = cv2.copyMakeBorder(right, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT)
            disp_p = cv2.copyMakeBorder(disp, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0.0)
        else:
            left_p, right_p, disp_p = left, right, disp

        left_t  = torch.from_numpy(left_p).unsqueeze(0).float()
        right_t = torch.from_numpy(right_p).unsqueeze(0).float()
        disp_t  = torch.from_numpy(disp_p).unsqueeze(0).float()

        return {
            'left': left_t,
            'right': right_t,
            'disp': disp_t,
            'orig_h': H0,
            'orig_w': W0,
            'pad_h': pad_h,
            'pad_w': pad_w
        }

def to_rgb_like(x):
    if x.size(1) == 1:
        return x.repeat(1, 3, 1, 1)
    return x

def compute_losses(pred_disp, gt_disp_c, left_c, right_c, l1_loss_fn, lambda_l1, lambda_ssim, lambda_smooth):
    """Вспомогательная функция для расчета набора потерь."""

    gt_disp_clamped = torch.clamp(gt_disp_c, 0.0, 96.0)
    loss_l1 = l1_loss_fn(pred_disp, gt_disp_clamped)

    warped_r = warp_right_to_left(right_c, pred_disp)
    loss_ssim = ssim(warped_r, left_c)
    
    loss_smooth = edge_aware_smoothness(pred_disp, left_c)

    total_loss = loss_l1 * lambda_l1 + loss_ssim * lambda_ssim + loss_smooth * lambda_smooth
    
    return total_loss, loss_l1, loss_ssim, loss_smooth

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[I] device:", device)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, 'checkpoints'); os.makedirs(ckpt_dir, exist_ok=True)
    samples_dir = os.path.join(args.out_dir, 'samples'); os.makedirs(samples_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'logs'))


    temp_dataset = InStereoSample(args.data_root, resize=(args.width, args.height), disp_scale=args.disp_scale, pad_to=16)
    all_samples = temp_dataset.samples
    np.random.shuffle(all_samples)
    
    n = len(all_samples)
    val_n = max(1, int(n * args.val_split))
    train_n = n - val_n
    
    train_samples = all_samples[val_n:]
    val_samples = all_samples[:val_n]
    
    print(f"[I] Data split: {train_n} train, {val_n} val samples")

    train_set = InStereoSample(args.data_root, resize=(args.width, args.height), disp_scale=args.disp_scale, 
                               pad_to=16, augment=True, sample_list=train_samples)
    val_set = InStereoSample(args.data_root, resize=(args.width, args.height), disp_scale=args.disp_scale, 
                             pad_to=16, augment=False, sample_list=val_samples)
    # -----------------------------------------------------------

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=='cuda'), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(0, min(2, args.num_workers)), pin_memory=(device.type=='cuda'))

    print("[I] importing BGNet model...")
    from models.BGNet.models.bgnet_plus import BGNet_Plus
    model = BGNet_Plus().to(device)

    if args.pretrained:
        print("[I] loading pretrained:", args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        try:
            model.load_state_dict(state, strict=False)
            print("[I] loaded state_dict (strict=False)")
        except Exception as e:
            print("[W] partial load failed:", e)
            if isinstance(state, dict):
                model_state = model.state_dict()
                new_state = {}
                for k, v in state.items():
                    if k in model_state and model_state[k].shape == v.shape:
                        new_state[k] = v
                model_state.update(new_state)
                model.load_state_dict(model_state)
                print(f"[I] partial load: {len(new_state)} params matched")
            else:
                raise

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    l1_loss = nn.SmoothL1Loss(reduction='mean')

    global_step = 0
    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            gt_disp = batch['disp'].to(device)
            orig_h = batch['orig_h'][0].item()
            orig_w = batch['orig_w'][0].item()

            optimizer.zero_grad()
            
            pred_disp_full, pred_disp_coarse_up = model(left, right)
            
            if pred_disp_full.dim() == 3:
                pred_disp_full = pred_disp_full.unsqueeze(1)
            if pred_disp_coarse_up.dim() == 3:
                pred_disp_coarse_up = pred_disp_coarse_up.unsqueeze(1)

            pred_disp = F.interpolate(pred_disp_full, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred_disp_coarse = F.interpolate(pred_disp_coarse_up, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

            gt_disp_c = gt_disp[..., :orig_h, :orig_w]
            left_c = left[..., :orig_h, :orig_w]
            right_c = right[..., :orig_h, :orig_w]

            loss_1, l1_1, ssim_1, smooth_1 = compute_losses(
                pred_disp, gt_disp_c, left_c, right_c, l1_loss,
                args.lambda_l1, args.lambda_ssim, args.lambda_smooth
            )
            
            loss_2, l1_2, ssim_2, smooth_2 = compute_losses(
                pred_disp_coarse, gt_disp_c, left_c, right_c, l1_loss,
                args.lambda_l1, args.lambda_ssim, args.lambda_smooth
            )

            total_loss = 0.7 * loss_1 + 0.3 * loss_2

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            global_step += 1
            if global_step % args.log_interval == 0:
                writer.add_scalar('train/total_loss', total_loss.item(), global_step)
                writer.add_scalar('train/loss_refined', loss_1.item(), global_step)
                writer.add_scalar('train/loss_coarse', loss_2.item(), global_step)
                writer.add_scalar('train/l1_refined', l1_1.item(), global_step)
                writer.add_scalar('train/ssim_refined', ssim_1.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

            pbar.set_postfix({'loss': total_loss.item(), 'l1': l1_1.item(), 'ssim': ssim_1.item()})

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_l1_total = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                left = batch['left'].to(device)
                right = batch['right'].to(device)
                gt_disp = batch['disp'].to(device)
                orig_h = batch['orig_h'][0].item()
                orig_w = batch['orig_w'][0].item()

                pred_disp_full, pred_disp_coarse_up = model(left, right)
                
                if pred_disp_full.dim() == 3:
                    pred_disp_full = pred_disp_full.unsqueeze(1)
                if pred_disp_coarse_up.dim() == 3:
                    pred_disp_coarse_up = pred_disp_coarse_up.unsqueeze(1)

                pred_disp = F.interpolate(pred_disp_full, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                pred_disp_coarse = F.interpolate(pred_disp_coarse_up, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

                gt_disp_c = gt_disp[..., :orig_h, :orig_w]
                left_c = left[..., :orig_h, :orig_w]
                right_c = right[..., :orig_h, :orig_w]
                
                loss_1, l1_1, _, _ = compute_losses(
                    pred_disp, gt_disp_c, left_c, right_c, l1_loss,
                    args.lambda_l1, args.lambda_ssim, args.lambda_smooth
                )
                loss_2, _, _, _ = compute_losses(
                    pred_disp_coarse, gt_disp_c, left_c, right_c, l1_loss,
                    args.lambda_l1, args.lambda_ssim, args.lambda_smooth
                )

                tl = 0.7 * loss_1 + 0.3 * loss_2
                val_loss += tl.item()
                val_l1_total += l1_1.item()

                if i == 0:
                    disp_vis = pred_disp.clone()
                    mn = disp_vis.amin(dim=(2,3), keepdim=True)
                    mx = disp_vis.amax(dim=(2,3), keepdim=True)
                    disp_vis = (disp_vis - mn) / (mx - mn + 1e-6)
                    out_img = torch.cat([to_rgb_like(left_c), to_rgb_like(right_c), to_rgb_like(disp_vis)], dim=0)
                    save_image(out_img, os.path.join(samples_dir, f'epoch{epoch:03d}_sample.png'), nrow=left_c.size(0), normalize=False)

        val_loss = val_loss / max(1, len(val_loader))
        val_l1 = val_l1_total / max(1, len(val_loader))
        writer.add_scalar('val/total_loss', val_loss, epoch)
        writer.add_scalar('val/l1_error', val_l1, epoch)
        print(f"[EPOCH {epoch}] val_loss={val_loss:.6f}  val_l1(px)={val_l1:.4f}")

        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt_path)
        
        if val_l1 < best_val:
            best_val = val_l1
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, os.path.join(ckpt_dir, 'best.pth'))
            print(f"[I] new best saved (val_l1 {best_val:g} px)")

    writer.close()
    print("Training finished. Outputs saved to", args.out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, required=False, default='data/instereo2k_sample', help='path to instereo2k_sample root')
    p.add_argument('--bgnet-root', type=str, required=False, help='(not used) bgnet repo root — adjust import if needed')
    p.add_argument('--pretrained', type=str, default='models/BGNet/models/kitti_12_BGNet_Plus.pth', help='path to BGNet pretrained .pth (optional)')
    p.add_argument('--out-dir', type=str, default='results/second_task', help='where to write checkpoints/logs')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4, help='dataloader workers (0 recommended in containers, 4+ for speed)')
    p.add_argument('--lr', type=float, default=1e-7)
    p.add_argument('--width', type=int, default=512, help='resize width')
    p.add_argument('--height', type=int, default=256, help='resize height')
    p.add_argument('--disp-scale', type=float, default=0.00390625, help='multiply loaded disparity by this (если disp сохранен как float * 256, используйте 1/256)')
    p.add_argument('--val-split', type=float, default=0.1, help='fraction to use for validation')
    p.add_argument('--lambda-l1', type=float, default=0.8, help='Weight for L1 loss')
    p.add_argument('--lambda-ssim', type=float, default=0.3, help='Weight for SSIM loss')
    p.add_argument('--lambda-smooth', type=float, default=0.05, help='Weight for Smoothness loss')
    p.add_argument('--log-interval', type=int, default=20)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)