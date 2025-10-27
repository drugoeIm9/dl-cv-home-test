import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import cv2
import torch
import onnxruntime as ort
import tempfile
import shutil

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from models.FoundationStereo.core.utils.utils import InputPadder
from models.FoundationStereo.Utils import *

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_image_pair(left_path: str, right_path: str, scale: float = 1.0):
    """Load and resize images. Returns (img0_np, img1_np, img0_ori) where np images are float32 HWC."""
    img0 = imageio.imread(left_path)
    img1 = imageio.imread(right_path)
    assert scale <= 1.0
    if scale != 1.0:
        H0, W0 = img0.shape[:2]
        newW = int(W0 * scale + 0.5)
        newH = int(H0 * scale + 0.5)
        img0 = cv2.resize(img0, (newW, newH), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, (newW, newH), interpolation=cv2.INTER_LINEAR)
    img0_ori = img0.copy()
    return img0.astype(np.float32), img1.astype(np.float32), img0_ori


def make_onnx_session(onnx_path: str):
    """Create ONNX Runtime session, prefer CUDA if available."""
    providers = []
    try:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    except Exception:
        providers = ["CPUExecutionProvider"]

    logging.info("Creating ONNX InferenceSession for '%s' with providers=%s", onnx_path, providers)
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    return sess


def normalize_input(img_np: np.ndarray, norm: str = "none", mean=None, std=None):
    """
    img_np: NHWC float32 or HWC (expects 0..255)
    norm:
      - none: keep 0..255
      - 01: divide by 255 -> 0..1
      - imagenet: divide by 255 and subtract mean/std (mean,std provided or default ImageNet)
    """
    if norm == "none":
        return img_np
    if norm == "01":
        return img_np / 255.0
    if norm == "imagenet":
        if mean is None:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        if std is None:
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img_np / 255.0
        img = (img - mean[None, None, :]) / std[None, None, :]
        return img
    raise ValueError("Unknown norm option: " + str(norm))


def infer_onnx(session: ort.InferenceSession, img0_t: torch.Tensor, img1_t: torch.Tensor, padder: InputPadder,
               norm: str = "none"):
    """
    Run ONNX inference and return disparity HxW (float32).
    img?_t: torch.Tensor on CPU shape (1, C, H, W) float32 (values as loaded)
    padder: InputPadder instance
    """
    img0_p, img1_p = padder.pad(img0_t, img1_t)  # torch tensors on CPU
    img0_np = img0_p.numpy().astype(np.float32)
    img1_np = img1_p.numpy().astype(np.float32)

    if norm != "none":
        img0_nhwc = img0_np.transpose(0, 2, 3, 1)
        img1_nhwc = img1_np.transpose(0, 2, 3, 1)
        img0_nhwc = normalize_input(img0_nhwc, norm=norm)
        img1_nhwc = normalize_input(img1_nhwc, norm=norm)
        img0_np = img0_nhwc.transpose(0, 3, 1, 2)
        img1_np = img1_nhwc.transpose(0, 3, 1, 2)

    in_meta = session.get_inputs()
    in_names = [i.name for i in in_meta]
    logging.debug("ONNX model inputs metadata: %s", [(i.name, i.shape) for i in in_meta])
    out_meta = session.get_outputs()
    logging.debug("ONNX model outputs metadata: %s", [(o.name, o.shape) for o in out_meta])

    feed = {}
    if len(in_names) >= 2:
        feed[in_names[0]] = img0_np
        feed[in_names[1]] = img1_np
    else:
        name = in_names[0]
        shape = in_meta[0].shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 4 and (shape[1] == 6 or shape[1] == 3*2):
            feed[name] = np.concatenate([img0_np, img1_np], axis=1)
        else:
            feed[name] = img0_np
            logging.warning("ONNX model has single input and not 6 channels; feeding left only.")

    outs = session.run(None, feed)
    if len(outs) == 0:
        raise RuntimeError("ONNX session returned no outputs")
    out0 = outs[0]

    if out0.ndim == 4:
        disp_raw = out0[0, 0]
    elif out0.ndim == 3:
        if out0.shape[0] == 1:
            disp_raw = out0[0]
        else:
            disp_raw = out0[0]
    elif out0.ndim == 2:
        disp_raw = out0
    else:
        raise RuntimeError(f"Unexpected ONNX output dims: {out0.shape}")

    disp_t = torch.from_numpy(disp_raw.astype(np.float32))[None, None, :, :]
    disp_unp = padder.unpad(disp_t)
    disp_np = disp_unp.cpu().numpy().reshape(disp_unp.shape[2], disp_unp.shape[3])
    return disp_np


def find_left_right_in_folder(folder: Path):
    """Find left/right images in folder robustly."""
    files = [p for p in folder.iterdir() if p.is_file()]
    mapping = {p.name.lower(): p for p in files}
    left_keys = ["left.png", "left.jpg", "left.jpeg", "l.png", "l.jpg"]
    right_keys = ["right.png", "right.jpg", "right.jpeg", "r.png", "r.jpg"]
    left = None
    right = None
    for k in left_keys:
        if k in mapping:
            left = mapping[k]; break
    for k in right_keys:
        if k in mapping:
            right = mapping[k]; break
    if left is None or right is None:
        imgs = sorted([p for p in files if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if len(imgs) >= 2:
            if left is None: left = imgs[0]
            if right is None:
                right = imgs[1] if imgs[1] != left else (imgs[2] if len(imgs) > 2 else None)
    return left, right


def process_pair_and_save(session, left_path: Path, right_path: Path, out_dir: Path, args, verbose=True):
    """
    Run inference on a single left/right pair and save outputs in out_dir.
    Returns True on success.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        img0_np, img1_np, img0_ori = prepare_image_pair(str(left_path), str(right_path), scale=args.scale)
        H_orig, W_orig = img0_np.shape[:2]
        logging.info("Processing pair: %s / %s (orig H=%d W=%d)", left_path.name, right_path.name, H_orig, W_orig)

        K_resize_scale_w = 1.0
        K_resize_scale_h = 1.0
        try:
            in_meta = session.get_inputs()
            target_h = None
            target_w = None
            for imeta in in_meta:
                shp = imeta.shape
                if isinstance(shp, (list, tuple)) and len(shp) >= 4:
                    h = shp[2]; w = shp[3]
                    if isinstance(h, int) and isinstance(w, int):
                        target_h, target_w = int(h), int(w)
                        break
            if target_h is not None and target_w is not None:
                if (H_orig != target_h) or (W_orig != target_w):
                    logging.info("Resizing (%d,%d)->(%d,%d) to match ONNX", H_orig, W_orig, target_h, target_w)
                    img0_np = cv2.resize(img0_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    img1_np = cv2.resize(img1_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    img0_ori = img0_np.copy().astype(np.uint8)
                    K_resize_scale_w = float(target_w) / float(W_orig)
                    K_resize_scale_h = float(target_h) / float(H_orig)
            else:
                logging.debug("ONNX spatial dims are dynamic or unspecified; not forcing resize.")
        except Exception as e:
            logging.debug("Failed to read onnx input shapes: %s", e)

        img0_t = torch.from_numpy(img0_np.transpose(2, 0, 1))[None].float()
        img1_t = torch.from_numpy(img1_np.transpose(2, 0, 1))[None].float()

        padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)

        disp = infer_onnx(session, img0_t, img1_t, padder, norm=args.norm)

        if args.max_disp_scale != 1.0:
            disp = disp * float(args.max_disp_scale)

        try:
            vis = vis_disparity(disp)
        except Exception:
            vmax = np.nanmax(disp[np.isfinite(disp)]) if np.any(np.isfinite(disp)) else 1.0
            if vmax <= 0:
                vmax = 1.0
            vis = (disp / vmax * 255.0).clip(0, 255).astype(np.uint8)
            if vis.ndim == 2:
                vis = np.stack([vis, vis, vis], axis=-1)

        vis_concat = np.concatenate([img0_ori.astype(np.uint8), vis], axis=1)
        imageio.imwrite(str(out_dir / "vis.png"), vis_concat)

        np.save(str(out_dir / "left_disp.npy"), disp.astype(np.float32))
        vmax = np.nanmax(disp[np.isfinite(disp)]) if np.any(np.isfinite(disp)) else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        disp_16 = (disp / vmax * 65535.0).clip(0, 65535).astype(np.uint16)
        imageio.imwrite(str(out_dir / "left_disp.png"), disp_16)

        if args.remove_invisible:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        if args.get_pc:
            if not Path(args.intrinsic_file).exists():
                logging.warning("Intrinsic file not found: %s; skipping depth/pc", args.intrinsic_file)
            else:
                try:
                    with open(args.intrinsic_file, 'r') as f:
                        lines = f.readlines()
                        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
                        baseline = float(lines[1])
                    K[0, :] *= (args.scale * K_resize_scale_w)
                    K[1, :] *= (args.scale * K_resize_scale_h)
                    depth = (K[0, 0] * baseline) / disp
                    np.save(str(out_dir / "depth_meter.npy"), depth.astype(np.float32))

                    xyz_map = depth2xyzmap(depth, K)
                    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
                    keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
                    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
                    pcd = pcd.select_by_index(keep_ids)
                    ply_path = str(out_dir / "cloud.ply")
                    o3d.io.write_point_cloud(ply_path, pcd)
                    if args.denoise_cloud and _HAS_O3D:
                        cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points,
                                                            radius=args.denoise_radius)
                        inlier_cloud = pcd.select_by_index(ind)
                        o3d.io.write_point_cloud(str(out_dir / "cloud_denoise.ply"), inlier_cloud)
                except Exception as e:
                    logging.exception("Failed to compute/save depth/pc: %s", e)

        logging.info("Saved outputs to %s", out_dir)
        return True
    except Exception as e:
        logging.exception("Failed inference for pair %s / %s : %s", left_path, right_path, e)
        return False


def process_dataset(session, dataset_root: Path, out_root: Path, args):
    scenes = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if args.max_samples:
        scenes = scenes[: args.max_samples]
    logging.info("Found %d scenes to process under %s", len(scenes), dataset_root)

    for scene in scenes:
        left, right = find_left_right_in_folder(scene)
        if left is None or right is None:
            logging.warning("Skipping scene %s — left/right not found", scene)
            continue
        out_scene = out_root / scene.name
        if args.skip_existing and (out_scene / "left_disp.npy").exists() and (not args.use_swap_for_right or (out_scene / "right_disp.npy").exists()):
            logging.info("Skipping %s — outputs already exist (use --skip_existing)", scene)
            continue

        ok = process_pair_and_save(session, left, right, out_scene, args)
        if not ok:
            logging.error("Scene %s failed (left). Check logs", scene)
            continue

        if args.use_swap_for_right:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                ok2 = process_pair_and_save(session, right, left, td_path, args)
                if not ok2:
                    logging.error("Swap run failed for %s (right).", scene)
                else:
                    if (td_path / "left_disp.npy").exists():
                        shutil.move(str(td_path / "left_disp.npy"), str(out_scene / "right_disp.npy"))
                    if (td_path / "left_disp.png").exists():
                        shutil.move(str(td_path / "left_disp.png"), str(out_scene / "right_disp.png"))
                    if (td_path / "vis.png").exists():
                        shutil.move(str(td_path / "vis.png"), str(out_scene / "vis_swap.png"))
                    # move depth/pc files if present too (optional)
                    if (td_path / "depth_meter.npy").exists():
                        shutil.move(str(td_path / "depth_meter.npy"), str(out_scene / "depth_meter_swap.npy"))
                    if (td_path / "cloud.ply").exists():
                        shutil.move(str(td_path / "cloud.ply"), str(out_scene / "cloud_swap.ply"))

    logging.info("Batch processing finished. Results in %s", out_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
    parser.add_argument('--dataset_root', default=None, type=str, help="If set, process every subfolder as a scene (batch mode).")
    parser.add_argument('--out_root', default=f'{code_dir}/../output_batch/', type=str, help="Output root when using --dataset_root.")
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str,
                        help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/foundationstereo.onnx', type=str,
                        help='ONNX model path (.onnx)')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results (single-pair)')
    parser.add_argument('--scale', default=1.0, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='kept for API parity, not used for ONNX')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='kept for API parity, may be unused by ONNX')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='nb_points for radius removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius for outlier removal')
    parser.add_argument('--norm', type=str, default='none', choices=('none', '01', 'imagenet'),
                        help="Normalization to apply to input images before feeding ONNX (default: none=0..255)")
    parser.add_argument('--max_disp_scale', type=float, default=1.0,
                        help="Optional: scale disparity output (if your ONNX was exported with scaled disparity)")
    parser.add_argument('--use_swap_for_right', action='store_true', help="Also run swapped pair to obtain right disparity")
    parser.add_argument('--max_samples', type=int, default=None, help="Limit number of scenes (for testing).")
    parser.add_argument('--skip_existing', action='store_true', help="Skip scenes where outputs already exist")
    args = parser.parse_args()

    logging.info("Args: %s", args)
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    if not Path(args.ckpt_dir).exists():
        logging.error("ONNX file not found: %s", args.ckpt_dir)
        return

    session = make_onnx_session(args.ckpt_dir)
    logging.info("ONNX model inputs: %s", [(i.name, i.shape) for i in session.get_inputs()])
    logging.info("ONNX model outputs: %s", [(o.name, o.shape) for o in session.get_outputs()])

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        out_root = Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        process_dataset(session, dataset_root, out_root, args)
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        left = Path(args.left_file)
        right = Path(args.right_file)
        if not left.exists() or not right.exists():
            logging.error("left or right file not found: %s , %s", left, right)
            return
        process_pair_and_save(session, left, right, out_dir, args)

    logging.info("All done.")


if __name__ == "__main__":
    main()
