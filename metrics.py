"""PSNR / SSIM on uint8 sRGB blocks, plus the cross-camera drop."""
import numpy as np
from skimage.metrics import structural_similarity

def psnr(a, b):
    # Scale to [0, 1] so the peak value is 1: PSNR = 10 * log10(1 / MSE).
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))

def ssim(a, b):
    # scikit-image SSIM on the RGB block (channel_axis=-1), default 7x7 window, pixel range 0-255.
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255))

def eval_blocks(pred_blocks, gt_blocks):
    """Mean PSNR/SSIM over blocks (per-block, then averaged)."""
    assert len(pred_blocks) == len(gt_blocks) and len(pred_blocks) > 0
    # Metric per block first, then the mean over blocks (not the PSNR of the pooled MSE).
    p = [psnr(a, b) for a, b in zip(pred_blocks, gt_blocks)]
    s = [ssim(a, b) for a, b in zip(pred_blocks, gt_blocks)]
    return float(np.mean(p)), float(np.mean(s))

def cross_camera_drop(seen_psnr, unseen_psnr):
    """Drop = performance on seen cameras minus unseen camera. Lower is better."""
    return seen_psnr - unseen_psnr

def load_official_blocks(noisy_mat, gt_mat):
    """Official SIDD validation blocks: arrays of shape (40, 32, 256, 256, 3)."""
    # The official files are MATLAB arrays of shape (40 images, 32 blocks, 256, 256, 3).
    from scipy.io import loadmat
    n = loadmat(noisy_mat)["ValidationNoisyBlocksSrgb"]
    g = loadmat(gt_mat)["ValidationGtBlocksSrgb"]
    assert n.shape == g.shape and n.shape[-3:] == (256, 256, 3), f"unexpected shape {n.shape}"
    # Flatten (image, block) into a single list of 1280 blocks.
    return n.reshape(-1, 256, 256, 3), g.reshape(-1, 256, 256, 3)
