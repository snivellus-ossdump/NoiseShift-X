"""Qualitative figure: noisy / baseline / +aug / ground truth, zoom crops and error maps on a held-out-camera test block.
  python visualize.py --cam S6 [--scene 0] [--block 40]   ->  results/visual_S6.png"""
import argparse, os, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from data import index_sidd, split_scenes, load_png, fixed_eval_blocks
from model import SmallDenoiser
from metrics import psnr

# Load a trained checkpoint in evaluation mode.
def load_model(name, dev):
    m = SmallDenoiser().to(dev); m.load_state_dict(torch.load(os.path.join(C.CKPT_DIR, name + ".pt"), map_location=dev)["m"]); return m.eval()

# Denoise one 256 x 256 block.
@torch.no_grad()
def run(m, blk, dev):
    x = torch.from_numpy(blk[None]).permute(0, 3, 1, 2).float().div(255).to(dev)
    return m(x).clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()[0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cam", required=True); ap.add_argument("--scene", type=int, default=0); ap.add_argument("--block", type=int, default=40)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Pick a test scene of the chosen camera and one block of its fixed grid.
    items = [it for it in split_scenes(index_sidd(C.SIDD_SMALL_ROOT), C.TEST_FRAC, C.SEED)[1] if it["camera"] == a.cam]
    it = items[a.scene % len(items)]; bl = fixed_eval_blocks(load_png(it["noisy"]), load_png(it["gt"]), C.EVAL_BLOCK); n, g = bl[a.block % len(bl)]
    # The same block through the baseline and the augmented hold-out models (neither saw this camera).
    base, aug = run(load_model(f"base_hold_{a.cam}", dev), n, dev), run(load_model(f"aug_hold_{a.cam}", dev), n, dev)
    # 3 x 4 figure: row 1 = full block, row 2 = 64 x 64 zoom, row 3 = absolute error against the ground truth.
    ims = [n, base, aug, g]; titles = [f"Noisy ({psnr(n, g):.2f} dB)", f"Baseline ({psnr(base, g):.2f} dB)", f"+PG aug ({psnr(aug, g):.2f} dB)", "Ground truth"]
    fig, ax = plt.subplots(3, 4, figsize=(13, 10)); y0, x0, z = 96, 96, 64
    for j in range(4):
        ax[0, j].imshow(ims[j]); ax[0, j].set_title(titles[j], fontsize=10); ax[1, j].imshow(ims[j][y0:y0+z, x0:x0+z], interpolation="nearest"); ax[1, j].set_title("zoom", fontsize=8)
        if j < 3: ax[2, j].imshow(np.abs(ims[j].astype(float) - g.astype(float)).mean(-1), cmap="inferno", vmin=0, vmax=40); ax[2, j].set_title("|error| vs GT", fontsize=8)
    ax[2, 3].axis("off")
    for r in ax:
        for c in r: c.set_xticks([]); c.set_yticks([])
    fig.suptitle(f"Held-out camera {a.cam}: models never saw this camera"); fig.tight_layout(); os.makedirs(C.RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(C.RESULTS_DIR, f"visual_{a.cam}.png"), dpi=150); print("saved")

if __name__ == "__main__":
    main()
