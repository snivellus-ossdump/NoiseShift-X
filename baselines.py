"""Classical vs learned denoisers on the SAME fixed test blocks (answers: which methods, how do they compare).

  pip install opencv-python bm3d
  python baselines.py                       # all cameras, 4 blocks per test scene
  python baselines.py --cams S6 --per_scene 2 --no_bm3d     # quick check

For each camera K: noisy input (no denoising), Gaussian blur, median filter, Non-Local Means, BM3D, and (if their
checkpoints exist) the held-out-camera models base_hold_K / aug_hold_K, all scored on identical blocks. Blocks are spread
evenly over each test image, away from the corners. Classical methods use fixed untuned settings and a blind per-block
noise estimate (no ground truth is used to tune anything). Writes results/baselines.csv and results/baselines.md.
"""
import argparse, os, csv, time, numpy as np, cv2
from skimage.restoration import estimate_sigma
import config as C
from data import index_sidd, split_scenes, load_png, fixed_eval_blocks
from metrics import eval_blocks

# Blind noise-level estimate (wavelet MAD, scikit-image) for one block; no ground truth is used.
def sigma01(blk):   # blind noise estimate in [0,1] units, averaged over channels
    return float(max(estimate_sigma(blk.astype(np.float32) / 255.0, channel_axis=-1, average_sigmas=True), 1e-3))

# Classical methods: each takes and returns a uint8 RGB block.
def m_gauss(b):  return cv2.GaussianBlur(b, (0, 0), 1.0)
def m_median(b): return cv2.medianBlur(b, 3)
def m_nlm(b, sig=None):
    h = max(3.0, (sig if sig is not None else sigma01(b)) * 255.0)
    return cv2.fastNlMeansDenoisingColored(b, None, h, h, 7, 21)
# BM3D needs the 'bm3d' package; it is imported lazily so the script still runs without it.
def make_bm3d():
    import bm3d
    return lambda b, sig=None: np.clip(bm3d.bm3d_rgb(b.astype(np.float32) / 255.0, sig if sig is not None else sigma01(b)) * 255.0, 0, 255).round().astype(np.uint8)

# Wrap a trained DnCNN checkpoint as a function block -> denoised block (None if no checkpoint exists).
def learned(name):
    path = os.path.join(C.CKPT_DIR, name + ".pt")
    if not os.path.exists(path): return None
    import torch
    from model import SmallDenoiser
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SmallDenoiser().to(dev); m.load_state_dict(torch.load(path, map_location=dev)["m"]); m.eval()
    @torch.no_grad()
    def f(b):
        x = torch.from_numpy(np.ascontiguousarray(b)[None]).permute(0, 3, 1, 2).float().div(255).to(dev)
        return m(x).clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()[0]
    return f

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cams", nargs="*", default=C.CAMERAS)
    ap.add_argument("--per_scene", type=int, default=4); ap.add_argument("--no_bm3d", action="store_true")
    ap.add_argument("--oracle_sigma", action="store_true", help="also run NLM/BM3D with the TRUE per-block noise sigma (uses ground truth; an upper bound, not a fair blind method)")
    a = ap.parse_args(); os.makedirs(C.RESULTS_DIR, exist_ok=True)
    # Only the reserved test scenes are used.
    test_items = split_scenes(index_sidd(C.SIDD_SMALL_ROOT), C.TEST_FRAC, C.SEED)[1]
    rows = []
    for cam in a.cams:
        nb, gb = [], []
        # Pick `per_scene` blocks per test image, evenly spread over its block grid (away from the corners).
        for it in [i for i in test_items if i["camera"] == cam]:
            bl = fixed_eval_blocks(load_png(it["noisy"]), load_png(it["gt"]), C.EVAL_BLOCK)
            for i in np.unique(((np.arange(a.per_scene) + 0.5) / a.per_scene * len(bl)).astype(int)): nb.append(bl[i][0]); gb.append(bl[i][1])   # evenly spread, away from the corners
        # Oracle noise level: std of (noisy - ground truth) per block. It uses the answer, so it is only an upper bound.
        sig = [float(np.std(n.astype(np.float32) - g.astype(np.float32)) / 255.0) for n, g in zip(nb, gb)]
        # Each entry: (name, function, uses_oracle_sigma).
        methods = [("Noisy input", lambda b: b, False), ("Gaussian blur (s=1)", m_gauss, False), ("Median 3x3", m_median, False), ("Non-Local Means (blind sigma)", m_nlm, False)]
        if a.oracle_sigma: methods.append(("Non-Local Means (oracle sigma)", m_nlm, True))
        if not a.no_bm3d:
            try:
                bm = make_bm3d(); methods.append(("BM3D (blind sigma)", bm, False))
                if a.oracle_sigma: methods.append(("BM3D (oracle sigma)", bm, True))
            except ImportError: print("bm3d not installed: skipping BM3D (pip install bm3d)")
        for tag, nm in (("base_hold_%s", "DnCNN baseline (unseen cam)"), ("aug_hold_%s", "DnCNN + noise aug (unseen cam)")):
            f = learned(tag % cam)
            if f is not None: methods.append((nm, f, False))
        # Score every method on exactly the same blocks.
        for name, f, oracle in methods:
            t0 = time.time(); p, s = eval_blocks([f(b, sg) if oracle else f(b) for b, sg in zip(nb, sig)], gb)
            rows.append(dict(camera=cam, method=name, psnr=p, ssim=s, n_blocks=len(nb), sec_per_block=(time.time() - t0) / len(nb)))
            print(f"{cam:3s} {name:32s} PSNR {p:6.2f}  SSIM {s:.4f}  ({len(nb)} blocks, {rows[-1]['sec_per_block']:.2f}s/block)", flush=True)
    # Save the raw rows (csv) and a per-camera markdown table with the means.
    with open(os.path.join(C.RESULTS_DIR, "baselines.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    names = list(dict.fromkeys(r["method"] for r in rows)); cams = list(dict.fromkeys(r["camera"] for r in rows))
    lines = ["| Method | " + " | ".join(cams) + " | Mean PSNR | Mean SSIM |", "|---|" + "---|" * (len(cams) + 2)]
    for n in names:
        rs = {r["camera"]: r for r in rows if r["method"] == n}
        lines.append(f"| {n} | " + " | ".join(f"{rs[c]['psnr']:.2f}" if c in rs else "n/a" for c in cams) + f" | {np.mean([r['psnr'] for r in rs.values()]):.2f} | {np.mean([r['ssim'] for r in rs.values()]):.4f} |")
    open(os.path.join(C.RESULTS_DIR, "baselines.md"), "w").write("PSNR (dB) per camera, same blocks for every method\n\n" + "\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    main()
