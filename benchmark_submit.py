"""Denoise the official SIDD *benchmark* blocks (hidden ground truth) and write the Kaggle submission CSV directly.

  python benchmark_submit.py --ckpt ckpt\\aug_all.pt
Input : content\\BenchmarkNoisyBlocksSrgb.mat   (variable BenchmarkNoisyBlocksSrgb, uint8, shape 40 x 32 x 256 x 256 x 3)
Output: submission\\<ckpt-name>_SubmitSrgb.csv  with columns ID (0..1279) and BLOCK (base64 of each block's raw uint8 bytes,
        blocks ordered image-major then block), the same layout as the official sidd_kaggle_submit example code.
Submit the CSV at kaggle.com/competitions/sidd-benchmark-srgb-psnr (and the SSIM competition for the SSIM score)."""
import argparse, base64, csv, os
import numpy as np
import config as C

# Read the official benchmark noisy blocks (hidden ground truth) and check shape and dtype.
def load_blocks(path):
    from scipy.io import loadmat
    d = loadmat(path); assert "BenchmarkNoisyBlocksSrgb" in d, f"variable BenchmarkNoisyBlocksSrgb not found; keys: {[k for k in d if not k.startswith('__')]}"
    x = d["BenchmarkNoisyBlocksSrgb"]; assert x.shape == (40, 32, 256, 256, 3) and x.dtype == np.uint8, f"unexpected {x.shape} {x.dtype}"
    return x

# Kaggle format: one row per block, ID = 0..1279, BLOCK = base64 of the raw uint8 bytes (image-major order).
def save_csv(den, path):
    assert den.dtype == np.uint8 and den.ndim == 5
    os.makedirs(os.path.dirname(path), exist_ok=True); n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["ID", "BLOCK"])
        for i in range(den.shape[0]):
            for j in range(den.shape[1]):
                w.writerow([n, base64.b64encode(np.ascontiguousarray(den[i, j]).tobytes()).decode("utf-8")]); n += 1
    return n

# Re-read the written CSV and verify the IDs and a few decoded blocks (needs pandas).
def check_csv(path, den, rows=(0, 1, 37, -1)):
    import pandas as pd
    df = pd.read_csv(path); assert list(df.columns) == ["ID", "BLOCK"] and len(df) == den.shape[0] * den.shape[1] and (df["ID"].values == np.arange(len(df))).all()
    flat = den.reshape(-1, *den.shape[2:])
    for r in rows:
        dec = np.frombuffer(base64.b64decode(df["BLOCK"].iloc[r]), dtype=np.uint8).reshape(256, 256, 3); assert (dec == flat[r]).all(), f"row {r} mismatch"
    return len(df)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--noisy", default=os.path.join(C.DATA, "BenchmarkNoisyBlocksSrgb.mat"))
    ap.add_argument("--out_dir", default=os.path.join(C.BASE, "submission"))
    a = ap.parse_args()
    import torch
    from model import SmallDenoiser
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SmallDenoiser().to(dev); m.load_state_dict(torch.load(a.ckpt, map_location=dev)["m"]); m.eval()
    # Denoise all 1280 blocks in batches of 8.
    x = load_blocks(a.noisy); flat = x.reshape(-1, 256, 256, 3); out = np.empty_like(flat)
    with torch.no_grad():
        for i in range(0, len(flat), 8):
            t = torch.from_numpy(np.ascontiguousarray(flat[i:i+8])).permute(0, 3, 1, 2).float().div(255).to(dev)
            out[i:i+8] = m(t).clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    # Restore the (40, 32, 256, 256, 3) layout, write the CSV and verify it.
    den = out.reshape(x.shape); name = os.path.splitext(os.path.basename(a.ckpt))[0]; path = os.path.join(a.out_dir, name + "_SubmitSrgb.csv")
    save_csv(den, path); n = check_csv(path, den)
    print(f"saved {path}: {n} blocks, {os.path.getsize(path) / 1e6:.0f} MB, mean abs change vs noisy {np.abs(den.astype(float) - x.astype(float)).mean():.2f}")

if __name__ == "__main__":
    main()
