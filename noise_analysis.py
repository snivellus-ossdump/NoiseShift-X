"""What noise does SIDD-Small actually contain?  (answers: why Poisson-Gaussian, how do cameras differ)
CPU only, no torch. Noise = noisy - ground truth (sRGB, gamma-encoded, after the camera ISP).

  python noise_analysis.py
Outputs in results/: noise_vs_intensity.png, noise_vs_iso.png, noise_stats.csv
  - variance vs gray intensity per camera + least-squares fit  var = a*x + b   (x in [0,1])
  - overall noise std per image vs ISO
  - lag-1 spatial correlation of the noise (pure sensor noise would be ~0; ISP processing makes it > 0)
"""
import os, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from data import index_sidd, load_png

# NB intensity bins; SAMPLE random pixels per image (enough for stable statistics and much faster than all pixels).
NB = 32; SAMPLE = 200_000
def iso_of(folder):
    return int(os.path.basename(folder.rstrip("/\\")).split("_")[3])

def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(C.SEED); items = index_sidd(C.SIDD_SMALL_ROOT)
    # Running sums per camera and intensity bin: pixel count, sum of noise, sum of squared noise.
    acc = {c: dict(n=np.zeros(NB), s1=np.zeros(NB), s2=np.zeros(NB), corr=[], iso=[], std=[]) for c in C.CAMERAS}
    for k, it in enumerate(items):
        # noise = noisy - ground truth (scaled to [0, 1]); intensity = gray value of the ground truth.
        nz, gt = load_png(it["noisy"]).astype(np.float32), load_png(it["gt"]).astype(np.float32)
        noise = (nz - gt) / 255.0; x = gt.mean(-1) / 255.0                      # intensity in [0,1]
        H, W = x.shape; ys = rng.integers(0, H, SAMPLE); xs = rng.integers(0, W, SAMPLE)
        # Per-channel noise variance (mean of squares over R, G, B) as a function of ground-truth intensity.
        ns = noise[ys, xs]; nv, nv2, xv = ns.mean(-1), (ns ** 2).mean(-1), x[ys, xs]   # per-channel statistics, gray intensity
        b = np.minimum((xv * NB).astype(int), NB - 1); A = acc[it["camera"]]
        A["n"] += np.bincount(b, minlength=NB); A["s1"] += np.bincount(b, nv, NB); A["s2"] += np.bincount(b, nv2, NB)
        # Lag-1 correlation of the noise in a central crop: about 0 for white noise, clearly > 0 here
        # because of the in-camera processing.
        cy, cx = H // 2, W // 2; cr = noise[max(cy-512, 0):cy+512, max(cx-512, 0):cx+512].mean(-1); cr = cr - cr.mean()
        A["corr"].append(float((cr[:, :-1] * cr[:, 1:]).mean() / (cr * cr).mean()))
        A["iso"].append(iso_of(it["folder"])); A["std"].append(float(noise.std()))
        if (k + 1) % 20 == 0: print(f"{k+1}/{len(items)} scenes", flush=True)
    rows, fig, ax = [], *plt.subplots(figsize=(6.5, 4.5))
    # One variance curve per camera (bins with fewer than 500 samples are ignored).
    centers = (np.arange(NB) + 0.5) / NB
    for cam in C.CAMERAS:
        A = acc[cam]; ok = A["n"] >= 500
        if not ok.any(): continue
        mean = A["s1"][ok] / A["n"][ok]; var = A["s2"][ok] / A["n"][ok] - mean ** 2; xc = centers[ok]
        # Straight-line fit var = a*x + b for the Poisson-Gaussian discussion. The curves are NOT straight,
        # so this fit is only indicative and should not be presented as evidence.
        a_, b_ = np.polyfit(xc, var, 1, w=np.sqrt(A["n"][ok]))
        ax.plot(xc, var, marker="o", ms=3, label=f"{cam} (a={a_:.4f}, b={b_:.5f})")
        rows.append(dict(camera=cam, n_scenes=len(A["iso"]), fit_a=a_, fit_b=b_, mean_noise_std=np.mean(A["std"]),
                         lag1_corr=np.mean(A["corr"]), iso_min=min(A["iso"]), iso_max=max(A["iso"])))
    ax.set_xlabel("ground-truth intensity (gray, 0-1)"); ax.set_ylabel("noise variance"); ax.set_title("SIDD-Small: noise variance vs intensity")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(C.RESULTS_DIR, "noise_vs_intensity.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # Second figure: noise level of every scene against its ISO.
    for cam in C.CAMERAS: ax.scatter(acc[cam]["iso"], acc[cam]["std"], s=12, label=cam)
    ax.set_xscale("log"); ax.set_xlabel("ISO"); ax.set_ylabel("noise std (0-1 scale)"); ax.set_title("Noise level vs ISO per camera"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(C.RESULTS_DIR, "noise_vs_iso.png"), dpi=150); plt.close(fig)
    with open(os.path.join(C.RESULTS_DIR, "noise_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows: print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

if __name__ == "__main__":
    main()
