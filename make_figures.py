"""Slide figures built from the multi-seed results (replaces the old seed-0-only drop_bar.png / unseen_bar.png).
  python make_figures.py      ->  results/fig_unseen_seeds.png, fig_drop.png, fig_methods.png"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from aggregate_all import seeds, ms

def main():
    # Camera labels and bar positions.
    cams, x, w = C.CAMERAS, np.arange(len(C.CAMERAS)), 0.27
    # 1. unseen-camera PSNR, mean +- std over seeds
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); allv = []
    for off, (lab, pat) in zip((-w, 0, w), (("Baseline", "base_hold_{k}"), ("Poisson-Gaussian aug", "aug_hold_{k}"), ("Gaussian-only aug", "abl_gauss_hold_{k}"))):
        m = [ms(seeds(pat, k)) for k in cams]; allv += [v[0] for v in m]
        ax.bar(x + off, [v[0] for v in m], w, yerr=[0 if np.isnan(v[1]) else v[1] for v in m], capsize=2, label=lab)
    ax.set_ylim(min(allv) - 1.5, max(allv) + 0.8); ax.set_xticks(x); ax.set_xticklabels(cams); ax.set_ylabel("PSNR on held-out camera (dB)")
    ax.set_title("Unseen-camera PSNR, mean ± std over 3 seeds"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(C.RESULTS_DIR, "fig_unseen_seeds.png"), dpi=150); plt.close(fig)
    # 2. drop vs full-data seen model (seed-averaged)
    fig, ax = plt.subplots(figsize=(7, 4)); d = {}
    for m_ in ("base", "aug"):
        d[m_] = [ms(seeds(m_ + "_all", k))[0] - ms(seeds(m_ + "_hold_{k}", k))[0] for k in cams]
    ax.bar(x - 0.2, d["base"], 0.4, label="Baseline"); ax.bar(x + 0.2, d["aug"], 0.4, label="Noise augmentation"); ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cams); ax.set_ylabel("seen minus unseen PSNR (dB), lower is better"); ax.set_title("Cross-camera drop (seed-averaged)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(C.RESULTS_DIR, "fig_drop.png"), dpi=150); plt.close(fig)
    # 3. method comparison from baselines.csv (mean PSNR over cameras)
    # Figure 3 needs results/baselines.csv, written by baselines.py; it is skipped if that file does not exist.
    p = os.path.join(C.RESULTS_DIR, "baselines.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p))); names = list(dict.fromkeys(r["method"] for r in rows))
        vals = [np.mean([float(r["psnr"]) for r in rows if r["method"] == n]) for n in names]
        fig, ax = plt.subplots(figsize=(7.5, 4.2)); bars = ax.barh(names[::-1], vals[::-1])
        for b_, n in zip(bars, names[::-1]):
            if "oracle" in n: b_.set_hatch("//")
        ax.set_xlim(min(vals) - 1, max(vals) + 1); ax.set_xlabel("mean PSNR over 5 cameras (dB)"); ax.set_title("Same blocks for every method (hatched = uses true noise level)")
        fig.tight_layout(); fig.savefig(os.path.join(C.RESULTS_DIR, "fig_methods.png"), dpi=150); plt.close(fig)
    print("saved fig_unseen_seeds.png, fig_drop.png" + (", fig_methods.png" if os.path.exists(p) else ""))

if __name__ == "__main__":
    main()
