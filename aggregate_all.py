"""Full results across seeds, controls and size-matched models.   python aggregate_all.py
Writes results/summary_full.md (+ gain_bars.png). Seed 0 = plain model name; seeds 1,2 = name_s1, name_s2.
 A  unseen-camera PSNR/SSIM: mean +- std over seeds, and paired gain (aug - base)
 B  Gaussian-only control vs Poisson-Gaussian vs baseline (means over seeds)
 C  cross-camera drop: seen model on the SAME test scenes minus unseen model; also vs a size-matched seen model"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

# Read one metric of one model on one camera from results/<name>.json (None if missing).
def get(name, cam, key="psnr"):
    p = os.path.join(C.RESULTS_DIR, name + ".json")
    if not os.path.exists(p): return None
    return json.load(open(p)).get("cams", {}).get(cam, {}).get(key)

# Collect a metric over seeds: seed 0 = plain model name, seeds 1 and 2 = name_s1, name_s2.
def seeds(prefix, cam, key="psnr", suffix=""):
    out = {}
    for s, tag in ((0, ""), (1, "_s1"), (2, "_s2")):
        v = get(prefix.format(k=cam) + tag + suffix, cam, key)
        if v is not None: out[s] = v
    return out

# Mean, sample standard deviation (ddof=1) and count of a {seed: value} dictionary.
def ms(d):
    v = list(d.values()); return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v)) if v else (float("nan"),) * 2 + (0,)
def fmt(t, p=2): return "n/a" if np.isnan(t[0]) else (f"{t[0]:.{p}f}" + (f" ± {t[1]:.{p}f}" if not np.isnan(t[1]) else "") + f" (n={t[2]})")
# Per-seed difference b - a (only for seeds present in both).
def paired(a, b): return {s: b[s] - a[s] for s in a if s in b}

def main():
    # Markdown tables: A = unseen-camera performance, B = Gaussian-only control, D = cross-camera drop.
    A = ["| Camera | Base unseen PSNR | Aug unseen PSNR | Gain (paired, aug - base) | Base SSIM | Aug SSIM |", "|---|---|---|---|---|---|"]
    B = ["| Camera | Baseline (mean) | Poisson-Gaussian (mean) | Gaussian-only (mean ± std) | Gaussian-only minus P-G |", "|---|---|---|---|---|"]
    D = ["| Camera | Base drop vs full-data seen | Aug drop vs full-data seen | Base drop vs size-matched seen | Aug drop vs size-matched seen |", "|---|---|---|---|---|"]
    gains, gstd, gauss_gain, rowsA, rowsD, gdiff = [], [], [], [], [], []
    bmeans, ameans = [], []
    for k in C.CAMERAS:
        # One table row per held-out camera. bu / au = baseline / augmented unseen PSNR per seed.
        bu, au = seeds("base_hold_{k}", k), seeds("aug_hold_{k}", k); g = paired(bu, au)
        A.append(f"| {k} | {fmt(ms(bu))} | {fmt(ms(au))} | {fmt(ms(g))} | {fmt(ms(seeds('base_hold_{k}', k, 'ssim')), 3)} | {fmt(ms(seeds('aug_hold_{k}', k, 'ssim')), 3)} |")
        gm = ms(g); gains.append(gm[0]); gstd.append(0 if np.isnan(gm[1]) else gm[1]); bmeans.append(ms(bu)[0]); ameans.append(ms(au)[0])
        # Table B: Gaussian-only control compared with the baseline and Poisson-Gaussian means.
        gs = seeds("abl_gauss_hold_{k}", k); bmn, amn, gmn = ms(bu), ms(au), ms(gs)
        B.append(f"| {k} | {bmn[0]:.2f} | {amn[0]:.2f} | {fmt(gmn)} | {gmn[0] - amn[0]:+.2f} |")
        gauss_gain.append(gmn[0] - bmn[0]); gdiff.append(gmn[0] - amn[0])
        # Table C: drop = seen-model PSNR - unseen-model PSNR (seed-averaged), then the same against the size-matched seen model.
        row = [k]; nums = []
        for m in ("base", "aug"):
            seen = ms(seeds(m + "_all", k)); unseen = ms(seeds(m + "_hold_{k}", k)); v = seen[0] - unseen[0]
            nums.append(v); row.append("n/a" if np.isnan(v) else f"{v:.2f}")
        for m in ("base", "aug"):
            mt, h0 = get(f"{m}_match_{k}", k), seeds(m + "_hold_{k}", k).get(0); v = np.nan if mt is None or h0 is None else mt - h0
            nums.append(v); row.append("n/a" if np.isnan(v) else f"{v:.2f}")
        D.append("| " + " | ".join(row) + " |"); rowsD.append(nums)
    A.append(f"| **MEAN** | {np.nanmean(bmeans):.2f} | {np.nanmean(ameans):.2f} | {np.nanmean(gains):.2f} | | |")
    B.append(f"| **MEAN** | {np.nanmean(bmeans):.2f} | {np.nanmean(ameans):.2f} | | {np.nanmean(gdiff):+.2f} |")
    D.append("| **MEAN** | " + " | ".join(f"{np.nanmean([r[i] for r in rowsD]):.2f}" for i in range(4)) + " |")
    # Assemble and save the markdown report.
    txt = ("# Full results (PSNR in dB on each camera's fixed test scenes)\n\n## A. Unseen-camera performance, mean ± std over seeds\n\n" + "\n".join(A) +
           "\n\n## B. Does the signal-dependent part matter? (baseline and Poisson-Gaussian: mean of 3 seeds; Gaussian-only: mean of the seeds available)\n\n" + "\n".join(B) +
           "\n\n## C. Cross-camera drop (lower is better)\n\n" + "\n".join(D) + "\n")
    os.makedirs(C.RESULTS_DIR, exist_ok=True); open(os.path.join(C.RESULTS_DIR, "summary_full.md"), "w").write(txt); print(txt)
    # Bar chart: gain over the baseline per camera; error bars = std over seeds; dots = Gaussian-only control.
    x = np.arange(len(C.CAMERAS)); fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x, gains, 0.5, yerr=gstd, capsize=3, label="Poisson-Gaussian gain (mean ± std over seeds)")
    ax.scatter(x, gauss_gain, color="k", zorder=3, label="Gaussian-only gain (mean over its seeds)"); ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(C.CAMERAS); ax.set_ylabel("unseen-camera PSNR gain over baseline (dB)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS_DIR, "gain_bars.png"), dpi=150)

if __name__ == "__main__":
    main()
