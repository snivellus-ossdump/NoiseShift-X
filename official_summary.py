"""Official SIDD validation-block PSNR/SSIM of the seen-camera models over seeds.   python official_summary.py"""
import json, os
import numpy as np
import config as C

# Collect the official-validation scores of base_all, base_all_s1, base_all_s2 (and the same for aug).
for m in ("base", "aug"):
    rows = []
    for tag in ("", "_s1", "_s2"):
        p = os.path.join(C.RESULTS_DIR, f"{m}_all{tag}.json")
        if os.path.exists(p) and "official_val" in json.load(open(p)): rows.append(json.load(open(p))["official_val"])
    if rows:
        # Mean and sample standard deviation over seeds.
        ps, ss = [r["psnr"] for r in rows], [r["ssim"] for r in rows]
        sd = lambda v: f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else ""
        print(f"{m}_all  n={len(rows)} seeds  PSNR {np.mean(ps):.3f}{sd(ps)}   SSIM {np.mean(ss):.4f}{sd(ss)}   per-seed PSNR {[round(p, 3) for p in ps]}")
