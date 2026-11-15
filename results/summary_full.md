# Full results (PSNR in dB on each camera's fixed test scenes)

## A. Unseen-camera performance, mean ± std over seeds

| Camera | Base unseen PSNR | Aug unseen PSNR | Gain (paired, aug - base) | Base SSIM | Aug SSIM |
|---|---|---|---|---|---|
| GP | 38.76 ± 0.17 (n=3) | 38.93 ± 0.05 (n=3) | 0.17 ± 0.13 (n=3) | 0.935 ± 0.002 (n=3) | 0.940 ± 0.001 (n=3) |
| IP | 38.33 ± 0.35 (n=3) | 38.53 ± 0.14 (n=3) | 0.20 ± 0.29 (n=3) | 0.918 ± 0.009 (n=3) | 0.926 ± 0.004 (n=3) |
| S6 | 36.09 ± 0.34 (n=3) | 36.67 ± 0.17 (n=3) | 0.58 ± 0.22 (n=3) | 0.825 ± 0.010 (n=3) | 0.852 ± 0.004 (n=3) |
| N6 | 36.26 ± 0.40 (n=3) | 36.36 ± 0.07 (n=3) | 0.10 ± 0.46 (n=3) | 0.858 ± 0.014 (n=3) | 0.865 ± 0.003 (n=3) |
| G4 | 40.49 ± 0.44 (n=3) | 40.40 ± 0.24 (n=3) | -0.10 ± 0.42 (n=3) | 0.959 ± 0.004 (n=3) | 0.959 ± 0.004 (n=3) |
| **MEAN** | 37.99 | 38.18 | 0.19 | | |

## B. Does the signal-dependent part matter? (baseline and Poisson-Gaussian: mean of 3 seeds; Gaussian-only: mean of the seeds available)

| Camera | Baseline (mean) | Poisson-Gaussian (mean) | Gaussian-only (mean ± std) | Gaussian-only minus P-G |
|---|---|---|---|---|
| GP | 38.76 | 38.93 | 38.61 ± 0.08 (n=3) | -0.32 |
| IP | 38.33 | 38.53 | 38.82 ± 0.29 (n=3) | +0.29 |
| S6 | 36.09 | 36.67 | 36.67 ± 0.27 (n=3) | -0.00 |
| N6 | 36.26 | 36.36 | 36.47 ± 0.21 (n=3) | +0.11 |
| G4 | 40.49 | 40.40 | 40.57 ± 0.51 (n=3) | +0.18 |
| **MEAN** | 37.99 | 38.18 | | +0.05 |

## C. Cross-camera drop (lower is better)

| Camera | Base drop vs full-data seen | Aug drop vs full-data seen | Base drop vs size-matched seen | Aug drop vs size-matched seen |
|---|---|---|---|---|
| GP | 0.20 | 0.28 | 0.15 | -0.40 |
| IP | 0.90 | 1.06 | 0.53 | 0.35 |
| S6 | 0.92 | 0.65 | 0.34 | 0.08 |
| N6 | 0.13 | 0.24 | -0.11 | -0.10 |
| G4 | 0.03 | 0.38 | 0.92 | -0.22 |
| **MEAN** | 0.43 | 0.52 | 0.37 | -0.06 |
