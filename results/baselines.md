PSNR (dB) per camera, same blocks for every method

| Method | GP | IP | S6 | N6 | G4 | Mean PSNR | Mean SSIM |
|---|---|---|---|---|---|---|---|
| Noisy input | 31.87 | 28.77 | 25.95 | 25.12 | 31.46 | 28.64 | 0.5419 |
| Gaussian blur (s=1) | 36.58 | 33.64 | 31.03 | 30.13 | 36.41 | 33.56 | 0.7697 |
| Median 3x3 | 35.39 | 32.17 | 29.58 | 28.11 | 35.04 | 32.06 | 0.7041 |
| Non-Local Means (blind sigma) | 36.25 | 31.68 | 30.13 | 27.52 | 35.29 | 32.18 | 0.7194 |
| Non-Local Means (oracle sigma) | 36.73 | 34.05 | 33.18 | 30.06 | 37.50 | 34.30 | 0.8294 |
| BM3D (blind sigma) | 32.90 | 30.34 | 26.91 | 26.50 | 32.30 | 29.79 | 0.5975 |
| BM3D (oracle sigma) | 36.12 | 34.09 | 31.22 | 30.00 | 36.20 | 33.52 | 0.7741 |
| DnCNN baseline (unseen cam) | 39.02 | 38.37 | 35.64 | 35.81 | 40.66 | 37.90 | 0.9015 |
| DnCNN + noise aug (unseen cam) | 39.08 | 38.80 | 36.87 | 35.99 | 40.88 | 38.32 | 0.9135 |
