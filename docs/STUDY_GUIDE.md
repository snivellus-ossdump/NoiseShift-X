# NoiseShift-X: end-to-end study guide (for team members and for pasting into an AI tutor)

## 0. How to use this file
**Paste this whole file (and `CONTEXT.md`) into an AI chat with this prompt:**
> You are my tutor for the project described below. Teach me module by module in the order given. After each module, ask me 3 questions, wait for my answer, and correct me. Do not invent numbers: use only the numbers in CONTEXT.md. If something is a general fact you are unsure of, say so. At the end, run a 20-question viva on me, one question at a time.

Goal: every team member can explain **theory, literature and code** (the teacher evaluates individuals). Time needed: about 6 to 8 hours spread over 2 days (modules 1-7 about 4 hours, code 1.5 hours, rehearsal 1.5 hours).

## 1. The project in one minute (memorise this)
Real photos from different phone cameras have different noise. A denoising network trained on some cameras may do worse on a camera it never saw. We trained a small CNN denoiser (DnCNN-style) on the SIDD-Small dataset (5 smartphones, 160 scenes), measured how much it loses on a held-out camera (about 0.4 dB PSNR on average), and tested one idea: adding synthetic noise during training. The idea gave a small average gain (+0.19 dB), clear only on one camera (S6, +0.58 dB), did not reduce the average cross-camera drop, was matched by simple Gaussian noise, and disappeared when training was 3 times longer. A learned model beat classical methods (Gaussian, median, NLM, BM3D) by about 4 dB. All results use fixed test scenes, 3 training seeds, and we also report official SIDD benchmark scores.

## 2. Module A: images, noise and why real noise is hard
**Learn:**
- An image is an array of pixel intensities; sRGB images are 8-bit (0 to 255), 3 channels. We rescale to [0, 1] for computing.
- **Noise sources in a camera sensor:** *shot noise* (light arrives as discrete photons; variance grows with brightness; Poisson), *read noise* (electronics; roughly constant variance; Gaussian). Raw model: `y = x + N(0, a*x + b)` (Poisson-Gaussian, Foi et al. 2008): `a` = shot-noise coefficient, `b` = read-noise variance.
- **The camera pipeline (ISP)** turns raw sensor data into the sRGB photo: demosaicing, white balance, tone mapping / gamma, denoising, compression. These steps change the noise: it becomes spatially correlated (neighbouring pixels share noise) and its relationship to brightness is no longer a simple formula. So Gaussian/Poisson-Gaussian noise is only an approximation for sRGB photos.
- **ISO** = sensor gain: higher ISO means more noise.
- **Our measurements (noise_analysis.py):** noise std rises with ISO (about 0.015-0.025 at ISO 100 to 0.12-0.155 at ISO 3200); cameras differ at the same ISO; neighbour correlation 0.33-0.38; variance versus intensity is not monotonic and differs per camera.
**Be able to explain:** why a model trained on synthetic white Gaussian noise fails on real photos; why different cameras have different noise.
**Self-test:** What are shot and read noise? Why is real sRGB noise correlated? Why did we not present the fitted (a, b) values as evidence?

## 3. Module B: the datasets
- **SIDD** (Abdelhamed, Lin, Brown, CVPR 2018): real noisy/clean image pairs from 5 smartphones (GP, IP, S6, N6, G4), many scenes, ISOs and lighting. The clean image is obtained by capturing many frames of the same static scene and combining them. **SIDD-Small**: 160 scene instances (IP 54, S6 35, GP 33, N6 22, G4 16).
- **Official SIDD data we used:** validation blocks (40 images x 32 blocks of 256x256, GT available, no camera labels) and benchmark blocks (hidden ground truth; scored on Kaggle because the official server was down).
- **DND** (Plotz and Roth, CVPR 2017): 50 scenes, hidden ground truth, scored through a server; planned but not done.
- **Why a fixed test set:** we reserve about 25% of scenes per camera (IP 14, S6 9, GP 9, N6 6, G4 4) and no model trains on them.
**Self-test:** How many scenes per camera? Why are the benchmark ground truths hidden? Why are the validation blocks not used for the per-camera test?

## 4. Module C: metrics
- **MSE** = mean of squared pixel differences. **PSNR** = `10 * log10(MAX^2 / MSE)` in dB; with images in [0,1], MAX = 1. Higher is better; +1 dB means about 21% lower MSE. We compute PSNR per 256x256 block, then average over blocks.
- **SSIM** compares local luminance, contrast and structure: `((2 mu_x mu_y + c1)(2 sigma_xy + c2)) / ((mu_x^2 + mu_y^2 + c1)(sigma_x^2 + sigma_y^2 + c2))`, between -1 and 1, 1 = identical (scikit-image, 7x7 window).
- **Cross-camera drop** = PSNR of a model that saw the camera minus PSNR of a model that did not, on the **same test scenes**. Lower is better.
- **Gain** = augmented minus baseline PSNR on the unseen camera (paired by seed).
**Self-test:** Why per-block averaging? Why compare seen and unseen models on the same scenes? What does +0.2 dB mean in practice (small)?

## 5. Module D: classical denoising methods (teacher's questions 1 and 3)
| Method | Idea | Assumption / weakness |
|---|---|---|
| Gaussian blur | Replace each pixel by a weighted local average | Removes noise but blurs edges and detail |
| Median filter | Replace each pixel by the median of its neighbours | Good for salt-and-pepper noise; weak on dense noise |
| Non-Local Means (Buades et al., CVPR 2005) | Average pixels whose surrounding *patches* look similar, anywhere in the image | Needs a filter strength `h` tied to the noise level |
| BM3D (Dabov et al., TIP 2007) | Group similar patches into 3D stacks, filter them together in a transform domain, then aggregate (two stages: hard threshold, then Wiener) | Needs the noise level; assumes white Gaussian noise |
**Our result:** with a blind noise estimate, BM3D (29.79 dB mean) barely beat the noisy input (28.64); with the true noise level (oracle) it reached 33.52 dB; NLM went from 32.18 to 34.30; Gaussian blur gave 33.56; the unseen-camera DnCNN gave 37.90. Why: the real noise is correlated and the blind estimate is too low (likely explanation, not separately proven). Settings were untuned.
**Self-test:** Why does BM3D need sigma? What is an "oracle" row and why is it not a fair method?

## 6. Module E: CNN denoising (theory behind our model)
- **Convolution layer:** learned 3x3 filters slide over the image; stacking layers builds a larger *receptive field*. Our 10 layers of 3x3 give a 21x21 receptive field.
- **ReLU** (non-linearity), **batch normalisation** (normalises layer activations to stabilise training).
- **Residual learning (DnCNN, Zhang et al., TIP 2017):** the network predicts the noise `n_hat`; output = `x_noisy - n_hat`. Easier than predicting the clean image because the noise is a simpler signal. Original DnCNN uses 17-20 layers; ours is a smaller 10-layer variant (about 0.3 M parameters).
- **Loss:** L1 = mean absolute error (robust, tends to give sharper results than L2).
- **Training:** *patch* = random 128x128 crop (cheap, many samples); *batch* = 32 patches; *epoch* = one pass over the training patches; **Adam** optimiser (adaptive learning rates); **cosine schedule** `lr(e) = lr0 * 0.5 * (1 + cos(pi e / E))`, lr0 = 1e-3; random 90-degree rotations as basic augmentation.
- **Overfitting / generalisation:** performing well on training data but worse on new data; our test scenes are never trained on.
- **Seed:** controls weight initialisation, data order and augmentation noise; different seeds give different results, which is why we report 3 seeds.
**Self-test:** What is the receptive field? Why predict noise instead of the image? What does the cosine schedule do?

## 7. Module F: domain shift and our idea
- **Domain shift / cross-camera generalization:** the test camera's noise distribution differs from the training cameras' distributions.
- **Leave-one-camera-out:** train on 4 cameras, test on the 5th, rotate (5 folds).
- **Our idea:** noise augmentation. With probability 0.5 per patch add synthetic noise `N(0, a*x + b)` on top of the real noisy patch (Poisson-Gaussian), with `a`, `b` drawn log-uniformly. Hope: more noise levels seen in training means better coverage of a new camera.
- **Control:** Gaussian-only (a = 0, constant variance). It matched Poisson-Gaussian, so the signal-dependent part is not what matters here.
- **Related literature:** CBDNet (Guo et al., CVPR 2019) trains a blind denoiser with a noise-estimation subnetwork on synthetic Poisson-Gaussian + ISP noise plus real noise. NAFNet (Chen et al., ECCV 2022) and Restormer (Zamir et al., CVPR 2022) are strong modern backbones (not trained by us). Verify citation details before presenting.

## 8. Module G: experimental design (explain this well)
1. **Fixed scene-level test set**, stratified by camera, seed 0, never trained on by any model.
2. **Models:** `*_all` (all cameras, the seen reference), `*_hold_K` (camera K unseen), `*_match_K` (size-matched control), each as baseline and augmented.
3. **Seeds:** 3 per setting (0, 1, 2). Report mean ± standard deviation and paired gains.
4. **Fixed evaluation grid** of non-overlapping 256x256 blocks (up to 200 per image), never random crops.
5. **Official protocol:** validation blocks (3 seeds) and benchmark blocks via Kaggle (1 model each).
6. **Controls and checks:** Gaussian-only noise, size-matched models, ablations of probability and strength, 30-epoch training.
**Why each exists:** fixed test scenes avoid testing on training images; seeds measure luck; size-matched models separate "less data" from "unseen camera"; Gaussian control tests whether signal-dependence matters; 30 epochs tests whether results survive better training.

## 9. Module H: the results (know these numbers; all from CONTEXT.md)
- Unseen-camera PSNR, baseline vs augmented (3 seeds): GP 38.76 vs 38.93; IP 38.33 vs 38.53; S6 36.09 vs 36.67; N6 36.26 vs 36.36; G4 40.49 vs 40.40. Mean gain +0.19 dB; clear only on S6 (+0.58 ± 0.22).
- Augmented models vary less between seeds (mean std 0.13 vs 0.34 dB).
- Cross-camera drop vs full-data seen model: baseline mean 0.43 dB, augmented 0.52 dB: **no reduction**.
- Gaussian-only vs Poisson-Gaussian: mean difference +0.05 dB: **no detectable difference**.
- Classical comparison (mean PSNR): noisy 28.64, Gaussian blur 33.56, median 32.06, NLM 32.18 (oracle 34.30), BM3D 29.79 (oracle 33.52), DnCNN 37.90, DnCNN+aug 38.32.
- Official validation blocks (3 seeds): baseline 35.51 ± 0.23 dB / SSIM 0.836; augmented 35.86 ± 0.26 dB / SSIM 0.850. Kaggle benchmark: PSNR 36.2546 vs 36.2904; SSIM 0.842252 vs 0.849825.
- 30 epochs on S6 (3 seeds): baseline 37.97 ± 0.14, augmented 37.91 ± 0.36: gain -0.06 ± 0.24 (gone). Training longer helped about 1.2-1.9 dB.
**Interpretation to be able to give:** the model loses little on unseen cameras; augmentation mostly helps an undertrained model early on; claims must follow the data; the honest, mixed result is the finding.

## 10. Module I: reading statistics honestly
- **Mean and standard deviation over seeds:** std measures how much results change just from random initialisation and data order. A difference smaller than about one std (for example +0.1 dB with std 0.3) cannot be called real.
- **Paired comparison:** compare augmented and baseline with the same seed, then average the per-seed differences.
- **Why not pick the best seed:** that is cherry-picking and overstates results.
- **Single-run evidence** (size-matched controls, ablations, Kaggle scores) cannot carry strong claims.
- **Small test sets:** G4 has 4 test scenes and N6 6, so those folds are noisy.

## 11. Module J: the code, file by file (every member should read these)
| File | Read for |
|---|---|
| `config.py` | all shared settings (patch size 128, 100 patches per image, test fraction 0.25, 10 epochs, seed 0) |
| `data.py` | `parse_camera`, `index_sidd`, `split_scenes` (fixed test split), `build_patch_cache` (random aligned crops on disk), `load_cache`, `fixed_eval_blocks` |
| `model.py` | `SmallDenoiser`: 10 conv layers, predicts noise, returns `x - net(x)` |
| `augment.py` | `add_poisson_gaussian` (a, b log-uniform; `gauss_only` control) and `maybe_augment` |
| `metrics.py` | `psnr`, `ssim`, `eval_blocks`, `cross_camera_drop`, `load_official_blocks` |
| `train.py` | `PatchDS` dataset, training loop, `--hold`, `--aug`, `--aug_gauss`, `--match`, `--seed`, checkpoint and resume |
| `evaluate.py` | `denoise_blocks`, per-camera test evaluation and official validation blocks; writes `results/<name>.json` |
| `run_fold.py`, `run_queue.py` | how all experiments were launched; resumable job queue |
| `aggregate_all.py`, `make_figures.py`, `visualize.py` | tables, charts and the qualitative figure |
| `noise_analysis.py` | noise vs intensity, noise vs ISO, neighbour correlation |
| `baselines.py` | Gaussian, median, NLM, BM3D, blind and oracle noise level |
| `benchmark_submit.py` | creates the Kaggle CSV (base64 of each block) |
| `test_synthetic.py` | six sanity tests on fake data |
**Walk-through exercise:** trace one training step: cache patch -> random rotation -> optional noise -> tensor in [0,1] -> network -> L1 against clean patch -> Adam update. Then trace one evaluation: test scene -> block grid -> network -> PSNR/SSIM per block -> mean.
**Hands-on (15 minutes each):** (1) `python test_synthetic.py` and explain each line printed; (2) open `results/summary_full.md` and find the S6 row; (3) run `python evaluate.py --ckpt ckpt/base_hold_S6.pt --cams S6` (if checkpoints are available) and match the printed PSNR to the table.

## 12. Module K: honest limitations (say these before being asked)
Five cameras; tiny test sets for G4 and N6; 3 seeds; a small model trained for only 10 epochs (30 epochs improved S6 by 1.2-1.9 dB); hold-out models see fewer training patches; sRGB noise is correlated so Poisson-Gaussian is an approximation; our per-camera numbers use our own block grid over reserved scenes (the official blocks have no camera labels as far as known); DND not done; LPIPS not done; results are not comparable to published numbers.

## 13. Presentation rehearsal
- **Two-minute version:** problem (cross-camera shift), what we built (fixed test scenes, hold-out, 3 seeds), what we found (small drop; augmentation helps early training on S6, no average drop reduction, Gaussian equals Poisson-Gaussian, gain vanishes with longer training; DnCNN beats classical methods), what we would do next (stronger backbone, longer training, DND).
- **Who explains what (suggestion):** theory and literature / methodology and code / results and limitations; everyone must still be able to answer any part.

## 14. Viva question bank (practise answering aloud in 2-3 sentences)
1. What is the problem and why does it matter? 2. What noise does SIDD contain? 3. Why is real sRGB noise hard compared with Gaussian noise? 4. What is shot noise versus read noise? 5. What is ISO? 6. Explain PSNR and SSIM. 7. Why average per block? 8. What is the cross-camera drop and how exactly did you compute it? 9. Why fixed test scenes? 10. What is leave-one-camera-out? 11. What is a size-matched control and why did you add it? 12. Explain residual learning. 13. Why L1 loss? 14. What are patches, batches, epochs? 15. What does Adam do? What does the cosine schedule do? 16. Why batch normalisation? 17. What is the receptive field of your network? 18. How does BM3D work? Why did it do badly? 19. How does NLM work? 20. What is an oracle noise level? 21. What is Poisson-Gaussian noise and why did you test it? 22. What did the Gaussian-only control show? 23. Did augmentation reduce the drop? 24. Why did you run 3 seeds and what did they change? 25. Why did the S6 gain differ from the others? (hypothesis: S6 has the noisiest high-ISO images; not tested) 26. Why did the gain vanish at 30 epochs? (hypothesis: augmentation helps early learning of an undertrained model) 27. What are the Kaggle scores and why can't you compare them to papers? 28. Why DnCNN and not NAFNet? 29. What would you change with more time? 30. Walk me through `train.py`. 31. How does `split_scenes` guarantee no leakage? 32. What does `build_patch_cache` do and why on disk? 33. What bugs or mistakes did you find and fix? (see CONTEXT.md section 7) 34. What is the limitation you worry about most?

## 15. Quick formulas card
- PSNR = 10 log10(1/MSE) for images in [0,1]. Roughly +1 dB = 21% less MSE.
- Raw noise model: y = x + N(0, a x + b).
- Residual learning: x_hat = y - f(y).
- L1 loss = mean |x_hat - x|.
- Cosine LR: lr(e) = lr0 * 0.5 * (1 + cos(pi e / E)).
- Cross-camera drop = PSNR(seen model) - PSNR(unseen model), same test scenes.
- Receptive field of L stacked 3x3 convs = 2L + 1 (L = 10 gives 21).
