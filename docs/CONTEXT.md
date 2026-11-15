# NoiseShift-X: complete project context (hand-off file)

Purpose of this file: it lets anyone (a teammate or another AI model) understand what the project is, what was actually done, what the measured results are, what the results do and do not support, and what the final presentation must contain. **Every number below was produced by the team's own runs, except where explicitly marked as a general fact or citation to verify.** Do not invent numbers that are not here.

## 0. Logistics and teacher requirements

- Course: DSE-312 Computer Vision, instructor Dr. Samiran Das, IISER Bhopal.
- Team: Suryansh Upadhyay (24342), Prakhar Jain (24253), Siddharth (24325).
- Final presentation: **13 Nov 2026** (slot per the final schedule). **All experiments are complete; the code is commented and packaged (section 13).**
- Teacher's instructions for the final presentation:
  - Slides must cover: **problem statement, objectives, literature review, theory/model/methodology (in brief), dataset details, experiments, results.**
  - Arrive 15-20 minutes early. **Submit the final presentation and properly commented code at least 2 hours before the slot** (drive link to be shared). Bring the **PPT as a PDF** and the **code on a pendrive/laptop**.
  - **Every team member must be able to explain every aspect: theory, literature and code.** Team and individual performance are both evaluated.
- Teacher's feedback on the midterm (all must be visibly addressed): (1) which denoising methods will be investigated; (2) "almost no mention of this" (unclear, probably literature/methods; not confirmed with her); (3) understand standard denoising methods; (4) why Gaussian and Poisson augmentation, and what noise types the dataset contains; (5) quantify and visualize the results.

## 1. Project brief (from the assignment sheet)

- Title: **NoiseShift-X: Cross-Camera Real Image Denoising.** Problem: generalize denoising across real camera sensors and signal-dependent noise distributions.
- Research gap: SIDD and DND expose real-noise behaviour; many methods still rely on dataset/camera-specific training.
- Novelty required: a sensor/noise-invariant representation plus held-out-camera evaluation.
- Metrics: PSNR (higher better), SSIM (higher better), cross-camera drop (lower better), LPIPS optional (not done).
- Protocol: SIDD official benchmark blocks (32 x 256 x 256 per image, 40 images) for PSNR/SSIM; DND official server for the final comparison. Main reviewer risk: do not compute SIDD metrics on arbitrary crops when comparing to official results.
- Team-chosen scope (honest framing): not claiming to solve cross-camera generalization or beat the state of the art. Deliverables: (1) quantify the cross-camera gap of a standard denoiser with leave-one-camera-out on SIDD; (2) test one invariance idea (noise augmentation) with the same backbone and protocol; (3) official benchmark check.

## 2. Dataset

- **SIDD-Small, sRGB** (Smartphone Image Denoising Dataset; Abdelhamed, Lin, Brown, CVPR 2018): 160 scene instances, paired noisy/ground-truth images, 5 smartphones. The camera code is the 3rd underscore-separated field of the folder name (for example `0001_001_S6_00100_00060_3200_L`; fields: instance, scene, camera, ISO, shutter, colour temperature, brightness).
- Scenes per camera: **IP 54, S6 35, GP 33, N6 22, G4 16.** (The midterm slide's "about 32 per camera" was wrong.) Cameras: GP = Google Pixel, IP = iPhone 7, S6 = Samsung Galaxy S6 Edge, N6 = Motorola Nexus 6, G4 = LG G4 (standard SIDD codes; verify wording before presenting).
- ISO ranges seen in SIDD-Small: GP 50-10000, IP 100-2000, S6 100-3200, N6 100-3200, G4 100-800.
- **Official benchmark data** (different from SIDD-Small): validation blocks `ValidationNoisyBlocksSrgb.mat` / `ValidationGtBlocksSrgb.mat` (40 x 32 blocks of 256 x 256, GT available; no per-block camera labels in these files, as far as known) and benchmark blocks `BenchmarkNoisyBlocksSrgb.mat` (hidden ground truth, scored via Kaggle because the official server was down).
- DND (Plotz and Roth, CVPR 2017): planned as final test, **not done** (needs registration, specific format, server turnaround). Listed as future work.

### Noise analysis of the dataset (script `noise_analysis.py`, noise = noisy minus ground truth, sRGB)
| Camera | mean noise std (0-1 scale) | neighbour (lag-1) correlation |
|---|---|---|
| GP | 0.042 | 0.36 |
| IP | 0.038 | 0.33 |
| S6 | 0.069 | 0.34 |
| N6 | 0.065 | 0.36 |
| G4 | 0.037 | 0.38 |

- Noise std rises steeply with ISO (about 0.015-0.025 at ISO 100 up to about 0.12-0.155 at ISO 3200) and **differs between cameras at the same ISO** (at ISO 1600 S6 is about 0.095-0.11, N6 about 0.075-0.09, IP about 0.04-0.047).
- The noise is **spatially correlated** (about 0.33-0.38 between neighbouring pixels), so it is not white.
- Noise variance versus intensity is **not** a simple increasing function in sRGB: curves are non-monotonic and differ per camera (S6 has a large spike in the darkest bin, peaks near intensity 0.4 and 0.75; N6 peaks near 0.2; GP/IP/G4 are low and slowly falling; all go to about zero at white because of clipping). The linear fits `a, b` printed by the script are not meaningful (negative `a` for four cameras) and **must not be shown as evidence**. The Poisson-Gaussian model describes raw sensor noise; in sRGB (after demosaicing, tone mapping, compression) it is only an approximation.
- Figures: `results/noise_vs_iso.png`, `results/noise_vs_intensity.png`.

## 3. Methodology

### Backbone (identical for every model)
`model.py` `SmallDenoiser`: DnCNN-style residual CNN. 10 convolution layers (3x3, 64 channels): Conv+ReLU, then 8 x (Conv+BatchNorm+ReLU), then a final Conv to 3 channels, about 0.3 million parameters. It **predicts the noise** and outputs `x - net(x)` (residual learning; Zhang et al., TIP 2017). Receptive field 21 x 21. Chosen over NAFNet/Restormer because of compute and time; NAFNet is future work.

### Training (identical for every model unless stated)
L1 loss; Adam, learning rate 1e-3 with cosine decay to 0 (set explicitly per epoch); batch 32; **10 epochs**; 128 x 128 training patches, 100 random crops per training scene cached once on disk as uint8 (`trainval_*.npy`); random 90-degree rotations; seed 0 unless stated. About 2 minutes per epoch on the team's laptop GPU for 11,800 patches.

### Train/test design
- A **fixed scene-level test set** is reserved before any training: about 25% of scenes per camera (`data.split_scenes`, seed 0): IP 14, S6 9, GP 9, N6 6, G4 4 test scenes (total 42); train scenes IP 40, S6 26, GP 24, N6 16, G4 12 (total 118). **No model ever trains on test scenes.**
- Evaluation uses a **fixed, non-overlapping 256 x 256 block grid** over each test scene (up to 200 blocks per image; never random crops). PSNR is computed per block on [0,1] images then averaged; SSIM via scikit-image (`structural_similarity`, channel_axis=-1, data_range=255), per block then averaged.
- Models:
  - `base_all`, `aug_all`: trained on train scenes of all 5 cameras (the "seen-camera" reference).
  - `base_hold_K`, `aug_hold_K` (K in GP, IP, S6, N6, G4): trained on train scenes of the other 4 cameras; camera K unseen. 11,800 patches for the all-camera models; 7,800 (IP), 9,200 (S6), 9,400 (GP), 10,200 (N6), 10,600 (G4) for the hold-out models.
  - `*_match_K`: size-matched control: all cameras (including K) but only as many train scenes as the hold-K model had (isolates camera shift from "less training data").
- Metrics per camera K, all on K's test scenes: **unseen PSNR/SSIM** (hold-K model); **gain = aug unseen minus base unseen** (paired by seed); **cross-camera drop = seen model PSNR minus unseen model PSNR on the same test scenes** (lower is better). (An earlier definition comparing different cameras, or testing the seen model on its own training scenes, was rejected as confounded.)

### The proposed idea: noise augmentation (`augment.py`)
With probability 0.5 per patch, add synthetic noise on top of the real noisy input: `y = x + N(0, a*x + b)` with `x` in [0,1], `a` (signal-dependent / shot) log-uniform in [1e-4, 5e-3], `b` (signal-independent / read) log-uniform in [1e-5, 2e-3] (Poisson-Gaussian model; Foi et al., 2008, a raw-domain model). Motivation: one camera gives one noise profile; cameras differ in noise level, so augmentation exposes the model to more noise levels.
- **Control** (`--aug_gauss`): signal-independent Gaussian noise (`a = 0`, `b` log-uniform in [1.5e-4, 2.5e-3]) with similar average added variance (verified flat across brightness).
- Ablation options: `--aug_p` (probability), `--aug_mult` (scale of the noise ranges).

### Other methods compared (`baselines.py`)
Classical, untuned fixed settings, same blocks for every method: Gaussian blur (sigma = 1 px), median 3x3, Non-Local Means (OpenCV `fastNlMeansDenoisingColored`, h from a noise estimate), BM3D (`bm3d` package, sigma from a noise estimate). Noise level is estimated blind per block (scikit-image `estimate_sigma`); **oracle rows** use the true per-block noise std computed from ground truth (an upper bound, not a blind method).

## 4. Experiments performed and results

All PSNR in dB. "n=3" means 3 training seeds (seeds 0, 1, 2); unseen = camera not in training.

### 4.1 Main result: unseen-camera performance (mean ± std over 3 seeds)
| Camera | Baseline PSNR | + Noise aug PSNR | Paired gain | Baseline SSIM | Aug SSIM |
|---|---|---|---|---|---|
| GP | 38.76 ± 0.17 | 38.93 ± 0.05 | +0.17 ± 0.13 | 0.935 | 0.940 |
| IP | 38.33 ± 0.35 | 38.53 ± 0.14 | +0.20 ± 0.29 | 0.918 | 0.926 |
| S6 | 36.09 ± 0.34 | 36.67 ± 0.17 | +0.58 ± 0.22 | 0.825 | 0.852 |
| N6 | 36.26 ± 0.40 | 36.36 ± 0.07 | +0.10 ± 0.46 | 0.858 | 0.865 |
| G4 | 40.49 ± 0.44 | 40.40 ± 0.24 | -0.10 ± 0.42 | 0.959 | 0.959 |
| **Mean** | **37.99** | **38.18** | **+0.19** | | |

Reading: augmentation gives a **clear, consistent gain only on S6** (positive in all 3 seeds); small/uncertain on GP and IP; none detectable on N6 and G4. Mean gain 0.19 dB, mean SSIM gain about 0.009. The augmented models are **more consistent across seeds** (mean std 0.13 dB versus 0.34 dB for baseline; only 3 seeds each, so rough). Earlier single-seed (seed 0) results suggested a larger mean gain (+0.36); seed 0's baseline was a below-average draw on all five cameras, so seed 0 alone overstated the gain.

### 4.2 Does the signal-dependent (Poisson) part matter? Gaussian-only control (3 seeds each)
| Camera | Baseline | Poisson-Gaussian | Gaussian-only | Gaussian-only minus P-G |
|---|---|---|---|---|
| GP | 38.76 | 38.93 | 38.61 ± 0.08 | -0.32 |
| IP | 38.33 | 38.53 | 38.82 ± 0.29 | +0.29 |
| S6 | 36.09 | 36.67 | 36.67 ± 0.27 | 0.00 |
| N6 | 36.26 | 36.36 | 36.47 ± 0.21 | +0.11 |
| G4 | 40.49 | 40.40 | 40.57 ± 0.51 | +0.18 |
| **Mean** | 37.99 | 38.18 | | **+0.05** |

Reading: **no detectable difference** between Poisson-Gaussian and Gaussian-only noise (mean +0.05 dB, per-camera differences inside seed noise). Both give about +0.2 dB over baseline on average. So the signal-dependent part is **not** shown to matter; what helps (when it helps) is exposure to more noise levels. This is a finding that goes against the original Poisson-Gaussian framing and must be reported as such.

### 4.3 Cross-camera drop (seen minus unseen PSNR; lower is better)
| Camera | Base drop vs full-data seen | Aug drop vs full-data seen | Base drop vs size-matched seen | Aug drop vs size-matched seen |
|---|---|---|---|---|
| GP | 0.20 | 0.28 | 0.15 | -0.40 |
| IP | 0.90 | 1.06 | 0.53 | 0.35 |
| S6 | 0.92 | 0.65 | 0.34 | 0.08 |
| N6 | 0.13 | 0.24 | -0.11 | -0.10 |
| G4 | 0.03 | 0.38 | 0.92 | -0.22 |
| **Mean** | **0.43** | **0.52** | **0.37** | **-0.06** |

Reading: the baseline's cross-camera gap is **small (about 0.4 dB on average)**, largest for IP and S6 (about 0.9 dB), near zero for N6 and G4, comparable to seed noise. **Augmentation does not reduce the average drop** against the full-data seen model (0.52 versus 0.43; it shrinks only on S6). The size-matched columns favour augmentation but use a single seed per model (seed noise about +/-0.3-0.4 dB), so they are not strong evidence. Earlier seed-0-only drop numbers (mean 0.79 to 0.42) were misleading and **must not be used** (files `summary.md`, `drop_bar.png`, `unseen_bar.png` come from the old seed-0 aggregation; deprecated).

### 4.4 Comparison with standard methods (`baselines.py --per_scene 3 --oracle_sigma`; 3 evenly spread blocks per test scene, same blocks for all methods; models are the unseen-camera DnCNNs, seed 0)
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

Reading: even a DnCNN that **never saw the test camera** beats every classical method on every camera (about 3.6 dB above the best oracle classical method, about 4.3 dB above the best blind one). BM3D/NLM assume white Gaussian noise of known level; the real noise is correlated and the blind level estimate is too low, which explains the large gap between blind and oracle rows (likely explanation; not separately proven). Classical settings are untuned. Absolute numbers here differ from section 4.1 because the blocks differ (3 per scene versus up to 200); compare methods only within this table.

### 4.5 Official SIDD protocol
- **Official validation blocks (1280 blocks, all cameras mixed), seen-camera models, 3 seeds:** baseline **35.51 ± 0.23 dB**, SSIM 0.836 ± 0.006 (seeds: 35.734, 35.535, 35.270); augmented **35.86 ± 0.26 dB**, SSIM 0.850 ± 0.008 (seeds: 35.779, 36.157, 35.651). Augmented higher in all 3 seeds (paired gains +0.05, +0.62, +0.38), mean +0.35 dB, mean SSIM +0.014: suggestive, not conclusive.
- **Official benchmark, hidden ground truth, Kaggle sRGB (seed-0 models, one submission each):** PSNR baseline **36.2546**, augmented **36.2904** (+0.036, a tie); SSIM baseline **0.842252**, augmented **0.849825** (+0.0076). Seed 0 is the least favourable seed for augmentation on the validation blocks, so the Kaggle PSNR tie understates the average gain.
- Caveat: these models train on about 75% of SIDD-Small with a small network for 10 epochs; **do not compare to published numbers.**

### 4.6 S6 ablations (held-out S6, seed 0 unless noted; PSNR / SSIM)
Baseline seeds 0/1/2: 36.02/35.79/36.46 (SSIM 0.825 mean). Poisson-Gaussian seeds 0/1/2: 36.75/36.48/36.79 (SSIM 0.852 mean). Gaussian-only seed 0: 36.70 / 0.855. Noise 3x stronger (`--aug_mult 3`): 36.70 / 0.846 (**resumed after an interruption with a learning-rate bug, marked with an asterisk**; later fixed). Augmentation probability 0.9: 36.98 / 0.866. All variants lie between 36.5 and 37.0, above all baseline seeds (max 36.46), and differ from each other by less than the seed spread: variants cannot be ranked.

### 4.7 Training-length check (S6 held out, 3 seeds each, complete)
| Setting | Baseline | + Noise augmentation | Paired gain (aug - base) |
|---|---|---|---|
| 10 epochs (main setting), 3 seeds | 36.09 ± 0.34 | 36.67 ± 0.17 | +0.58 ± 0.22 |
| 30 epochs, 3 seeds | 37.97 ± 0.14 (seeds: 38.113, 37.844, 37.953) | 37.91 ± 0.36 (seeds: 38.195, 37.503, 38.020) | **-0.06 ± 0.24** (per seed +0.082, -0.340, +0.066) |

30-epoch SSIM: baseline 0.8860 ± 0.0054, augmented 0.8875 ± 0.0077.

Reading: training 3x longer raised held-out S6 PSNR by about **1.9 dB (baseline)** and **1.2 dB (augmented)**, and the augmentation gain **disappeared** (-0.06 ± 0.24 dB, inside the seed noise). The 10-epoch gain is therefore best read as faster early learning / regularisation of an undertrained model, not as a lasting benefit. This check covers S6 only; the cross-camera drop and the Gaussian-only control were not repeated at 30 epochs, and even the 30-epoch models were not fully converged (loss still slowly falling).

### 4.8 Qualitative
`visualize.py --cam S6` produced `results/visual_S6.png` (noisy / baseline / augmented / ground truth, zoom and error maps). On the default block: noisy 33.18 dB, baseline 39.60 dB, augmented 41.65 dB; this is a smooth patch with a bigger gain than the S6 average (+0.58 dB), so show the mean gain next to it and, if possible, a block where augmentation does not help (use `--scene` and `--block` to choose others).

## 5. What the results support / do not support

| Claim | Status |
|---|---|
| A model trained on 4 cameras loses some PSNR on an unseen camera | Supported, small (about 0.4 dB mean; about 0.9 dB for IP and S6), near seed noise |
| Noise augmentation improves unseen-camera PSNR | Weak: +0.19 dB mean; clear only on S6 (+0.58 dB); none on N6/G4; **disappears with 30-epoch training on S6 (3 seeds: -0.06 ± 0.24 dB)** |
| It improves SSIM modestly | Supported (S6 +0.027; others 0 to +0.008; official val +0.014) |
| It reduces seed-to-seed variability | Observed on all five cameras (3 seeds) |
| It reduces the cross-camera drop | **Not supported** on average (0.52 versus 0.43 dB) |
| The Poisson (signal-dependent) part is what helps | **Not supported** (Gaussian-only is equivalent) |
| Learned denoiser beats classical methods | Strongly supported (about 4 dB) |
| Results are comparable to published SIDD numbers | **No** (small model, 10 epochs, 75% of SIDD-Small) |

## 6. Limitations to state honestly
Only 5 cameras; small test sets (G4 has 4 test scenes, N6 6); 3 seeds (std estimates are themselves rough); small lightweight model trained for 10 epochs (undertrained: 30 epochs improved S6 by about 1.2-1.9 dB); the Gaussian control used 3 seeds but the size-matched controls and ablations used one; sRGB noise is spatially correlated and non-monotonic in intensity, so the Poisson-Gaussian model is only an approximation; the hold-out models train on 20-35% fewer patches than the all-camera model; evaluation uses up to 200 blocks per test image in reading order (large images are covered almost fully, but not randomly); DND and LPIPS not done; held-out-camera numbers are on our own fixed block grid over reserved scenes (the official blocks carry no camera labels as far as known), not the official benchmark.

## 7. Corrections made during the project (for transparency)
Midterm slide said about 32 scenes per camera (real counts above); midterm theory slide said brighter pixels have larger noise variance (not true for this sRGB data); the first cross-camera-drop definition compared different cameras and the second tested the seen model on its own training scenes, so a fixed reserved test set was introduced; the first Colab/local runs crashed from RAM because the patch cache held all 128-pixel patches of every 12-megapixel image (fixed with random crops on disk); a Windows glob bug duplicated files (fixed); `baselines.py` first used only corner blocks (fixed to evenly spread blocks); a learning-rate bug in resuming interrupted runs (fixed; only the 3x-noise ablation was affected); the single-seed claim "augmentation halves the drop (0.79 to 0.42)" was retracted after the 3-seed runs.

## 8. Code map (folder `noiseshift/`)
| File | Purpose |
|---|---|
| `config.py` | Paths and the shared experiment settings (patch size, patches per image, test fraction, epochs, seed): must be identical for every run |
| `data.py` | Index SIDD-Small, parse camera from folder name, fixed scene split, patch cache, fixed evaluation blocks |
| `model.py` | `SmallDenoiser` (residual DnCNN-style CNN) |
| `augment.py` | Poisson-Gaussian and Gaussian-only noise augmentation |
| `metrics.py` | PSNR, SSIM, block evaluation, cross-camera drop, official block loader |
| `train.py` | Training (options: `--hold K`, `--aug`, `--aug_gauss`, `--aug_p`, `--aug_mult`, `--match K`, `--seed`, `--epochs`); per-epoch checkpoints and exact resume; loss history in `results/*_loss.json` |
| `evaluate.py` | Scores a checkpoint on a camera's test scenes (`--cams`) and on the official validation blocks (`--official`); results in `results/<ckpt>.json` |
| `run_fold.py`, `run_queue.py` | Run all experiments (resumable queue: gauss, seeds, match, allseeds, gaussseeds phases) |
| `aggregate_all.py` | Result tables over seeds, controls and size-matched models (`results/summary_full.md`, `gain_bars.png`). (The old seed-0-only `aggregate.py` is not part of the final repository.) |
| `noise_analysis.py` | Noise statistics of the dataset |
| `baselines.py` | Classical-method comparison on identical blocks |
| `visualize.py`, `make_figures.py` | Qualitative figure and slide figures |
| `benchmark_submit.py` | Creates the Kaggle submission CSV from the benchmark noisy blocks |
| `official_summary.py`, `e30_summary.py` | Seed summaries for the official validation blocks and the 30-epoch check |
| `test_synthetic.py` | Sanity tests on synthetic data (split, cache, blocks, PSNR/SSIM, augmentation) |

Reproduce: place data in `content/` (SIDD folder and `.mat` files), `pip install -r requirements.txt` plus CUDA PyTorch, `python test_synthetic.py`, `python run_fold.py all` and `python run_fold.py <GP|IP|S6|N6|G4>` or `python run_queue.py`, then `python aggregate_all.py`, `python make_figures.py`.

## 9. Literature to cover (verify details before presenting; these are standard references)
- Noise and datasets: SIDD (Abdelhamed, Lin, Brown, CVPR 2018); DND (Plotz and Roth, CVPR 2017); Poisson-Gaussian noise modelling (Foi et al., IEEE TIP 2008).
- Classical: Gaussian/median filtering; Non-Local Means (Buades, Coll, Morel, CVPR 2005): average similar patches across the image; BM3D (Dabov et al., IEEE TIP 2007): group similar patches into 3-D stacks, filter in a transform domain, needs the noise level, assumes white Gaussian noise.
- Learning-based: DnCNN (Zhang et al., IEEE TIP 2017): residual learning, predicts noise; CBDNet (Guo et al., CVPR 2019): blind real-photo denoising with a noise-estimation subnetwork, trained on synthetic Poisson-Gaussian plus real noise (closest to our idea); NAFNet (Chen et al., ECCV 2022) and Restormer (Zamir et al., CVPR 2022): stronger modern backbones (related work only, not trained here).

## 10. Presentation plan (PDF; every point answers a teacher requirement)
1. Title and team. 2. **Problem statement and gap** (cross-camera generalization, SIDD/DND). 3. **Objectives** (quantify the gap; test noise augmentation; follow protocol; report honestly). 4. **Literature review** (section 9, grouped: datasets, classical, deep, real-noise methods; connect to our idea). 5. **Theory** (noise model: raw Poisson-Gaussian versus sRGB reality; PSNR and SSIM formulas; residual learning; cross-camera drop definition). 6. **Methodology** (backbone, training, fixed test scenes, hold-out design, augmentation and Gaussian control; one pipeline diagram). 7. **Dataset** (real counts, ISO ranges, noise plots; test/train scenes). 8. **Experiments** (list: 5 folds x base/aug x 3 seeds, Gaussian control, size-matched controls, ablations, classical baselines, official validation and Kaggle, 30-epoch check). 9. **Results: unseen-camera table and chart** (4.1; `fig_unseen_seeds.png`, `gain_bars.png`). 10. **Cross-camera drop** (4.3; `fig_drop.png`; state plainly that augmentation does not reduce the mean drop). 11. **Does the Poisson part matter?** (4.2). 12. **Comparison with standard methods** (4.4; `fig_methods.png`). 13. **Official protocol** (4.5). 14. **Does training longer change the picture?** (4.7). 15. **Qualitative results** (`visual_S6.png`, `visual_IP.png`). 16. **Limitations, conclusion and future work** (sections 5, 6; future: stronger backbone, longer training for all models, DND, noise matched to the camera pipeline, correlated-noise synthesis).

Suggested one-sentence conclusion: "A small denoiser loses about 0.4 dB on an unseen camera; noise augmentation gives a small average gain (+0.2 dB, clear on S6 only) that disappears with longer training (30 epochs, 3 seeds), does not reduce the average cross-camera drop, and Gaussian noise does as well as Poisson-Gaussian."

Figures to use: `noise_vs_iso.png`, `noise_vs_intensity.png`, `fig_unseen_seeds.png`, `gain_bars.png`, `fig_drop.png`, `fig_methods.png`, `visual_S6.png`, `visual_IP.png`, `train_curves.png`. **Do not use** `drop_bar.png`, `unseen_bar.png`, `summary.md`, `summary.csv` (seed-0 only).

## 11. Questions the team should be ready to answer
- Why does BM3D do badly here? Assumes white Gaussian noise of known level; the real noise is correlated and the blind estimate is too low; with the true level it recovers about 3.7 dB but still trails DnCNN.
- Why Poisson-Gaussian, and why did the Gaussian control match it? Standard raw-sensor noise model; in sRGB the processing pipeline breaks the assumption, and our control shows the signal-dependence does not matter here.
- Why fixed test scenes and a hold-out design? So seen and unseen models are compared on identical scenes; avoids testing on training images; avoids arbitrary random crops.
- Why is the gain small, and why did longer training shrink it? Small models and short training leave room for regularisation to help; with more training the baseline catches up (one-seed check).
- Why not NAFNet or the official benchmark for camera labels? Compute and time; official blocks have no camera labels as far as known.
- What is residual learning? The network predicts the noise and subtracts it from the input, which is easier than predicting the clean image directly.
- How are PSNR and SSIM defined? PSNR = 10 log10(1/MSE) on [0,1] images; SSIM compares local luminance, contrast and structure.
- Why L1 loss, why patches, why a cosine schedule? Robust to outliers and tends to give sharper results than L2; patches make training feasible and give many samples; cosine decay lowers the learning rate smoothly to settle training.

## 12. Remaining to-do list
1. DONE: all experiments (main grid, 3 seeds, Gaussian control with 3 seeds, size-matched controls, ablations, classical baselines, official validation and Kaggle, 30-epoch check with 3 seeds).
2. Run `python make_figures.py` (and `python aggregate_all.py`), open every figure and check it before use; `results/baselines.md` and `results/summary_full.md` hold the final tables.
3. Build the PPT per section 10 and export to **PDF**; submit it together with the code **at least 2 hours before the slot**. Bring the PDF and the code on a pendrive/laptop.
4. DONE: code commented (every script has a header docstring and explanatory comments; verified to behave identically to the tested version). Do not include datasets, checkpoints or the 336 MB submission CSVs.
5. Every member rehearses theory, literature and code (see `STUDY_GUIDE.md`); suggested split of the talk: member 1 theory and literature, member 2 code and experiments, member 3 results and analysis, but **each must be able to explain everything**.
6. Optional: DND submission (not started; list as future work).

## 13. Final repository layout (what to upload and submit)
```
README.md   requirements.txt   .gitignore
config.py data.py model.py augment.py metrics.py          core library
train.py evaluate.py run_fold.py run_queue.py             training / evaluation / experiment runners
aggregate_all.py make_figures.py visualize.py             tables and figures
noise_analysis.py baselines.py                            dataset noise study, classical methods
benchmark_submit.py official_summary.py e30_summary.py    benchmark CSV and seed summaries
test_synthetic.py                                         unit tests (no data or GPU needed)
results/   final JSON results, tables (.md/.csv) and figures (.png)
logs/      raw training/evaluation logs
docs/      CONTEXT.md, STUDY_GUIDE.md
```
Not in the repository: `content/` (dataset), `ckpt/` (checkpoints), `submission/` (Kaggle CSVs), `.venv/`.
