# NoiseShift-X: cross-camera real image denoising

Course project, DSE-312 Computer Vision (Dr. Samiran Das), IISER Bhopal.
Team: Suryansh Upadhyay (24342), Prakhar Jain (24253), Siddharth (24325).

**Question.** How much does a small CNN denoiser lose on a smartphone camera it never saw (leave-one-camera-out on SIDD-Small), and does noise augmentation reduce that loss?

## Main findings (all measured by us; details in `docs/CONTEXT.md` and `results/`)
- A DnCNN-style denoiser (10 layers, about 0.3 M parameters) loses about **0.4 dB** PSNR on an unseen camera on average (about 0.9 dB for IP and S6).
- Noise augmentation gave a small average gain on the unseen camera (**+0.19 dB**, 3 seeds), clear only on S6 (+0.58 ± 0.22 dB). It did **not** reduce the average cross-camera drop (0.52 vs 0.43 dB).
- Gaussian-only noise performed the same as Poisson-Gaussian noise (mean difference +0.05 dB): the signal-dependent part did not matter.
- With 3x longer training (30 epochs, S6, 3 seeds) the augmentation gain disappeared (-0.06 ± 0.24 dB).
- The learned model beat classical methods (Gaussian, median, NLM, BM3D) by about 4 dB PSNR on identical blocks.
- Official SIDD benchmark (hidden ground truth, Kaggle sRGB): PSNR 36.2546 (baseline) vs 36.2904 (augmented); SSIM 0.842252 vs 0.849825. Official validation blocks, 3 seeds: 35.51 ± 0.23 vs 35.86 ± 0.26 dB. These models use about 75% of SIDD-Small, a small network and 10 epochs, so they are **not comparable to published results**.

## Protocol
Fixed scene-level test set (about 25% of the scenes of each camera, never trained on); fixed non-overlapping 256x256 evaluation blocks (never random crops); seen and unseen models compared on the same test scenes; 3 training seeds; cross-camera drop = PSNR(seen) - PSNR(unseen).

## Setup and running
```
pip install -r requirements.txt        # plus the CUDA build of PyTorch
# place the data in content/ :
#   content/SIDD_Small_sRGB_Only/Data/<scene folders>
#   content/ValidationNoisyBlocksSrgb.mat, content/ValidationGtBlocksSrgb.mat
python test_synthetic.py                # sanity tests, no data or GPU needed
python run_fold.py all                  # seen-camera models (baseline + augmented)
python run_fold.py S6                   # one held-out camera (GP, IP, S6, N6, G4)
python run_queue.py                     # seeds, controls, size-matched models
python aggregate_all.py                 # results/summary_full.md
python make_figures.py                  # figures
python noise_analysis.py                # noise statistics of the dataset
python baselines.py --per_scene 3 --oracle_sigma   # classical comparison
```
All settings that must be identical across runs are in `config.py`.

## Files
`config.py data.py model.py augment.py metrics.py` core library; `train.py evaluate.py run_fold.py run_queue.py` training and experiments; `aggregate_all.py make_figures.py visualize.py` tables and figures; `noise_analysis.py baselines.py` dataset noise study and classical methods; `benchmark_submit.py` Kaggle submission CSV; `official_summary.py e30_summary.py` seed summaries; `test_synthetic.py` unit tests. `results/` holds final result files, `logs/` raw logs, `docs/` project context and study guide. Datasets, checkpoints and submission CSVs are not included.

## Limitations
Five cameras; small test sets for G4 (4 scenes) and N6 (6); 3 seeds; a small model trained for 10 epochs (30 epochs improved S6 by about 1.2-1.9 dB); sRGB noise is spatially correlated, so the Poisson-Gaussian model is only an approximation; DND and LPIPS were not done.

## References
SIDD: Abdelhamed, Lin, Brown, CVPR 2018. DND: Plotz, Roth, CVPR 2017. DnCNN: Zhang et al., IEEE TIP 2017. BM3D: Dabov et al., IEEE TIP 2007. Non-Local Means: Buades, Coll, Morel, CVPR 2005. CBDNet: Guo et al., CVPR 2019. NAFNet: Chen et al., ECCV 2022. Restormer: Zamir et al., CVPR 2022. Poisson-Gaussian noise modelling: Foi et al., IEEE TIP 2008.
