"""Sanity tests on synthetic data (no torch, no real SIDD needed)."""
import os, tempfile, numpy as np
from PIL import Image
from data import index_sidd, split_by_camera, split_scenes, build_patch_cache, fixed_eval_blocks, parse_camera
from metrics import psnr, ssim, eval_blocks, cross_camera_drop
from augment import add_poisson_gaussian, maybe_augment

# Build a tiny fake SIDD-like dataset: 10 scenes (2 per camera) with synthetic noise. No real data or GPU needed.
rng = np.random.default_rng(0)
root = tempfile.mkdtemp()
cams = ["GP", "IP", "S6", "N6", "G4"]
for i, c in enumerate(cams * 2):
    f = os.path.join(root, f"{i+1:04d}_{i+1:03d}_{c}_00100_00060_3200_L"); os.makedirs(f)
    gt = (rng.random((512, 512, 3)) * 200 + 20).astype(np.uint8)
    noisy = np.clip(gt + rng.normal(0, 15, gt.shape), 0, 255).astype(np.uint8)
    Image.fromarray(noisy).save(f + f"/{i+1:04d}_NOISY_SRGB_010.PNG"); Image.fromarray(gt).save(f + f"/{i+1:04d}_GT_SRGB_010.PNG")

# 1. Indexing and camera parsing.
items = index_sidd(root); assert len(items) == 10 and {it["camera"] for it in items} == set(cams); print("index OK:", len(items), "scenes")
tr, te = split_by_camera(items, "S6"); assert len(te) == 2 and all(it["camera"] != "S6" for it in tr); print("split OK: train", len(tr), "test", len(te))
assert not (set(i["folder"] for i in tr) & set(i["folder"] for i in te)); print("no train/test leakage by folder")
from data import cache_ready, load_cache
# 2. Fixed scene split: train and test are disjoint, deterministic, one test scene per camera here.
trn, tst = split_scenes(items, 0.25, 0); assert len(tst) == 5 and len(trn) == 5 and not (set(i["folder"] for i in trn) & set(i["folder"] for i in tst))
assert {i["camera"] for i in tst} == set(cams); assert [i["folder"] for i in split_scenes(items, 0.25, 0)[1]] == [i["folder"] for i in tst]; print("scene split OK: 1 test scene per camera, disjoint, deterministic")
# 3. Patch cache and held-out-camera filtering.
pre = os.path.join(root, "cache"); n = build_patch_cache(trn, pre, 128, 20); assert cache_ready(pre) and n == 5 * 20
nm, gm, cam, idx = load_cache(pre, "S6"); assert nm.shape == (n, 128, 128, 3) and "S6" not in set(cam[idx]) and len(idx) == 4 * 20
print("patch cache OK:", n, "patches;", len(idx), "usable when S6 held out")
# 4. Evaluation blocks: a 512 x 512 image gives exactly four non-overlapping 256 x 256 blocks.
b = fixed_eval_blocks(np.zeros((512, 512, 3), np.uint8), np.zeros((512, 512, 3), np.uint8), 256); assert len(b) == 4; print("fixed blocks OK:", len(b))
# 5. Metrics: PSNR must match the closed form; SSIM of identical images must be 1.
a = np.full((64, 64, 3), 100, np.uint8); assert psnr(a, a) == float("inf") and abs(psnr(a, a + 10) - 10 * np.log10(255**2 / 100)) < 1e-6; print("PSNR matches closed form")
assert abs(ssim(a, a) - 1.0) < 1e-9; print("SSIM identity OK")
# 6. Augmentation: with a > 0 the noise variance must grow with brightness.
x = np.full((256, 256, 3), 128, np.uint8); y, (aa, bb) = add_poisson_gaussian(x, rng); print("aug std(noise)=%.2f  a=%.5f b=%.5f" % ((y.astype(float) - 128).std(), aa, bb))
bright = np.full((256, 256, 3), 220, np.uint8); dark = np.full((256, 256, 3), 20, np.uint8)
r = np.random.default_rng(1); sb = np.mean([(add_poisson_gaussian(bright, np.random.default_rng(s), (1e-3,1e-3),(1e-5,1e-5))[0].astype(float)-220).std() for s in range(5)])
sd = np.mean([(add_poisson_gaussian(dark, np.random.default_rng(s), (1e-3,1e-3),(1e-5,1e-5))[0].astype(float)-20).std() for s in range(5)])
assert sb > sd; print("noise is signal-dependent: std bright %.2f > dark %.2f" % (sb, sd))
print("drop example:", cross_camera_drop(38.0, 35.5))
print("ALL TESTS PASSED")
