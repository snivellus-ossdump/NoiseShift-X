"""Camera-labelled SIDD-Small indexing, patch cache and held-out-camera splits.

SIDD-Small sRGB layout (verify on your download):
  <ROOT>/<scene-instance>_<scene>_<CAM>_<ISO>_<shutter>_<temp>_<bright>/
        0001_NOISY_SRGB_010.PNG
        0001_GT_SRGB_010.PNG
The camera code is the 3rd underscore-separated field of the folder name.
"""
import os, glob, re
import numpy as np
from PIL import Image

def parse_camera(folder_name):
    # Folder names look like instance_scene_CAMERA_ISO_shutter_colourtemp_brightness,
    # e.g. 0001_001_S6_00100_00060_3200_L, so the camera is the 3rd underscore-separated field.
    parts = os.path.basename(folder_name.rstrip("/")).split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected folder name: {folder_name}")
    return parts[2]

def _find_pair(folder):
    # set(): on Windows glob is case-insensitive, so *.PNG and *.png both match the same file
    noisy = sorted(set(glob.glob(os.path.join(folder, "*NOISY*SRGB*.PNG")) + glob.glob(os.path.join(folder, "*NOISY*SRGB*.png"))))
    gt = sorted(set(glob.glob(os.path.join(folder, "*GT*SRGB*.PNG")) + glob.glob(os.path.join(folder, "*GT*SRGB*.png"))))
    return noisy, gt

def index_sidd(root):
    """Return list of dicts: {camera, folder, noisy, gt}. Fails loudly on mismatches."""
    # One entry per scene-instance folder; each folder must contain exactly one noisy and one ground-truth PNG.
    items = []
    for folder in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(folder):
            continue
        noisy, gt = _find_pair(folder)
        if len(noisy) != 1 or len(gt) != 1:
            raise RuntimeError(f"{folder}: expected exactly 1 noisy and 1 GT file, got {len(noisy)}/{len(gt)}")
        items.append(dict(camera=parse_camera(folder), folder=folder, noisy=noisy[0], gt=gt[0]))
    if not items:
        raise RuntimeError(f"No scene folders found in {root}")
    return items

# Legacy whole-camera split, kept for reference and the unit test. The reported experiments use
# split_scenes() (fixed test scenes) together with the --hold option of train.py.
def split_by_camera(items, held_out):
    train = [it for it in items if it["camera"] != held_out]
    test = [it for it in items if it["camera"] == held_out]
    if not test:
        raise RuntimeError(f"No images for held-out camera {held_out}")
    return train, test

def split_scenes(items, test_frac, seed):
    """Fixed scene-level train/test split, stratified by camera. Test scenes are NEVER used for training by any
    model, so a 'seen-camera' model and a 'held-out-camera' model can be compared on the SAME test scenes."""
    import math
    # Seeded generator => exactly the same test scenes on every machine and every run.
    rng = np.random.default_rng(seed)
    train, test = [], []
    for cam in sorted({it["camera"] for it in items}):
        # All scenes of this camera, in a fixed (alphabetical) order.
        grp = sorted([it for it in items if it["camera"] == cam], key=lambda it: it["folder"])
        # Number of test scenes for this camera: test_frac of them (rounded up), at least 3,
        # but always leave at least one scene for training.
        k = min(len(grp) - 1, max(3, math.ceil(test_frac * len(grp))))
        # Randomly choose which scenes of this camera become test scenes.
        pick = set(rng.choice(len(grp), size=k, replace=False).tolist())
        for i, it in enumerate(grp):
            (test if i in pick else train).append(it)
    return train, test

def load_png(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def build_patch_cache(items, prefix, size, per_image, seed=0):
    """Random-crop `per_image` aligned noisy/GT patches from every scene and write them to disk as
    uint8 .npy files (memory-mapped; one image is in RAM at a time). Files: <prefix>_noisy.npy,
    <prefix>_gt.npy, <prefix>_cam.npy (written last, so its presence means the cache is complete)."""
    rng = np.random.default_rng(seed)
    # Pre-allocate disk-backed uint8 arrays (np.memmap) so only one full image is in RAM at a time.
    # (An earlier version kept every patch in RAM and ran out of memory.)
    N = len(items) * per_image
    shape = (N, size, size, 3)
    nm = np.lib.format.open_memmap(prefix + "_noisy.npy", mode="w+", dtype=np.uint8, shape=shape)
    gm = np.lib.format.open_memmap(prefix + "_gt.npy", mode="w+", dtype=np.uint8, shape=shape)
    cams, k = [], 0
    # Loop over scenes: load the noisy and the ground-truth image, then cut `per_image` random patches.
    for it in items:
        n, g = load_png(it["noisy"]), load_png(it["gt"])
        assert n.shape == g.shape, f"shape mismatch in {it['folder']}"
        H, W, _ = n.shape
        for _ in range(per_image):
            # Same random position in the noisy image and its ground truth => pixel-aligned (noisy, clean) pair.
            y = int(rng.integers(0, H - size + 1)); x = int(rng.integers(0, W - size + 1))
            nm[k] = n[y:y+size, x:x+size]; gm[k] = g[y:y+size, x:x+size]
            cams.append(it["camera"]); k += 1
        del n, g
    nm.flush(); gm.flush(); del nm, gm
    # Written last: its existence marks the cache as complete (a crash mid-build leaves no _cam.npy).
    np.save(prefix + "_cam.npy", np.array(cams))
    return N

def cache_ready(prefix):
    return all(os.path.exists(prefix + s) for s in ("_noisy.npy", "_gt.npy", "_cam.npy"))

def load_cache(prefix, held_out=None):
    """Memory-mapped arrays + indices of patches to train on (excludes the held-out camera)."""
    n = np.load(prefix + "_noisy.npy", mmap_mode="r"); g = np.load(prefix + "_gt.npy", mmap_mode="r")
    cam = np.load(prefix + "_cam.npy")
    # Indices of the patches whose camera is NOT the held-out one (no filtering if held_out is None).
    idx = np.where(cam != held_out)[0] if held_out else np.arange(len(cam))
    return n, g, cam, idx

def fixed_eval_blocks(noisy, gt, block=256, max_blocks=None):
    """Deterministic non-overlapping block grid (NOT random crops) for held-out-camera eval."""
    H, W, _ = noisy.shape
    # Deterministic grid: a block starts every `block` pixels, no overlap, no randomness.
    blocks = []
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            blocks.append((noisy[y:y+block, x:x+block], gt[y:y+block, x:x+block]))
    return blocks[:max_blocks] if max_blocks else blocks
