"""Train a baseline or augmented denoiser, optionally holding out one camera.

  python train.py --name base_all                    # seen-camera reference (all cameras, train scenes only)
  python train.py --name aug_all  --aug
  python train.py --name base_hold_S6 --hold S6      # camera S6 never seen
  python train.py --name aug_hold_S6  --hold S6 --aug
Test scenes (data.split_scenes) are excluded from every model.
"""
import argparse, os, json, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config as C
from data import index_sidd, split_scenes, build_patch_cache, cache_ready, load_cache
from augment import maybe_augment
from model import SmallDenoiser

# Serves (noisy, clean) patch pairs from the on-disk cache; augmentation is applied on the fly here.
class PatchDS(Dataset):
    def __init__(self, n, g, idx, aug, aug_p=0.5, aug_mult=1.0, gauss_only=False):
        # n, g: memory-mapped noisy / clean patch arrays; idx: the patches this model is allowed to train on.
        self.n, self.g, self.idx, self.aug, self.rng = n, g, idx, aug, None
        # Probability that a patch receives extra synthetic noise.
        self.p = aug_p
        # Default (Poisson-Gaussian) noise ranges; aug_mult scales them for the ablation.
        self.kw = dict(a_range=(1e-4 * aug_mult, 5e-3 * aug_mult), b_range=(1e-5 * aug_mult, 2e-3 * aug_mult))
        if gauss_only:   # control: signal-independent Gaussian noise with a similar average added variance (~8.5e-4)
            self.kw = dict(a_range=(1e-4, 5e-3), b_range=(1.5e-4 * aug_mult, 2.5e-3 * aug_mult), gauss_only=True)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        if self.rng is None:   # seeded lazily so every DataLoader worker/epoch gets its own stream
            self.rng = np.random.default_rng(torch.initial_seed() % (2**32))
        # Fetch the cached patch pair (np.array copies it out of the memory map).
        j = self.idx[i]; n, g = np.array(self.n[j]), np.array(self.g[j])
        k = int(self.rng.integers(4)); n, g = np.rot90(n, k), np.rot90(g, k)   # simple rotation aug
        if self.aug: n = maybe_augment(n, self.rng, p=self.p, **self.kw)
        # uint8 H x W x 3 array -> float tensor 3 x H x W scaled to [0, 1].
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
        return t(n), t(g)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--hold", default=None)
    ap.add_argument("--aug", action="store_true"); ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aug_p", type=float, default=0.5, help="probability of adding Poisson-Gaussian noise (ablation)")
    ap.add_argument("--aug_mult", type=float, default=1.0, help="scales the (a,b) noise ranges (ablation)")
    ap.add_argument("--aug_gauss", action="store_true", help="control: Gaussian-only (signal-independent) augmentation; use together with --aug")
    ap.add_argument("--match", default=None, help="size-matched control for camera K: all cameras, but only as many train scenes as the hold-K model uses")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=C.SEED, help="training seed (init, data order, augmentation); the test split never changes")
    a = ap.parse_args()
    # The seed controls weight initialisation, data order and augmentation noise (the test split never changes).
    torch.manual_seed(a.seed)
    for d in (C.CKPT_DIR, C.CACHE_DIR, C.RESULTS_DIR): os.makedirs(d, exist_ok=True)
    # All experiments share ONE patch cache built from the TRAIN scenes only; test scenes never enter it.
    prefix = os.path.join(C.CACHE_DIR, "trainval")     # ONE cache (train scenes only); the held-out camera is filtered below
    if not cache_ready(prefix):
        train_items, test_items = split_scenes(index_sidd(C.SIDD_SMALL_ROOT), C.TEST_FRAC, C.SEED)
        print(f"building patch cache from {len(train_items)} train scenes ({len(test_items)} test scenes kept out) ...", flush=True)
        print("patches:", build_patch_cache(train_items, prefix, C.PATCH, C.PATCHES_PER_IMAGE, C.SEED), flush=True)
    # idx = patches this model trains on: everything except the held-out camera (if --hold is given).
    n_arr, g_arr, cam, idx = load_cache(prefix, a.hold)
    # Safety check: the held-out camera must not appear among the training patches.
    assert not a.hold or a.hold not in set(cam[idx]), "held-out camera leaked into training"
    # Size-matched control: train on all cameras but with only as many scenes as the hold-out model had,
    # so 'less training data' can be separated from 'camera never seen'.
    if a.match:
        assert not a.hold, "use either --hold or --match"
        # The cache stores PATCHES_PER_IMAGE consecutive patches per scene, so patch index // P is the scene.
        P = C.PATCHES_PER_IMAGE; scene_cam = cam[::P]; n_keep = int((scene_cam != a.match).sum())
        keep = np.sort(np.random.default_rng(C.SEED + 1000 + C.CAMERAS.index(a.match)).choice(len(scene_cam), size=n_keep, replace=False))
        idx = (keep[:, None] * P + np.arange(P)[None]).ravel()
        print(f"size-matched control for {a.match}: {n_keep}/{len(scene_cam)} train scenes, {int((scene_cam[keep] == a.match).sum())} of them from {a.match}", flush=True)
    print(f"training patches: {len(idx)} (hold={a.hold}, aug={a.aug})", flush=True)
    # Use the GPU when available.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PatchDS(n_arr, g_arr, idx, a.aug, a.aug_p, a.aug_mult, a.aug_gauss)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    # Model and Adam optimiser.
    m = SmallDenoiser().to(dev); opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    # Cosine decay: lr(e) = lr0 * 0.5 * (1 + cos(pi * e / E)), from lr0 at epoch 0 towards 0 at the end.
    lr_at = lambda e: a.lr * 0.5 * (1 + math.cos(math.pi * e / a.epochs))   # cosine schedule, set explicitly so resume is exact
    loss_fn = nn.L1Loss()
    # A checkpoint is saved after every epoch, so an interrupted run resumes where it stopped.
    ck = os.path.join(C.CKPT_DIR, a.name + ".pt"); start = 0
    if os.path.exists(ck):                                  # resume after an interruption
        s = torch.load(ck, map_location=dev); m.load_state_dict(s["m"]); opt.load_state_dict(s["o"]); start = s["e"] + 1
    # Per-epoch training loss history (used for the training-curve plot).
    hist_path = os.path.join(C.RESULTS_DIR, a.name + "_loss.json")
    hist = json.load(open(hist_path))[:start] if os.path.exists(hist_path) else []
    # Training loop: one pass over all training patches per epoch.
    for e in range(start, a.epochs):
        for g in opt.param_groups: g['lr'] = lr_at(e)
        # Training mode (batch normalisation uses batch statistics).
        m.train(); tot = 0
        # One mini-batch of 32 (noisy, clean) patch pairs per iteration.
        for n, g in dl:
            # Standard step: forward pass, L1 loss against the clean patch, backpropagation, parameter update.
            n, g = n.to(dev), g.to(dev); opt.zero_grad(); l = loss_fn(m(n), g); l.backward(); opt.step(); tot += l.item()
        hist.append(tot / len(dl)); print(f"epoch {e+1}/{a.epochs} L1={hist[-1]:.5f}", flush=True)
        # Save weights, optimiser state and epoch number so training can resume exactly.
        torch.save(dict(m=m.state_dict(), o=opt.state_dict(), e=e), ck); json.dump(hist, open(hist_path, "w"))

if __name__ == "__main__":
    main()
