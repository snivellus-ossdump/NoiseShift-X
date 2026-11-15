"""Evaluate a checkpoint; results are merged into results/<ckpt-name>.json.

  python evaluate.py --ckpt ckpt/base_hold_S6.pt --cams S6
  python evaluate.py --ckpt ckpt/base_all.pt --cams GP IP S6 N6 G4 --official
--cams    PSNR/SSIM on the fixed 256x256 block grid of that camera's TEST scenes (same scenes for every model).
--official PSNR/SSIM on the official SIDD validation blocks (32 x 256 x 256 x 40; mixes all cameras).
"""
import argparse, os, json, numpy as np, torch
import config as C
from data import index_sidd, split_scenes, load_png, fixed_eval_blocks
from metrics import eval_blocks, load_official_blocks
from model import SmallDenoiser

# Run the network on a list of uint8 blocks without gradients; returns uint8 denoised blocks.
@torch.no_grad()
def denoise_blocks(model, blocks, dev, bs=8):
    out = []
    for i in range(0, len(blocks), bs):
        # A batch of blocks -> float tensor N x 3 x H x W with values in [0, 1].
        x = torch.from_numpy(np.stack(blocks[i:i+bs])).permute(0, 3, 1, 2).float().div(255).to(dev)
        # Denoise, clip to the valid range and round to uint8, exactly as if the result were saved as an image.
        y = model(x).clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        out += list(y)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cams", nargs="*", default=[]); ap.add_argument("--official", action="store_true")
    ap.add_argument("--max_blocks", type=int, default=200, help="max blocks per image")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the trained weights and switch to evaluation mode (batch norm uses running statistics).
    m = SmallDenoiser().to(dev); m.load_state_dict(torch.load(a.ckpt, map_location=dev)["m"]); m.eval()
    name = os.path.splitext(os.path.basename(a.ckpt))[0]
    os.makedirs(C.RESULTS_DIR, exist_ok=True); out_path = os.path.join(C.RESULTS_DIR, name + ".json")
    # Results are merged into one JSON file per checkpoint so repeated evaluations accumulate.
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res["ckpt"] = name
    # (1) Official SIDD validation blocks: 40 images x 32 blocks of 256 x 256, all cameras mixed.
    if a.official:
        n, g = load_official_blocks(C.OFFICIAL_VAL_NOISY, C.OFFICIAL_VAL_GT)
        p, s = eval_blocks(denoise_blocks(m, list(n), dev), list(g))
        res["official_val"] = dict(psnr=p, ssim=s, n_blocks=len(n))
        print("OFFICIAL SIDD val blocks  PSNR/SSIM: %.3f / %.4f" % (p, s), flush=True)
    # (2) Per-camera evaluation on that camera's reserved TEST scenes (identical scenes for every model).
    if a.cams:
        # split_scenes returns (train, test); [1] is the list of test scenes.
        test_items = split_scenes(index_sidd(C.SIDD_SMALL_ROOT), C.TEST_FRAC, C.SEED)[1]
        for cam in a.cams:
            items = [it for it in test_items if it["camera"] == cam]
            assert items, f"no test scenes for camera {cam}"
            # Collect noisy / ground-truth blocks of this camera's test scenes (fixed grid, at most max_blocks per image).
            nb, gb = [], []
            for it in items:
                for pn, pg in fixed_eval_blocks(load_png(it["noisy"]), load_png(it["gt"]), C.EVAL_BLOCK, a.max_blocks):
                    nb.append(pn); gb.append(pg)
            # Mean PSNR and SSIM over all collected blocks.
            p, s = eval_blocks(denoise_blocks(m, nb, dev), gb)
            res.setdefault("cams", {})[cam] = dict(psnr=p, ssim=s, n_images=len(items), n_blocks=len(nb))
            print(f"{name} on {cam} test scenes ({len(items)} images, {len(nb)} blocks): PSNR/SSIM %.3f / %.4f" % (p, s), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)

if __name__ == "__main__":
    main()
