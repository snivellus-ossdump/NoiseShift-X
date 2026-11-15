"""One command per job. Safe to re-run after an interruption (training resumes, evaluation repeats).

  python run_fold.py all     # seen-camera reference models: base_all + aug_all (+ official-block eval)
  python run_fold.py S6      # held-out-camera models base_hold_S6 + aug_hold_S6, evaluated on S6 test scenes
"""
import argparse, os, subprocess, sys
import config as C

# Run a command and stop immediately if it fails (check=True).
def run(cmd):
    print(">>", " ".join(cmd), flush=True); subprocess.run(cmd, check=True)

ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--epochs", type=int, default=C.EPOCHS)
a = ap.parse_args(); py = sys.executable
ckpt = lambda n: os.path.join(C.CKPT_DIR, n + ".pt")
# 'all': the seen-camera reference models (trained on every camera's train scenes),
# evaluated on all five cameras and on the official validation blocks.
if a.target == "all":
    for name, extra in [("base_all", []), ("aug_all", ["--aug"])]:
        run([py, "-u", "train.py", "--name", name, "--epochs", str(a.epochs)] + extra)
        run([py, "-u", "evaluate.py", "--ckpt", ckpt(name), "--cams"] + C.CAMERAS + ["--official"])
else:
    # A camera code: train baseline and augmented models WITHOUT that camera, then test on it.
    assert a.target in C.CAMERAS, f"camera must be one of {C.CAMERAS} or 'all'"
    for pre, extra in [("base", []), ("aug", ["--aug"])]:
        name = f"{pre}_hold_{a.target}"
        run([py, "-u", "train.py", "--name", name, "--hold", a.target, "--epochs", str(a.epochs)] + extra)
        run([py, "-u", "evaluate.py", "--ckpt", ckpt(name), "--cams", a.target])
