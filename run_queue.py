"""Run every remaining experiment in priority order. Safe to stop (Ctrl+C) and restart any time:
finished jobs are skipped, interrupted training resumes from its last epoch.

  python run_queue.py            # run everything remaining
  python run_queue.py --list     # show the queue and what is already done
  python run_queue.py --phase gauss seeds    # only some phases

Phases (priority order):
  gauss     Gaussian-only (signal-independent) control on IP, GP, N6, G4        (S6 already done)
  seeds     seeds 1 and 2 of base/aug held-out models for IP, GP, N6, G4         (S6 seeds already done)
  match     size-matched controls: all cameras but the same number of train scenes as the hold-K model
  allseeds  seeds 1 and 2 of base_all / aug_all (seen-camera references)
  gaussseeds  (optional, ~3 h) seeds 1 and 2 of the Gaussian-only control on all 5 cameras
"""
import argparse, json, os, subprocess, sys
import config as C

py = sys.executable
OTHER = [c for c in C.CAMERAS if c != "S6"]

# A job is finished when its results JSON already contains the requested entries.
def done(name, cams, official=False):
    p = os.path.join(C.RESULTS_DIR, name + ".json")
    if not os.path.exists(p): return False
    r = json.load(open(p))
    return all(c in r.get("cams", {}) for c in cams) and (not official or "official_val" in r)

# One queue entry: training arguments plus the cameras to evaluate afterwards.
def job(name, train_args, cams, official=False):
    return dict(name=name, train=["--name", name, "--epochs", str(C.EPOCHS)] + train_args, cams=cams, official=official)

# Build the list of jobs for the requested phases (see the module docstring).
def queue(phases):
    q = []
    if "gauss" in phases:
        for k in OTHER: q.append(job(f"abl_gauss_hold_{k}", ["--hold", k, "--aug", "--aug_gauss"], [k]))
    if "seeds" in phases:
        for s in (1, 2):
            for k in OTHER:
                q.append(job(f"base_hold_{k}_s{s}", ["--hold", k, "--seed", str(s)], [k]))
                q.append(job(f"aug_hold_{k}_s{s}", ["--hold", k, "--aug", "--seed", str(s)], [k]))
    if "match" in phases:
        for k in C.CAMERAS:
            q.append(job(f"base_match_{k}", ["--match", k], [k]))
            q.append(job(f"aug_match_{k}", ["--match", k, "--aug"], [k]))
    if "gaussseeds" in phases:
        for s in (1, 2):
            for k in C.CAMERAS: q.append(job(f"abl_gauss_hold_{k}_s{s}", ["--hold", k, "--aug", "--aug_gauss", "--seed", str(s)], [k]))
    if "allseeds" in phases:
        for s in (1, 2):
            q.append(job(f"base_all_s{s}", ["--seed", str(s)], C.CAMERAS, True))
            q.append(job(f"aug_all_s{s}", ["--aug", "--seed", str(s)], C.CAMERAS, True))
    return q

def run(cmd):
    print(">>", " ".join(cmd), flush=True); subprocess.run(cmd, check=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true")
    ap.add_argument("--phase", nargs="*", default=["gauss", "seeds", "match", "allseeds"])   # "gaussseeds" must be requested explicitly
    a = ap.parse_args(); q = queue(a.phase)
    # Skip finished jobs: the queue can be stopped and restarted at any time.
    todo = [j for j in q if not done(j["name"], j["cams"], j["official"])]
    print(f"{len(q) - len(todo)} of {len(q)} jobs already done, {len(todo)} to run", flush=True)
    if a.list:
        for j in q: print(("DONE " if j not in todo else "todo ") + j["name"])
        sys.exit(0)
    # Run each remaining job: first train, then evaluate.
    for i, j in enumerate(todo, 1):
        print(f"=== job {i}/{len(todo)}: {j['name']}", flush=True)
        run([py, "-u", "train.py"] + j["train"])
        run([py, "-u", "evaluate.py", "--ckpt", os.path.join(C.CKPT_DIR, j["name"] + ".pt"), "--cams"] + j["cams"] + (["--official"] if j["official"] else []))
    print("ALL DONE", flush=True)
