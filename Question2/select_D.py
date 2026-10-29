"""Choose Model D between the two candidates using the VALIDATION split only (never test).

Prerequisite: train D_p8 and D_dil2 with seed 0 (python train.py --model D_p8 --seed 0, ...).
Pre-declared rule (fixed before looking at results): pick the candidate with the lower mean validation
E(2) over speeds 1x, 2x, 3x for the seed-0 model. The winner's seed-0 run is reused as D seed 0
(identical protocol); train seed 1 with `python train.py --model D --seed 1`.
"""
import json
import os
import shutil

import torch

import common as c
import evaluate as ev

device = "cuda" if torch.cuda.is_available() else "cpu"
_, val, _ = c.split_videos(c.load_moving_mnist())

res = {}
for name in ("D_p8", "D_dil2"):
    m = c.load_model(name, 0, device)
    e2 = {k: ev.horizon_mse(m, val, device, k)[:, :2].mean().item() for k in (1, 2, 3)}
    e5 = ev.horizon_mse(m, val, device, 1).mean().item()
    res[name] = dict(params=c.count_params(m), reach_px=c.reach_px(m), val_E5_1x=e5,
                     val_E2={f"{k}x": v for k, v in e2.items()}, score=sum(e2.values()) / 3)
    print(name, json.dumps(res[name]))

chosen = min(res, key=lambda n: res[n]["score"])
print("chosen (validation only):", chosen)
os.makedirs(c.RUNS_DIR, exist_ok=True)
json.dump({"chosen": chosen, "candidates": res}, open(os.path.join(c.RUNS_DIR, "D_choice.json"), "w"), indent=1)
dst = c.run_dir("D", 0)
if not os.path.exists(dst):
    shutil.copytree(c.run_dir(chosen, 0), dst)
print(f"seed-0 run copied to {dst}. Now: python train.py --model D --seed 1")
