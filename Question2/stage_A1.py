"""Stage A1 (after training A, seeds 0 and 1): Table 2 and Figure 2.

Run this BEFORE the mirror/rotation/speed stress tests (stage_A2.py). It only uses 1x, t=1 forecasts.
Write and commit your Q2.5 prediction before running stage_A2.py.
"""
import json
import os

import torch

import common as c
import evaluate as ev
import viz

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
_, _, test = c.split_videos(c.load_moving_mnist())

models = {s: c.load_model("A", s, device) for s in (0, 1)}
mse = {s: ev.horizon_mse(models[s], test, device).mean(0) for s in (0, 1)}           # (5,)
refs = {n: ev.horizon_mse(f, test, device).mean(0) for n, f in ev.reference_fns().items()}
params = c.count_params(models[0])

lines = ["| Model | MSE_1 | MSE_2 | MSE_3 | MSE_4 | MSE_5 | E(5) | Params |", "|---|---|---|---|---|---|---|---|"]
cells = [ev.pair(mse[0][h].item(), mse[1][h].item()) for h in range(5)]
lines.append(f"| A (mean (seed 0, seed 1)) | " + " | ".join(cells) + " | "
             + ev.pair(mse[0].mean().item(), mse[1].mean().item()) + f" | {params:,} |")
for n, m in refs.items():
    lines.append(f"| {n} | " + " | ".join(c.sci(m[h].item()) for h in range(5)) + f" | {c.sci(m.mean().item())} | 0 |")
table = "\n".join(lines)
print(table)
open("results/table2.md", "w").write(table + f"\n\nHardware: {c.hardware()}\n")

# ---- Figure 2: three test videos, at least one with overlapping digits (blob heuristic, see viz.py)
ov, no = [], []
for i in range(len(test)):
    v = test[i]
    (ov if viz.overlaps(v, range(6, 11)) else no).append(i)
    if len(ov) >= 1 and len(no) >= 2:
        break
sel = [ov[0], no[0], no[1]]
print("Figure 2 videos (test-relative indices; global = +9000):", sel, "(first has overlapping digits)")

rows, labels = [], []
f = models[0]
with torch.no_grad():
    for i in sel:
        v = c.to_float(test[i:i + 1]).to(device)
        X, Y, _ = c.get_window_batch(v, 1, 1)
        P = f(X)
        rows.append([X[0, j, 0] for j in range(5)] + [Y[0, j, 0] for j in range(5)])
        rows.append([None] * 5 + [P[0, j, 0] for j in range(5)])
        labels += [f"video {9000+i}\ninput → truth", "prediction"]
fig = viz.grid(rows, labels, [f"in {j+1}" for j in range(5)] + [f"h{j+1}" for j in range(5)])
fig.savefig("figures/fig2.png", dpi=200)
json.dump({"fig2": sel}, open("results/selection_fig2.json", "w"))
print("saved figures/fig2.png, results/table2.md")
