"""Stage A2: Tables 3-4 and Figures 3-5 for model A (+ Blank, Copy-last).

ONLY run after Q2.5 has been written and committed (this is the first stress test of the baseline).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import common as c
import evaluate as ev
import viz

device = "cuda" if torch.cuda.is_available() else "cpu"
for d in ("results", "figures", "figures/gifs"):
    os.makedirs(d, exist_ok=True)
_, _, test = c.split_videos(c.load_moving_mnist())

models = {s: c.load_model("A", s, device) for s in (0, 1)}
met = {s: ev.cached_metrics(f"A_seed{s}", models[s], test, device) for s in (0, 1)}
ref = {n: ev.cached_metrics(f"{n}", f, test, device, time_it=False) for n, f in ev.reference_fns().items()}

# ---------------- Table 3
hdr = "| Model | E(5) | C_M | E_M | C_R | E_R |\n|---|---|---|---|---|---|"
row = "| A (mean (s0, s1)) | " + " | ".join(ev.pair(met[0][k], met[1][k]) for k in ("E5", "C_M", "E_M", "C_R", "E_R")) + " |"
rr = [f"| {n} | " + " | ".join(c.sci(m[k]) for k in ("E5", "C_M", "E_M", "C_R", "E_R")) + " |" for n, m in ref.items()]
t3 = "\n".join([hdr, row] + rr)
# ---------------- Table 4
hdr4 = "| Model | E(2) 1x | E(2) 2x | E(2) 3x | E(5) 1x | E(5) 2x |\n|---|---|---|---|---|---|"
row4 = "| A (mean (s0, s1)) | " + " | ".join(ev.pair(met[0][k], met[1][k]) for k in ("E2_1x", "E2_2x", "E2_3x", "E5", "E5_2x")) + " |"
rr4 = [f"| {n} | " + " | ".join(c.sci(m[k]) for k in ("E2_1x", "E2_2x", "E2_3x", "E5", "E5_2x")) + " |" for n, m in ref.items()]
t4 = "\n".join([hdr4, row4] + rr4)
print(t3, "\n\n", t4)
open("results/table3.md", "w").write(t3 + "\n"); open("results/table4.md", "w").write(t4 + "\n")

# ---------------- Figure 3 / 4 : seed-0 model, per-video consistency gaps
f0 = models[0]
CM, _ = ev.transform_metrics(f0, test, c.M, device)
CR, _ = ev.transform_metrics(f0, test, c.R, device)
top = {"M": CM.topk(2).indices.tolist(), "R": CR.topk(2).indices.tolist()}
print("top-2 per-video C_M:", top["M"], " top-2 per-video C_R:", top["R"])

Ts = {"M": c.M, "R": c.R}
panels = []
with torch.no_grad():
    for name in ("M", "R"):
        for rank, i in enumerate(top[name]):
            v = c.to_float(test[i:i + 1]).to(device)
            X, Y, _ = c.get_window_batch(v, 1, 1)
            T = Ts[name]
            fx, fTX = f0(X), f0(T(X))
            tag = f"fig3_{name}_rank{rank+1}_video{9000+i}"
            viz.symmetry_panel(X[0], fx[0], T(X)[0], T(Y)[0], fTX[0], T(fx)[0],
                               f"figures/{tag}.png", f"figures/gifs/{tag}.gif",
                               title=f"T={name}, video {9000+i}, C_{name}={CM[i].item() if name=='M' else CR[i].item():.3g}")
            panels.append({"T": name, "rank": rank + 1, "test_index": i, "png": f"figures/{tag}.png",
                           "gif": f"figures/gifs/{tag}.gif"})
print("Figure 3 panels:", json.dumps(panels, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, (name, C) in zip(ax, (("M", CM), ("R", CR))):
    a.hist(C.numpy(), bins=50, color="#4a6fa5")
    for i in top[name]:
        a.axvline(C[i].item(), color="crimson", lw=1)
    a.set_xlabel(f"per-video C_{name} (model A, seed 0)"); a.set_ylabel("videos"); a.set_yscale("log")
plt.tight_layout(); plt.savefig("figures/fig4.png", dpi=200)

# ---------------- Figure 5: one video, no overlapping digits in any displayed true frame
shown = {1: [1, 3, 5, 7, 9, 6, 7], 2: [1, 3, 5, 7, 9, 11, 13], 3: [1, 4, 7, 10, 13, 16, 19]}  # 1-idx
pick = None
for i in range(len(test)):
    if not any(viz.overlaps(test[i], fr) for fr in shown.values()):
        pick = i
        break
print("Figure 5 video:", pick)
rows, labels = [], []
with torch.no_grad():
    v = c.to_float(test[pick:pick + 1]).to(device)
    for k in (1, 2, 3):
        X, Y, _ = c.get_window_batch(v, k, 1)
        P = f0(X)
        rows.append([X[0, j, 0] for j in range(5)] + [Y[0, 0, 0], Y[0, 1, 0]])
        rows.append([None] * 5 + [P[0, 0, 0], P[0, 1, 0]])
        labels += [f"{k}x input → truth", f"{k}x prediction"]
fig = viz.grid(rows, labels, [f"in {j+1}" for j in range(5)] + ["h1", "h2"])
fig.savefig("figures/fig5.png", dpi=200)
json.dump({"fig2": json.load(open("results/selection_fig2.json"))["fig2"], "fig3": panels, "fig5": pick},
          open("results/selection.json", "w"), indent=1)
print("saved. Selection stored in results/selection.json (reused for Figure 6).")
