"""Final stage: Table 5 (A-D), Table 6 (symmetrised B, D [, DB]), Figure 6.

Requires runs for A, B, C, D with seeds 0 and 1, results/selection.json from stage_A2.py.
    python stage_final.py            # final model = symmetrised D
    python stage_final.py --final DB # if you trained D+B (python train.py --model DB ...) and Table 5 supports it
"""
import argparse
import json
import os

import torch

import common as c
import evaluate as ev
import viz

ap = argparse.ArgumentParser()
ap.add_argument("--final", default="D", choices=["D", "DB"])
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
for d in ("results", "figures", "figures/gifs"):
    os.makedirs(d, exist_ok=True)
_, _, test = c.split_videos(c.load_moving_mnist())

COLS = [("E5", "E(5) 1x"), ("C_M", "C_M"), ("E_M", "E_M"), ("C_R", "C_R"), ("E_R", "E_R"),
        ("E2_1x", "E(2) 1x"), ("E2_2x", "E(2) 2x"), ("E2_3x", "E(2) 3x")]


def gather(name, sym):
    """Metrics for both seeds of a learned model (optionally symmetrised) + params + train time."""
    out = []
    for s in (0, 1):
        m = c.load_model(name, s, device)
        f = c.sym_forecaster(ev.model_fn(m)) if sym else ev.model_fn(m)
        tag = f"{'sym_' if sym else ''}{name}_seed{s}"
        d = dict(ev.cached_metrics(tag, f, test, device))
        log = json.load(open(os.path.join(c.run_dir(name, s), "log.json")))
        d["params"], d["train_sec"] = log["params"], log["mean_epoch_sec"]
        out.append(d)
    return out


def table(rows):
    head = "| Model | " + " | ".join(h for _, h in COLS) + " | Params | Train s/epoch | Inference s |\n"
    head += "|" + "---|" * (len(COLS) + 4)
    lines = [head]
    for label, (m0, m1) in rows:
        cells = [ev.pair(m0[k], m1[k]) for k, _ in COLS]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {m0['params']:,} | "
                     + ev.pair(m0["train_sec"], m1["train_sec"]) + " | " + ev.pair(m0["inference_sec"], m1["inference_sec"]) + " |")
    return "\n".join(lines)


plain = {n: gather(n, False) for n in ("A", "B", "C", "D")}
t5 = table([(n, plain[n]) for n in plain])
print("Table 5\n", t5)
open("results/table5.md", "w").write(t5 + f"\n\nHardware: {c.hardware()}\n")

sym_rows = {"B": gather("B", True), "D": gather("D", True)}
rows6 = [("sym B", sym_rows["B"]), ("sym D", sym_rows["D"])]
if args.final == "DB":
    sym_rows["DB"] = gather("DB", True)
    rows6.append(("sym D+B (final)", sym_rows["DB"]))
t6 = table(rows6)
print("Table 6 (train time = underlying learned model only; inference includes 8 transforms/passes)\n", t6)
open("results/table6.md", "w").write(t6 + "\n")

# ---------------- Figure 6 with the final model (seed 0) on exactly the Figure 3 / Figure 5 videos
final = c.sym_forecaster(ev.model_fn(c.load_model(args.final, 0, device)))
sel = json.load(open("results/selection.json"))
Ts = {"M": c.M, "R": c.R}
with torch.no_grad():
    for p in sel["fig3"]:
        i, T = p["test_index"], Ts[p["T"]]
        v = c.to_float(test[i:i + 1]).to(device)
        X, Y, _ = c.get_window_batch(v, 1, 1)
        fx, fTX = final(X), final(T(X))
        tag = p["png"].replace("fig3_", "fig6a_").replace("figures/", "").replace(".png", "")
        viz.symmetry_panel(X[0], fx[0], T(X)[0], T(Y)[0], fTX[0], T(fx)[0], f"figures/{tag}.png",
                           f"figures/gifs/{tag}.gif", title=f"final model, T={p['T']}, video {9000+i}")
    pick = sel["fig5"]
    v = c.to_float(test[pick:pick + 1]).to(device)
    rows, labels = [], []
    for k in (1, 2, 3):
        X, Y, _ = c.get_window_batch(v, k, 1)
        P = final(X)
        rows.append([X[0, j, 0] for j in range(5)] + [Y[0, 0, 0], Y[0, 1, 0]])
        rows.append([None] * 5 + [P[0, 0, 0], P[0, 1, 0]])
        labels += [f"{k}x input → truth", f"{k}x prediction"]
fig = viz.grid(rows, labels, [f"in {j+1}" for j in range(5)] + ["h1", "h2"])
fig.savefig("figures/fig6b.png", dpi=200)
print("saved Tables 5-6 and Figure 6 (figures/fig6a_*.png, figures/gifs/fig6a_*.gif, figures/fig6b.png)")
