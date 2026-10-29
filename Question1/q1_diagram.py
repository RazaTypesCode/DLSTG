"""Draws the Q1 architecture diagram -> figures/q1_architecture.png  (python q1_diagram.py)

Layout: data sources -> as-of snapshot -> node encoders -> river graph (lag attention) -> fuse -> decoder -> output.
The announced future releases F reach the decoder along a separate dashed route (known-future path).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(15, 7.6))
ax.set_xlim(0, 15); ax.set_ylim(0, 7.6); ax.axis("off")

COL = dict(inp="#e8eef7", proc="#dff0ea", graph="#fdf1dc", out="#f9e3e3", panel="#fffaf0")
EDGE = "#33415c"


def box(x, y, w, h, text, color, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=EDGE, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(p0, p1, label=None, color=EDGE, ls="-", shrink=0, lw=1.3):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls,
                                shrinkA=shrink, shrinkB=shrink))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.12, label, fontsize=7,
                ha="center", color=color)


# ---------------------------------------------------------------- column titles
for x, t in ((1.4, "Data known at issuance t"), (4.25, "Alignment"), (6.9, "Encoding"),
             (10.25, "River graph"), (13.8, "Forecast")):
    ax.text(x, 7.3, t, ha="center", fontsize=10, fontweight="bold")

# ---------------------------------------------------------------- 1. data sources
inputs = [
    ("Water levels S1–S4, T1\nW: [B, L, 5]   (≤ t)", 6.0),
    ("Rain gauges R1, R2\nP: [B, L, 2]   (≤ t)", 4.9),
    ("Actual reservoir release\nU: [B, L, 1]   (≤ t)", 3.8),
    ("Announced release schedule\nF: [B, 6, d_f]   (announced ≤ t)", 2.7),
    ("Static river data G\nlocations, distances, adjacency", 1.6),
]
for text, y in inputs:
    box(0.1, y, 2.6, 0.9, text, COL["inp"], fs=8)
ax.text(1.4, 0.95, "Not available at t: future rain, future levels\n(targets appear only in training)", ha="center",
        fontsize=7.5, style="italic", color="#8a1c1c")

# ---------------------------------------------------------------- 2. as-of snapshot
box(3.3, 1.2, 1.9, 5.7,
    "As-of-t snapshot\n\n• records with arrival ≤ t\n• event time → 10-min grid\n• mask M, age a\n• anchor w_i(t) =\n  latest valid level\n\nx: [B, L, 6 nodes, d_x]\nnodes: R0, S1–S4, T1",
    COL["proc"], fs=8)
for _, y in inputs:
    arrow((2.7, y + 0.45), (3.3, y + 0.45))

# ---------------------------------------------------------------- 3. temporal encoder
box(5.9, 3.0, 2.0, 1.6, "Per-node temporal\nencoder (shared GRU\n+ node embedding e_i)\n\nh_{i,s}: [B, L, 6, 32]", COL["proc"], fs=8)
arrow((5.2, 3.8), (5.9, 3.8))

# ---------------------------------------------------------------- 4. river graph panel
ax.add_patch(FancyBboxPatch((8.4, 2.0), 3.9, 4.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc=COL["panel"], ec=EDGE, lw=1, ls="--"))
ax.text(10.35, 6.45, "lag attention along directed edges", ha="center", fontsize=8, style="italic")
arrow((7.9, 3.8), (8.4, 3.8), "h")
pos = {"R0": (9.0, 5.7), "S1": (9.7, 4.95), "S2": (10.4, 4.2), "S3": (11.1, 3.45), "S4": (11.8, 2.7),
       "T1": (9.0, 3.4), "latent": (11.2, 5.4)}
for n, (x, y) in pos.items():
    fc = "#fbe9a8" if n in ("S1", "S2", "S3", "S4") else ("#f4c7c3" if n == "latent" else "#d9d9d9")
    ax.add_patch(plt.Circle((x, y), 0.27, fc=fc, ec=EDGE, zorder=3))
    ax.text(x, y, n, ha="center", va="center", fontsize=7 if n != "latent" else 5.5, zorder=4)
for a, b in (("R0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("T1", "S2")):
    arrow(pos[a], pos[b], shrink=13)
arrow(pos["latent"], pos["S3"], color="#b03a2e", ls="--", shrink=13)
ax.text(11.2, 5.95, "unmonitored tributary:\nlatent inflow only", ha="center", fontsize=6.5, color="#b03a2e")

box(8.4, 0.75, 3.9, 0.95,
    r"$m_i=\sum_{j}\sum_{\lambda}\alpha_{ji}(\lambda)\,W[h_{j,t-\lambda};\psi_{ji}]$" "\n"
    "λ ∈ [λmin, λmax] from distance / celerity range;\nα adapts to the current flow state",
    COL["graph"], fs=7.5)

# ---------------------------------------------------------------- 5. fuse -> decoder -> output
box(12.8, 1.2, 2.0, 1.4, "Fuse\n" r"$z_i=\mathrm{Fuse}(h_i,m_i,r_i,e_i)$", COL["graph"], fs=8)
box(12.8, 3.4, 2.0, 1.8, "Decoder\n" r"$\hat y^{\,q}_{i,h}=w_i(t)+\sigma_i\,\delta^{q}$" "\n"
    r"$\delta^{q}=f(z_i,\,F,\,e_i,\,h)$" "\nq ∈ {0.1, 0.5, 0.9}", COL["out"], fs=8)
box(12.8, 5.9, 2.0, 0.9, "Output  Ŷ\n[B, 6, 4, 3]   S1–S4", COL["out"], fs=8, bold=True)
arrow((12.3, 1.9), (12.8, 1.9))                 # graph -> fuse
arrow((13.8, 2.6), (13.8, 3.4), "z")            # fuse -> decoder
arrow((13.8, 5.2), (13.8, 5.9))                 # decoder -> output

# ---------------------------------------------------------------- known-future path for announced schedule F
blue = "#2e6f9e"
ax.plot([1.4, 1.4], [2.7, 0.45], color=blue, ls="--", lw=1.3)
ax.plot([1.4, 12.5], [0.45, 0.45], color=blue, ls="--", lw=1.3)
ax.plot([12.5, 12.5], [0.45, 4.3], color=blue, ls="--", lw=1.3)
arrow((12.5, 4.3), (12.8, 4.3), color=blue, ls="--")
ax.text(6.2, 0.55, "announced future releases F (known at t) go straight to the decoder", fontsize=7.5, color=blue, ha="center")

plt.savefig("figures/q1_architecture.png", dpi=220, bbox_inches="tight")
print("saved figures/q1_architecture.png")
