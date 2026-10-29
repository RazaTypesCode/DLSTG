"""Draws the Q1 architecture diagram -> figures/q1_architecture.png  (python q1_diagram.py)"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")

C = dict(inp="#e8eef7", proc="#dff0ea", graph="#fdf1dc", out="#f9e3e3", note="#f2f2f2")


def box(x, y, w, h, text, color, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#33415c", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(x0, y0, x1, y1, label=None, color="#33415c", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, linestyle=ls))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, fontsize=6.5, ha="center", color=color)


# ---- column 1: data sources (what is known at issuance time t)
ax.text(1.25, 7.0, "Data available at issuance t", ha="center", fontsize=9, fontweight="bold")
src = [
    ("Water levels S1–S4, T1\nW: [B, L, 5]  (10-min, ≤ t)", 5.85),
    ("Rain gauges R1, R2\nP: [B, L, 2]  (10-min sums, ≤ t)", 4.75),
    ("Actual reservoir release\nU: [B, L, 1]  (≤ t)", 3.65),
    ("Announced release schedule\nF: [B, H=6, d_f]  (announced ≤ t,\neffective t+10 … t+60 min)", 2.45),
    ("Static river data G\nlocations, order, distances\nadjacency [4, d_s]", 1.2),
]
for text, y in src:
    box(0.1, y, 2.3, 0.85 if "Announced" not in text else 1.05, text, C["inp"], fs=7)
ax.text(1.25, 0.55, "Not available at t: future rain,\nfuture levels (targets only in training)", ha="center",
        fontsize=7, style="italic", color="#8a1c1c")

# ---- column 2: as-of alignment
box(3.0, 1.2, 1.9, 4.9, "As-of-t snapshot\n\n• keep records with\n  arrival time ≤ t\n• event-time → 10-min grid\n• mask M ∈ {0,1}\n• age a (min since\n  last valid value)\n• anchor w_i(t) =\n  latest valid level\n\n→ x: [B, L, 6 nodes, d_x]\n(R0, S1, S2, S3, S4, T1)",
    C["proc"], fs=7.5)
for _, y in src[:3]:
    arrow(2.4, y + 0.4, 3.0, 3.65)
arrow(2.4, 3.0, 3.0, 3.0); arrow(2.4, 1.6, 3.0, 2.0)

# ---- column 3: temporal encoders
box(5.4, 2.6, 1.9, 2.0, "Per-node temporal\nencoder (GRU, shared\nweights + station\nembedding e_i)\n\nh_{i,s}: [B, L, 6, d=32]\nh_i = h_{i,t}", C["proc"], fs=7.5)
arrow(4.9, 3.65, 5.4, 3.65, "x")

# ---- column 4: river graph
ax.text(9.0, 6.55, "Directed river graph (downstream flow)", ha="center", fontsize=9, fontweight="bold")
pos = {"R0": (7.9, 5.5), "S1": (8.7, 4.6), "S2": (9.5, 3.7), "S3": (10.3, 2.8), "S4": (11.1, 1.9),
       "T1": (8.0, 3.0), "Q": (10.4, 4.3)}
for n, (x, y) in pos.items():
    fc = "#fbe9a8" if n in ("S1", "S2", "S3", "S4") else ("#d9d9d9" if n != "Q" else "#f4c7c3")
    ax.add_patch(plt.Circle((x, y), 0.28, fc=fc, ec="#33415c"))
    ax.text(x, y, n if n != "Q" else "latent", ha="center", va="center", fontsize=7 if n != "Q" else 5.5)
for a, b in (("R0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("T1", "S2")):
    arrow(pos[a][0] + 0.2, pos[a][1] - 0.2, pos[b][0] - 0.2, pos[b][1] + 0.2)
arrow(pos["Q"][0] - 0.15, pos["Q"][1] - 0.25, pos["S3"][0] + 0.05, pos["S3"][1] + 0.3, color="#b03a2e", ls="--")
ax.text(11.45, 4.35, "unmonitored\ntributary:\nlatent inflow\nu_3 (not\nobserved)", fontsize=6.5, color="#b03a2e")
box(7.5, 0.45, 3.3, 0.95, "Message m_i = Σ_j Σ_λ α_ji(λ) W[h_{j,t−λ}; ψ_ji]\nlag window λ ∈ [λ_min, λ_max] from distance / celerity range\nα adapts to current flow ⇒ time-varying travel time", C["graph"], fs=6.8)
arrow(7.3, 3.6, 7.6, 2.5)
arrow(7.3, 3.9, 7.7, 4.9, "h_j,s")
box(11.3, 0.45, 1.55, 0.95, "fuse: z_i =\nFuse(h_i, m_i,\nrain r_i, release)", C["graph"], fs=6.8)
arrow(10.8, 0.9, 11.3, 0.9)

# ---- output
box(11.5, 5.0, 1.4, 1.5, "Decoder\nŷ_{i,h}^q = w_i(t) + σ_i·δ(z_i, F_h, e_i, h)\nq ∈ {0.1, 0.5, 0.9}", C["out"], fs=6.3)
arrow(12.05, 1.4, 12.05, 5.0, "z_i")
arrow(2.4, 3.0, 11.5, 5.55, "scheduled releases F (known future)", color="#2e6f9e", ls="--")
box(11.5, 6.65, 1.4, 0.45, "Ŷ: [B, 6, 4, 3]  S1–S4", C["out"], fs=7, bold=True)
arrow(12.2, 6.5, 12.2, 6.65)
plt.savefig("figures/q1_architecture.png", dpi=220, bbox_inches="tight")
print("saved figures/q1_architecture.png")
