"""Figure 1: optical-flow statistics of the first 50 training videos (Farneback, fixed settings)."""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_moving_mnist, split_videos

os.makedirs("figures", exist_ok=True)
train, _, _ = split_videos(load_moving_mnist())
vids = train[:50, :, 0].numpy()  # (50, 20, 64, 64) uint8 (== float*255)

theta, speed = [], []
for v in vids:
    for s in range(19):
        prev, nxt = np.ascontiguousarray(v[s]), np.ascontiguousarray(v[s + 1])
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(dx ** 2 + dy ** 2)
        keep = (prev > 51) & (nxt > 51) & (mag > 0.5)       # intensity > 0.2 in both frames
        theta.append(np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360)
        speed.append(mag[keep])
theta, speed = np.concatenate(theta), np.concatenate(speed)
n, med = len(speed), float(np.median(speed))
print(f"pixels kept: {n}, median speed: {med:.3f} px/frame")
print("speed percentiles 10/25/50/75/90/99:", np.percentile(speed, [10, 25, 50, 75, 90, 99]).round(2))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
w = np.full(n, 1.0 / n)
ax[0].hist(theta, bins=18, range=(0, 360), weights=w, color="#4a6fa5", edgecolor="white")
ax[0].axhline(1 / 18, color="k", ls="--", lw=1, label="uniform (1/18)")
ax[0].set_xlabel("flow direction θ (deg; 0° = right, 90° = up)"); ax[0].set_ylabel("fraction of kept pixels")
ax[0].set_xticks(range(0, 361, 60)); ax[0].legend(); ax[0].set_title("(a) direction")
ax[1].hist(speed, bins=50, weights=np.full(n, 1.0 / n), color="#4a6fa5", edgecolor="white")
ax[1].axvline(med, color="crimson", lw=1.5, label=f"median = {med:.2f} px/frame"); ax[1].legend()
ax[1].set_xlabel("flow speed (pixels / frame)"); ax[1].set_ylabel("fraction of kept pixels"); ax[1].set_title("(b) speed")
plt.tight_layout(); plt.savefig("figures/fig1.png", dpi=200)
print(f"CAPTION: median speed = {med:.3g} px/frame; {n} pixels kept from 950 frame pairs.")
