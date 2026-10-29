"""Figure helpers: frame grids, animated GIF panels, and a digit-overlap heuristic."""
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


def _img(x):
    return None if x is None else x.detach().cpu().numpy().reshape(64, 64)


def grid(rows, row_labels, col_titles=None, figsize_per=1.0):
    """rows: list of lists of (64x64 tensor | None). Returns a matplotlib figure."""
    nr, nc = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * figsize_per, nr * figsize_per + 0.3), squeeze=False)
    for i, r in enumerate(rows):
        for j in range(nc):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(r) and r[j] is not None:
                ax.imshow(_img(r[j]), cmap="gray", vmin=0, vmax=1)
            if j == 0:
                ax.text(-0.08, 0.5, row_labels[i], transform=ax.transAxes, ha="right", va="center", fontsize=8)
            if i == 0 and col_titles:
                ax.set_title(col_titles[j], fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.03)
    return fig


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def save_gif(frames, path, ms=700):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)


def panel_rows(X, futures, labels):
    """X: (5,1,64,64) shown in columns 1-5 of every row; futures: list of (5,1,64,64) shown in columns
    6-10 (one per row). Returns rows suitable for grid()."""
    rows = []
    for fut in futures:
        rows.append([X[j, 0] for j in range(5)] + [fut[j, 0] for j in range(5)])
    return rows


def symmetry_panel(X, fx, TX, TY, fTX, TfX, png_path, gif_path, title=None):
    """Figure 3 / 6(a) panel: 4 rows x 10 columns, plus a GIF stepping through the 5 horizons
    (inputs kept, one future column per frame). Rows: (1) X|f(X) (2) T(X)|T(Y) (3) T(X)|f(T(X))
    (4) T(X)|T(f(X))."""
    labels = ["X → f(X)", "T(X) → T(Y)", "T(X) → f(T(X))", "T(X) → T(f(X))"]
    ins = [X, TX, TX, TX]
    futs = [fx, TY, fTX, TfX]
    rows = [[i[j, 0] for j in range(5)] + [f[j, 0] for j in range(5)] for i, f in zip(ins, futs)]
    cols = [f"in {j+1}" for j in range(5)] + [f"h{j+1}" for j in range(5)]
    fig = grid(rows, labels, cols)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    frames = []
    for h in range(5):
        rr = [[i[j, 0] for j in range(5)] + [f[h, 0]] for i, f in zip(ins, futs)]
        frames.append(fig_to_pil(grid(rr, labels, cols[:5] + [f"h{h+1}"])))
    save_gif(frames, gif_path)


def n_digit_blobs(frame, thr=0.3, min_area=12):
    """Number of connected blobs above thr (two separate digits -> 2; merged digits -> 1)."""
    lab, n = ndimage.label(frame > thr)
    if n == 0:
        return 0
    areas = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    return int((np.asarray(areas) >= min_area).sum())


def overlaps(video_u8, frames_1idx):
    """True if any of the given 1-indexed frames shows merged/overlapping digits (heuristic: fewer than
    two blobs). video_u8: (20,1,64,64) uint8 tensor."""
    return any(n_digit_blobs(video_u8[f - 1, 0].numpy().astype(np.float32) / 255.0) < 2
               for f in frames_1idx)
