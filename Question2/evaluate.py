"""Metrics shared by all stages. A "forecaster" f is any callable (B,5,1,64,64) -> (B,5,1,64,64)."""
import json
import os

import torch

import common as c


def per_video_mse(a, b):
    return ((a - b) ** 2).mean(dim=(1, 2, 3, 4))  # (N,)


@torch.no_grad()
def transform_metrics(f, videos, T, device, bs=32):
    """Per-video consistency gap C_T = MSE(f(T X), T f(X)) and transformed error
    E_T = MSE(f(T X), T Y); 1x windows, t=1, five horizons."""
    C, E = [], []
    for i in range(0, len(videos), bs):
        v = c.to_float(videos[i:i + bs]).to(device)
        X, Y, _ = c.get_window_batch(v, 1, 1)
        fx, ftx = f(X), f(T(X))
        C.append(per_video_mse(ftx, T(fx)).cpu())
        E.append(per_video_mse(ftx, T(Y)).cpu())
    return torch.cat(C), torch.cat(E)


def horizon_mse(f, videos, device, k=1, t=1):
    """(N,5) per-video MSE per forecast horizon (only meaningful where the target exists)."""
    p, y, _ = c.predict_all(f, videos, device, k=k, t=t)
    return c.per_horizon_mse(p, y)


def full_metrics(f, videos, device, time_it=True):
    """Everything Tables 3-6 need for one forecaster."""
    d = {}
    h1 = horizon_mse(f, videos, device, 1)
    d["mse_h"] = h1.mean(0).tolist()
    d["E5"] = h1.mean().item()
    for name, T in (("M", c.M), ("R", c.R)):
        C, E = transform_metrics(f, videos, T, device)
        d["C_" + name], d["E_" + name] = C.mean().item(), E.mean().item()
    for k in (1, 2, 3):
        hk = h1 if k == 1 else horizon_mse(f, videos, device, k)
        d[f"E2_{k}x"] = hk[:, :2].mean().item()      # first two horizons only: valid at every speed
        if k == 2:
            d["E5_2x"] = hk.mean().item()
    if time_it:
        d["inference_sec"] = c.inference_time(f, videos, device)
    return d


def cached_metrics(tag, f, videos, device, time_it=True):
    path = os.path.join("results", "metrics", f"{tag}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = full_metrics(f, videos, device, time_it)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1)
    return d


def pair(a, b):
    """'mean (seed0, seed1)' with 3 s.f. / scientific below 1e-3."""
    return f"{c.sci((a + b) / 2)} ({c.sci(a)}, {c.sci(b)})"


def reference_fns():
    return {"Blank": c.blank_predictor, "Copy-last": c.copy_last_predictor}
