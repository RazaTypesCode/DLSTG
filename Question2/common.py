"""Shared code for Question 2 (Motion or Memory?).

Conventions
-----------
* Videos are uint8 tensors (N, 20, 1, 64, 64); converted to float in [0, 1] per batch.
* Frame numbers in the handout are 1-indexed (1..20); tensor indices are 0-indexed.
* Windows: X = (x_t, x_{t+k}, ..., x_{t+4k}), Y = (x_{t+5k}, ..., x_{t+9k}); only targets with
  frame number <= 20 are "available" (mask = 1).
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TRAIN, N_VAL, N_TEST = 8000, 1000, 1000  # videos 0-7999 / 8000-8999 / 9000-9999


# ----------------------------------------------------------------------------- data
def load_moving_mnist(root=None):
    root = root or os.environ.get("PA2_DATA", "data")
    import torchvision
    ds = torchvision.datasets.MovingMNIST(root=root, split=None, download=True)
    vids = torch.stack([ds[i] for i in range(len(ds))])  # (10000, 20, 1, 64, 64) uint8
    assert vids.shape == (10000, 20, 1, 64, 64), vids.shape
    return vids.to(torch.uint8)


def split_videos(vids):
    return vids[:N_TRAIN], vids[N_TRAIN:N_TRAIN + N_VAL], vids[N_TRAIN + N_VAL:]


def to_float(v):
    return v.float() / 255.0


# ----------------------------------------------------------------------------- symmetries
def M(x):
    """Mirror: (Mx)[i, j] = x[i, 63 - j]."""
    return torch.flip(x, dims=(-1,))


def R(x, a=1):
    """Rotation by a * 90 degrees anticlockwise (a may be negative)."""
    return torch.rot90(x, k=a, dims=(-2, -1))


# G = {R^a, M R^a : a = 0..3}; element g = (a, m): rotate first, then mirror if m.
G_ELEMENTS = [(a, m) for m in (0, 1) for a in range(4)]


def apply_g(x, g):
    a, m = g
    y = R(x, a) if a % 4 else x
    return M(y) if m else y


def apply_g_inv(x, g):
    a, m = g
    y = M(x) if m else x
    return R(y, -a) if a % 4 else y


def apply_g_batch(x, gidx):
    """Apply a per-sample symmetry. x: (B, ...,64,64); gidx: (B,) long indices into G_ELEMENTS."""
    out = torch.empty_like(x)
    for i, g in enumerate(G_ELEMENTS):
        sel = gidx == i
        if sel.any():
            out[sel] = apply_g(x[sel], g)
    return out


# ----------------------------------------------------------------------------- windows
def window_indices(k, t):
    """Return (input frame numbers, target frame numbers) as 1-indexed lists (targets may exceed 20)."""
    inp = [t + k * j for j in range(5)]
    tgt = [t + k * (5 + j) for j in range(5)]
    return inp, tgt


def get_window(video, k=1, t=1):
    """video: (20, 1, 64, 64). Returns X (5,...), Y (5,...; unavailable frames zero-filled), mask (5,)."""
    inp, tgt = window_indices(k, t)
    X = video[[i - 1 for i in inp]]
    mask = torch.tensor([float(f <= 20) for f in tgt])
    Y = torch.zeros_like(X)
    for j, f in enumerate(tgt):
        if f <= 20:
            Y[j] = video[f - 1]
    return X, Y, mask


def get_window_batch(videos, k, t):
    """videos: (B, 20, 1, 64, 64) float; k, t: ints or (B,) long tensors (1-indexed t)."""
    B = videos.shape[0]
    dev = videos.device
    k = torch.as_tensor(k, device=dev).expand(B)
    t = torch.as_tensor(t, device=dev).expand(B)
    j = torch.arange(5, device=dev)
    inp = t[:, None] + k[:, None] * j[None]            # (B, 5) frame numbers
    tgt = t[:, None] + k[:, None] * (5 + j[None])      # (B, 5)
    mask = (tgt <= 20).float()
    b = torch.arange(B, device=dev)[:, None]
    X = videos[b, inp - 1]
    Y = videos[b, (tgt - 1).clamp(max=19)] * mask[:, :, None, None, None]
    return X, Y, mask


def sample_starts(B, speed_aug, gen):
    """Sample (k, t) per training window following the handout rules."""
    if not speed_aug:
        k = torch.ones(B, dtype=torch.long)
        t = torch.randint(1, 12, (B,), generator=gen)            # t in 1..11
    else:
        k = torch.randint(1, 4, (B,), generator=gen)             # k in {1,2,3}
        tmax = 20 - 6 * k                                         # 14, 8, 2
        u = torch.rand(B, generator=gen)
        t = 1 + (u * tmax).long().clamp(max=tmax - 1)
    return k, t


# ----------------------------------------------------------------------------- model
class ConvLSTMCell(nn.Module):
    """Eq. (3)-(4): one conv produces [i, f, o, g]; no peephole terms."""

    def __init__(self, in_ch, hid, ks=3, dil=1):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, ks, stride=1, padding=dil * (ks - 1) // 2,
                              dilation=dil, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Model A by default (ks=(3,3), dil=(1,1), hid=64, patch=4). Model D reuses this class
    with different ks / dil / hid / patch (see experiments)."""

    def __init__(self, hid=64, ks=(3, 3), dil=(1, 1), patch=4):
        super().__init__()
        self.patch, self.hid, self.ks, self.dil = patch, hid, tuple(ks), tuple(dil)
        c_in = patch * patch
        self.l1 = ConvLSTMCell(c_in, hid, ks[0], dil[0])
        self.l2 = ConvLSTMCell(hid, hid, ks[1], dil[1])
        self.out = nn.Conv2d(hid, c_in, 1, bias=True)

    def forward(self, X):
        """X: (B, 5, 1, 64, 64) -> (B, 5, 1, 64, 64) predictions of frames 6..10 (9-step rollout,
        no teacher forcing, gradients flow through fed-back predictions)."""
        B = X.shape[0]
        s = 64 // self.patch
        z = X.new_zeros(B, self.hid, s, s)
        st1, st2 = (z, z.clone()), (z.clone(), z.clone())
        preds = []
        pred = None
        for tau in range(1, 10):
            u = X[:, tau - 1] if tau <= 5 else pred
            u = F.pixel_unshuffle(u, self.patch)
            st1 = self.l1(u, st1)
            st2 = self.l2(st1[0], st2)
            pred = torch.sigmoid(F.pixel_shuffle(self.out(st2[0]), self.patch))
            if tau >= 5:
                preds.append(pred)
        return torch.stack(preds, dim=1)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- baselines
def blank_predictor(X):
    return torch.zeros_like(X)


def copy_last_predictor(X):
    return X[:, -1:].expand(-1, 5, -1, -1, -1).clone()


# ----------------------------------------------------------------------------- loss / metrics
def masked_video_loss(pred, tgt, mask):
    """Squared error averaged over pixels and available frames per video, then over videos."""
    se = ((pred - tgt) ** 2).mean(dim=(2, 3, 4))          # (B, 5)
    per_video = (se * mask).sum(1) / mask.sum(1).clamp(min=1)
    return per_video.mean()


def per_horizon_mse(pred, tgt):
    """(B, 5) MSE per video and horizon, over the 64x64 pixels."""
    return ((pred - tgt) ** 2).mean(dim=(2, 3, 4))


@torch.no_grad()
def predict_all(f, videos, device, k=1, t=1, bs=32, T=None):
    """Run forecaster f (callable on (B,5,1,64,64) float tensors) on every video with window (k, t).
    If T is given, the input is T(X) and the returned target is T(Y).
    Returns (pred, tgt, mask) on CPU."""
    P, Yt, Mk = [], [], []
    for i in range(0, len(videos), bs):
        v = to_float(videos[i:i + bs]).to(device)
        X, Y, mask = get_window_batch(v, k, t)
        if T is not None:
            X, Y = T(X), T(Y)
        P.append(f(X).cpu()); Yt.append(Y.cpu()); Mk.append(mask.cpu())
    return torch.cat(P), torch.cat(Yt), torch.cat(Mk)


def mse_horizons(pred, tgt):
    """Mean over videos of the per-horizon MSE -> (5,) tensor."""
    return per_horizon_mse(pred, tgt).mean(0)


def sym_forecaster(f):
    """Eq. (7): average over g in G of g^{-1}(f(g(X)))."""
    def fbar(X):
        return sum(apply_g_inv(f(apply_g(X, g)), g) for g in G_ELEMENTS) / len(G_ELEMENTS)
    return fbar


def model_fn(model):
    model.eval()
    return lambda X: model(X)


# ----------------------------------------------------------------------------- training
def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def train_model(make_model, train_videos, seed, device, speed_aug=False, sym_aug=False,
                epochs=30, bs=32, lr=1e-3, val_videos=None, log=print, ckpt_path=None):
    """Train one model under the shared protocol. Returns (model, epoch_seconds list, history).
    If ckpt_path is given, state is saved after every epoch and training resumes from it
    (needed for free Colab sessions that disconnect)."""
    torch.manual_seed(seed)
    model = make_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = len(train_videos)
    epoch_times, hist, start_ep = [], [], 1
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"]); epoch_times, hist = ck["epoch_times"], ck["hist"]
        start_ep = ck["epoch"] + 1
        log(f"resumed from epoch {ck['epoch']}")
    if str(device).startswith("cuda"):
        train_videos = train_videos.to(device)  # uint8, ~650 MB; avoids CPU indexing per batch
    for ep in range(start_ep, epochs + 1):
        model.train()
        sync(device); t0 = time.time()
        perm = torch.randperm(n, generator=gen)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            v = to_float(train_videos[idx.to(train_videos.device)]).to(device)
            k, t = sample_starts(len(idx), speed_aug, gen)
            X, Y, mask = get_window_batch(v, k.to(device), t.to(device))
            if sym_aug:
                gi = torch.randint(0, 8, (len(idx),), generator=gen).to(device)
                X, Y = apply_g_batch(X, gi), apply_g_batch(Y, gi)
            loss = masked_video_loss(model(X), Y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx); cnt += len(idx)
        sync(device); epoch_times.append(time.time() - t0)
        rec = {"epoch": ep, "train_loss": tot / cnt, "sec": epoch_times[-1]}
        if val_videos is not None:
            model.eval()
            p, y, _ = predict_all(model, val_videos, device)
            rec["val_E5"] = mse_horizons(p, y).mean().item()
        hist.append(rec)
        log(rec)
        if ckpt_path:
            tmp = ckpt_path + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                        "epoch": ep, "epoch_times": epoch_times, "hist": hist}, tmp)
            os.replace(tmp, ckpt_path)
    return model, epoch_times, hist


# ----------------------------------------------------------------------------- model registry
RUNS_DIR = os.environ.get("PA2_RUNS", "runs")
DATA_DIR = os.environ.get("PA2_DATA", "data")

# A, B, C use the exact baseline. D candidates enlarge one-step reach (see reach_px); the chosen one
# is recorded by select_D.py (validation split only) and trained as "D".
MODEL_SPECS = {
    "A": dict(arch={}, speed=False, sym=False),
    "B": dict(arch={}, speed=False, sym=True),
    "C": dict(arch={}, speed=True, sym=False),
    "D_p8": dict(arch=dict(patch=8), speed=True, sym=False),          # 8x8 patches: 8 px per grid cell
    "D_dil2": dict(arch=dict(dil=(2, 2)), speed=True, sym=False),     # dilation 2: 2 cells per conv
}


def spec_of(name):
    if name in MODEL_SPECS:
        return MODEL_SPECS[name]
    if name in ("D", "DB"):
        with open(os.path.join(RUNS_DIR, "D_choice.json")) as fh:
            base = MODEL_SPECS[json.load(fh)["chosen"]]
        return dict(base, sym=(name == "DB"))
    raise KeyError(name)


def make_model(name):
    return ConvLSTMForecaster(**spec_of(name)["arch"])


def reach_px(model):
    """Longest one-step information path in original pixels (handout convention): a spatial conv
    with kernel ks and dilation d moves information d*(ks-1)/2 folded cells; one folded cell = patch
    original pixels; 1x1 conv / pixel (un)shuffle add nothing. Paths: u->l1->l2 and h1->l1->l2 cross
    both spatial convs, h2->l2 only the second."""
    r = [d * (k - 1) // 2 for k, d in zip(model.ks, model.dil)]
    return max(r[0] + r[1], r[1]) * model.patch


def run_dir(name, seed):
    return os.path.join(RUNS_DIR, f"{name}_seed{seed}")


def load_model(name, seed, device):
    m = make_model(name).to(device)
    m.load_state_dict(torch.load(os.path.join(run_dir(name, seed), "model.pt"), map_location=device))
    return m.eval()


def hardware():
    if torch.cuda.is_available():
        return f"CUDA GPU: {torch.cuda.get_device_name(0)}"
    return "CPU"


def inference_time(f, test_videos, device, bs=32):
    """Wall-clock seconds to predict all test videos (1x, t=1) at batch size 32; each batch is moved
    to the device before the clock starts, as the handout requires."""
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(test_videos), bs):
            v = to_float(test_videos[i:i + bs]).to(device)
            X, _, _ = get_window_batch(v, 1, 1)
            sync(device); t0 = time.time()
            f(X)
            sync(device); total += time.time() - t0
    return total


def sci(x):
    """3 significant figures; scientific notation below 1e-3."""
    return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.3g}"
