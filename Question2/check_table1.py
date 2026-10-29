"""Table 1: sanity checks of the transformation and window functions (synthetic tensors only)."""
import torch
from common import M, R, get_window, window_indices, ConvLSTMForecaster, count_params

torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)
p = torch.zeros(20, 1, 64, 64); p[7, 0, 10, 20] = 1

rows = []
rows.append(("(a) max|M(M(x)) - x|", (M(M(x)) - x).abs().max().item(), 0, torch.equal(M(M(x)), x)))
rows.append(("(b) max|R^4(x) - x|", (R(R(R(R(x)))) - x).abs().max().item(), 0, torch.equal(R(R(R(R(x)))), x)))
lhs, rhs = M(R(M(x))), R(x, -1)
rows.append(("(c) max|MRM(x) - R^-1(x)|", (lhs - rhs).abs().max().item(), 0, torch.equal(lhs, rhs)))
for name, val, exp, eq in rows:
    print(f"{name:32s} expected {exp}  yours {val}  torch.equal={eq}")

def bright(v):
    return tuple(int(i) for i in (v == v.max()).nonzero()[0][[0, 2, 3]])

print("(d) bright pixel of M(p):", bright(M(p)), "expected (7, 10, 43)")
print("(e) bright pixel of R(p):", bright(R(p)), "expected (7, 43, 10)")

# (f)-(h): frame f filled with value f -> read off which frames the window returns
vid = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).contiguous()
for tag, k, exp in [("(f) 1x", 1, "1-5 / 6-10"), ("(g) 2x", 2, "1,3,5,7,9 / 11,13,15,17,19"),
                    ("(h) 3x", 3, "1,4,7,10,13 / 16,19")]:
    X, Y, mask = get_window(vid, k=k, t=1)
    xin = [int(X[i].flatten()[0]) for i in range(5)]
    yt = [int(Y[i].flatten()[0]) for i in range(5) if mask[i] > 0]
    print(f"{tag} input / target frames: {xin} / {yt}   expected {exp}")

print("Model A params:", count_params(ConvLSTMForecaster()), "(expected 480784)")
