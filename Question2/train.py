"""Train one learned model under the PA2 protocol (Adam 1e-3, batch 32, 30 epochs, final weights).

    python train.py --model A --seed 0

Models: A, B, C, D_p8, D_dil2 (D candidates), D (chosen by select_D.py). Resumable: re-run the same
command after a Colab disconnect. Outputs runs/<model>_seed<seed>/{model.pt, log.json}.
"""
import argparse
import json
import os

import torch

import common as c

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seed", type=int, required=True, choices=[0, 1])
ap.add_argument("--epochs", type=int, default=30)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
spec = c.spec_of(args.model)
out = c.run_dir(args.model, args.seed)
os.makedirs(out, exist_ok=True)
if os.path.exists(os.path.join(out, "model.pt")):
    raise SystemExit(f"{out}/model.pt already exists - finished.")

train, val, _ = c.split_videos(c.load_moving_mnist())
probe = c.make_model(args.model)
n_params, reach = c.count_params(probe), c.reach_px(probe)
print(f"{args.model} seed {args.seed}: params={n_params}, reach={reach}px, device={device}, "
      f"speed_aug={spec['speed']}, sym_aug={spec['sym']}")
if args.model != "A" and spec["arch"] == {}:
    assert n_params == 480784
if args.model == "A":
    assert n_params == 480784, n_params
if spec["arch"]:  # D candidates: budget and strictly larger reach than A
    assert n_params <= 721176, n_params
    assert reach > 8, reach

model, epoch_times, hist = c.train_model(
    lambda: c.make_model(args.model), train, args.seed, device, speed_aug=spec["speed"],
    sym_aug=spec["sym"], epochs=args.epochs, val_videos=val,
    ckpt_path=os.path.join(out, "ckpt.pt"))

torch.save(model.state_dict(), os.path.join(out, "model.pt"))
with open(os.path.join(out, "log.json"), "w") as fh:
    json.dump({"model": args.model, "seed": args.seed, "params": n_params, "reach_px": reach,
               "hardware": c.hardware(), "mean_epoch_sec": sum(epoch_times) / len(epoch_times),
               "epoch_times": epoch_times, "hist": hist,
               "note": "epoch_times may span several Colab sessions if training was resumed"}, fh, indent=1)
print("done; mean epoch seconds:", sum(epoch_times) / len(epoch_times))
