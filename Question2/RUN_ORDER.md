# Question 2 – run order (free Colab friendly)

All commands run from inside `Question2/`. Dependencies: torch, torchvision, opencv-python, matplotlib,
scipy, pillow (all preinstalled on Colab).

## Colab setup (once per session)
```python
from google.colab import drive; drive.mount('/content/drive')
import os
os.environ['PA2_RUNS'] = '/content/drive/MyDrive/PA2/runs'   # checkpoints + weights survive disconnects
os.environ['PA2_DATA'] = '/content/data'                      # re-downloaded each session (fast)
%cd /content/PA2/Question2        # wherever you upload/clone this folder
```
Runtime → Change runtime type → T4 GPU. If a session dies, re-run the same `train.py` command: it
resumes from the last finished epoch. (Per-epoch times in the log then span sessions; say so in the report.)
`results/` and `figures/` are written next to the scripts – copy them to Drive too.

## Order (⛔ = commit something to git before continuing)
| # | Command | Produces |
|---|---|---|
| 1 | `python check_table1.py` | Table 1 (must match "Expected"; fix before anything else) |
| 2 | `python fig1_flow.py` | Figure 1 + median speed / pixel count |
| 3 | `python train.py --model A --seed 0` then `--seed 1` | model A |
| 4 | `python stage_A1.py` | Table 2, Figure 2 |
| ⛔ | **Write Q2.1–Q2.4 notes and Q2.5 prediction; commit Q2.5** | uses Figure 1, Table 2, reach = 8 px |
| 5 | `python stage_A2.py` | Tables 3–4, Figures 3–5 (+ GIFs in `figures/gifs/`) |
| ⛔ | **Write and commit Q2.11 (≤3 sentences)** | before training anything below |
| 6 | `python train.py --model B --seed 0/1`, `--model C --seed 0/1` | models B, C |
| 7 | `python train.py --model D_p8 --seed 0` and `--model D_dil2 --seed 0` | D candidates |
| 8 | `python select_D.py` | picks D on **validation only**, copies it as D seed 0 |
| 9 | `python train.py --model D --seed 1` | model D seed 1 |
| 10 | `python stage_final.py` | Tables 5–6, Figure 6 (+ GIFs) |

Optional D+B: `python train.py --model DB --seed 0/1` (after step 8), then `python stage_final.py --final DB`.

Runs needed in total: A×2, B×2, C×2, D candidates×2, D seed 1 = 9 trainings (30 epochs each).

## Report notes
* Hardware line for Tables 2/5: printed in `results/table*.md` and stored in each `runs/*/log.json`.
* Inference times are measured when a metrics file is first created (`results/metrics/*.json`); delete a
  JSON to re-time it on different hardware. Compare timings only between models measured in the same session.
* Figure 3/6 GIF paths: `figures/gifs/fig3_*.gif`, `figures/gifs/fig6a_*.gif`; put them in the public repo.
* Everything in `results/*.md` is already "mean (seed 0, seed 1)" with 3 s.f. / scientific below 1e-3.
