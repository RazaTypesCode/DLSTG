# Question 2 – run order

All scripts are run from inside `Question2/`. The Colab notebooks in `notebooks/` execute exactly these steps on a
T4 GPU with "Run all".

| # | Command | Produces |
|---|---|---|
| 1 | `python check_table1.py` | Table 1 (printed values should match the "Expected" column) |
| 2 | `python fig1_flow.py` | Figure 1, median speed, pixel count |
| 3 | `python train.py --model A --seed 0` and `--seed 1` | Model A |
| 4 | `python stage_A1.py` | Table 2, Figure 2 |
| 5 | `python stage_A2.py` | Tables 3–4, Figures 3–5 (+ GIFs in `figures/gifs/`) |
| 6 | `python train.py --model B/C --seed 0/1`, `--model D_p8 --seed 0`, `--model D_dil2 --seed 0` | Models B, C and the Model D candidates |
| 7 | `python select_D.py` | Chooses Model D on the validation split only (rule fixed in the file) |
| 8 | `python train.py --model D --seed 1` | Model D, seed 1 |
| 9 | `python stage_final.py` | Tables 5–6, Figure 6 (+ GIFs) |

`train.py` resumes unfinished runs and skips finished ones, so a step can safely be re-run. Outputs go to `results/`,
`figures/` and `figures/gifs/`. Reported numbers are "mean (seed 0, seed 1)" to three significant figures, in scientific
notation below 1e-3.

`predictions/Q2.5.md` and `predictions/Q2.11.md` (at the repository root) hold the predictions written before the
experiments; their commit times are the record.
