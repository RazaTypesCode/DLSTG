# AI651 Deep Learning for Space, Time and Graphs – Programming Assignment 2

Code for Programming Assignment 2 (branch `PA2`; PA1 is on the `PA1` branch).
Question 1 is a written design proposal for a river-forecasting model (this repo holds only its architecture diagram script).
Question 2 is a Moving MNIST ConvLSTM study of symmetry and speed ("Motion or Memory?"). The written report, with all
tables, figures and answers, is compiled separately and is not part of this repository.

## Repository layout

| Path | Content |
|---|---|
| `Question1/q1_diagram.py` | Script for the Question 1 architecture figure |
| `Question1/assignment2/`, `Question1/Question1.ipynb`, `Question1/checkpoints/` | Optional, ungraded companion notebook supplied with the handout |
| `Question2/` | Question 2 code, plus the generated `results/`, `figures/` and `figures/gifs/` |
| `Question2/RUN_ORDER.md` | The scripts in the order they are run |
| `notebooks/` | Colab driver notebooks (see below) |
| `predictions/Q2.5.md`, `predictions/Q2.11.md` | Predictions written **before** the experiments they refer to; their commit times are the record |

## Question 2 scripts

All scripts are run from inside `Question2/`.

| Script | Output |
|---|---|
| `check_table1.py` | Table 1 |
| `fig1_flow.py` | Figure 1 |
| `train.py --model {A,B,C,D_p8,D_dil2,D} --seed {0,1}` | Trained models (30 epochs, Adam 1e-3, batch 32) |
| `stage_A1.py` | Table 2, Figure 2 |
| `stage_A2.py` | Tables 3–4, Figures 3–5 (+ GIFs) |
| `select_D.py` | Model D choice, made on the validation split only (the rule is fixed in the file) |
| `stage_final.py` | Tables 5–6, Figure 6 (+ GIFs) |

Shared code: `common.py` (data, transforms, windows, ConvLSTM, training), `evaluate.py` (metrics), `viz.py` (figures).
The model registry in `common.py` defines models A, B, C and the two Model D candidates; `select_D.py` records which
candidate was chosen.

## Reproducing the results

The experiments were run on free Google Colab with a T4 GPU, driven by a single notebook that needs no Google Drive.
The notebook contains no model code: it clones this repository (`git clone --branch PA2`), runs the scripts in
`Question2/` in the order given in `Question2/RUN_ORDER.md`, and displays the output. At the end it zips the generated
outputs so they can be downloaded. Trained weights are not stored in the repository.

To reproduce: open Colab, upload a notebook from `notebooks/`, select Runtime → T4 GPU, and run all cells.
Each model trains in roughly 5 minutes per seed on a T4 (about 11 s per epoch for Model A); a CPU-only run is possible but much slower.

Notebooks come in pairs: `PA2_<n>a.ipynb` is the notebook as uploaded to Colab and `PA2_<n>b.ipynb` is the same
notebook after execution, with outputs. `PA2_3b.ipynb` is the final executed run; the tables and figures in
`Question2/results/` and `Question2/figures/` come from these runs.

## AI assistance
Generative AI (Claude, via Claude Code) was used for this assignment, as permitted by the course policy. The use is
described in the AI-use section at the end of the report.
