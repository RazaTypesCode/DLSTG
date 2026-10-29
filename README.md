# AI651 Deep Learning for Space, Time and Graphs – Programming Assignment 2

Branch `PA2` of this repository. (PA1 is on the `PA1` branch.)

| Path | Content |
|---|---|
| `Question1/q1_diagram.py` | Question 1: script for the architecture figure of the river forecasting proposal (the written proposal is in the report) |
| `Question1/assignment2/`, `Question1/Question1.ipynb`, `Question1/checkpoints/` | Optional, ungraded companion notebook supplied with the handout |
| `Question2/` | Question 2: Moving MNIST ConvLSTM study (code, tables, figures, GIFs) |
| `Question2/RUN_ORDER.md` | Run order of the scripts |
| `notebooks/` | Colab driver notebooks: `PA2_<n>a` (input) and `PA2_<n>b` (executed output) per round |
| `predictions/Q2.5.md`, `predictions/Q2.11.md` | Predictions written **before** the experiments they refer to; their commit times are the evidence |

## How the experiments were run

Everything for Question 2 was run on free **Google Colab (T4 GPU)** with a driver notebook kept in `notebooks/`.
The notebook contains no model code: it clones this public repository (`git clone --branch PA2`), runs the scripts
in `Question2/` in the required order, and displays the results. No Google Drive is used; trained weights live in the
runtime's local disk and are backed up/restored by zip download/upload.

Notebook versions are numbered per round: `PA2_<n>a.ipynb` is the notebook prepared locally and uploaded to Colab,
`PA2_<n>b.ipynb` is the same notebook after execution on Colab (with outputs). The next round is `PA2_<n+1>a`.

Workflow: edit code locally → `git push` → upload `PA2_<n>a` to Colab → run → save the executed copy as `PA2_<n>b` →
results are copied back into the repo (`Question2/results/`, `Question2/figures/`, `Question2/figures/gifs/`).

The predictions required by the handout (`predictions/Q2.5.md`, `predictions/Q2.11.md`) were committed before the
experiments they refer to; their commit times are the record.

### Reproduce
1. Open Colab, upload the latest `notebooks/PA2_<n>a.ipynb`, set Runtime → T4 GPU.
2. Run all cells (setup → Table 1 / Figure 1 → model A → stress tests → B, C, D → final tables).

### Question 2 file map
| Script | Output |
|---|---|
| `check_table1.py` | Table 1 |
| `fig1_flow.py` | Figure 1 |
| `train.py --model {A,B,C,D_p8,D_dil2,D} --seed {0,1}` | trained models (30 epochs, Adam 1e-3, batch 32) |
| `stage_A1.py` | Table 2, Figure 2 |
| `stage_A2.py` | Tables 3–4, Figures 3–5 (+ GIFs) |
| `select_D.py` | Model D choice on the validation split only (rule fixed in the file) |
| `stage_final.py` | Tables 5–6, Figure 6 (+ GIFs) |

Shared code: `common.py` (data, transforms, windows, ConvLSTM, training), `evaluate.py` (metrics), `viz.py` (figures).

## AI assistance
Generative AI (Claude, via Claude Code) was used for this assignment, as permitted by the course policy. The prompts,
outputs and the edits made are listed in the AI-use section at the end of the submitted report (the report itself is
compiled separately and is not part of this repository).
