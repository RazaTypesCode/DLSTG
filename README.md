# PA3 — Spatio-Temporal GNN Forecasting on LargeST-SD

AI651 Deep Learning for Space, Time and Graphs, Fall 2026, Assignment 3.

Everything (Questions 1–7) is in one notebook written for Google Colab:

- [`PA3.ipynb`](PA3.ipynb): the notebook without outputs, ready to run.
- `PA3_results.ipynb`: the same notebook after the reported run, with all printed results and figures.

## Reproducing the results

1. Open `PA3.ipynb` in Colab (*File → Upload notebook*). A GPU runtime is optional; it only shortens training.
2. **Smoke test (recommended first):** set `SMOKE_TEST = True` in the first code cell and choose *Runtime → Run all*. It builds tiny synthetic files in the same format as the real data and runs every step in a few minutes.
3. **Real run:** set `SMOKE_TEST = False`, then *Runtime → Restart session and run all*. The notebook downloads about 2.3 GB, extracts the 716 San Diego sensors, trains the four models and writes all tables and figures.
4. The last cell downloads `pa3_outputs.zip` containing `results/`, `figures/` and `checkpoints/`. Save the executed notebook with *File → Download → .ipynb*.

Switches in the first code cell:

| Switch | Default | Meaning |
|---|---|---|
| `SMOKE_TEST` | `False` | synthetic data + 2 epochs, to check the pipeline |
| `USE_DRIVE` | `False` | cache processed data (~100 MB) and results in `MyDrive/DLSTG_PA3` |
| `RESUME` | `True` | reuse a model whose checkpoint and metrics already exist |
| `SEED` | `42` | seed for Python, NumPy and PyTorch |
| `AGG` | `'sum'` | 15-minute aggregation of the 5-minute counts |

## Data

LargeST (Liu et al., NeurIPS 2023 Datasets and Benchmarks), licence CC BY-NC 4.0. No data files are committed to this repository.

The notebook downloads from the Hugging Face copy used by the Torch Spatiotemporal library, so no Kaggle account is needed:

| File | Size |
|---|---|
| `large_st/sensors.zip` (metadata + road-distance adjacency) | 0.7 MB |
| `large_st/2020/data.h5` (5-minute flow, all 8,600 sensors) | 1.16 GB |
| `large_st/2021/data.h5` | 1.17 GB |

Base URL: `https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/large_st/`. The official release is on Kaggle (`liuxu77/largest`).

## Setup summary

| Item | Choice |
|---|---|
| Sensors | `district == 11` (San Diego), 716 sensors, metadata row order used for the data and both axes of the adjacency |
| Resampling | 5-minute vehicle counts summed to 15 minutes; a 15-minute value requires all three readings |
| Splits | train 2020, validation Jan–Mar 2021, test Apr–Dec 2021; windows never cross a split |
| Windows | 8 inputs (2 h) → 4 targets (1 h) |
| Normalisation | per-sensor z-score from observed 2020 values only |
| Missing values | inputs filled with the sensor's training mean; targets masked from loss and metrics |
| Model | Conv1d temporal encoder (d = 32, k = 3) → one graph convolution with a residual connection, H = Z + ReLU(ÂZW + b) → linear head |
| Graphs | none (temporal-only), fixed road graph, adaptive (top-10), hybrid (α = 0.5) |
| Training | Adam, lr 1e-3, batch 64, masked MAE, up to 50 epochs, early stopping (patience 5) on validation MAE |

The exact configuration of a run is written to `results/config.json`, the hardware to `results/hardware.json`, and per-epoch histories and final metrics to `results/history_*.json` and `results/metrics_*.json`.

## Code this work builds on

- LargeST official code, commit `b7c6ff5`: San Diego extraction (`data/sd/generate_sd_dataset.ipynb`), resampling (`data/ca/process_ca_his.ipynb`), window offsets (`data/generate_data_for_training.py`), adjacency normalisation (`src/utils/graph_algo.py`), masked metrics (`src/utils/metrics.py`). <https://github.com/liuxu77/LargeST>
- Graph WaveNet adaptive adjacency `softmax(relu(E1 E2ᵀ))` (`src/models/gwnet.py` in LargeST; original <https://github.com/nnzhan/Graph-WaveNet>).
- AGCRN node-embedding graph (<https://github.com/LeiBAI/AGCRN>).
- Torch Spatiotemporal LargeST loader, commit `c46a23e`: data URLs, file layout, district selection (`tsl/datasets/large_st.py`). <https://github.com/TorchSpatiotemporal/tsl>

Exact line links are in the first cell of the notebook.
