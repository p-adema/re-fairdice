# PEDA Experiment Environment

This repository is a copy of the original [PEDA](https://github.com/baitingzbt/PEDA) project, enhanced with improved environment management and figure reproduction scripts.

## Environment Management

### Training (Pixi)
For training (Linux-based), we use [Pixi](https://pixi.sh/).

### Plotting (Pip)
For checking and reproducing figures (lighter, works on macOS/Linux), use standard pip.

Setup (using venv):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r figures/requirements.txt
```

## Figures
The `figures/` directory contains scripts and data to reproduce Figures 4, 5, and 6.

To reproduce the figures using the provided CSV data:
```bash
source .venv/bin/activate
python3 figures/plot_figure_4.py
python3 figures/plot_figure_5.py
python3 figures/plot_figure_6.py
```

The raw result processing scripts (`data_figure_X.py`) are also included but require the full raw experiment results (pkl files) to run.
