# Kruskal-Wallis Test Analysis

This repository contains scripts to perform Kruskal-Wallis H-tests on experimental scores to evaluate statistical significance across different environments and gradients.

## Files

*   **`gen_kw_fixed.py`**: Analyzes "Fixed" type experiments (e.g., `Fixed-Grad0`, `Fixed-Grad0.1`). Grouped by Environment, Type, and Method.
*   **`gen_kw_rerun.py`**: Analyzes "Rerun" type experiments. Grouped by Environment and Type.
*   **`scores_v2.py`**: Contains the raw experimental data used for analysis.

## Usage

Run the scripts using Python 3:

```bash
python3 gen_kw_fixed.py
python3 gen_kw_rerun.py
```

## Dependencies

*   Python 3+
*   `scipy`
