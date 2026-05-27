# Multiaccuracy and Multicalibration with Proxy Groups

This is the code release for the paper [*Multiaccuracy and Multicalibration with Proxy Groups*](https://arxiv.org/abs/2409.20427) @ [ICML 2025](https://arxiv.org/pdf/2503.02870?)

by [Beepul Bharti](https://beepulbharti.github.io), Mary Versa Clemens-Sewall, Paul Yi, and [Jeremias Sulam](https://sites.google.com/view/jsulam).

---

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for Python environment and dependency management.

### 1. Install `uv`

If you do not already have `uv` installed, install it with `pip`:

```bash
pip install uv
```

### 2. Install dependencies

From the root of the repository, run:

```bash
uv sync
```

This creates a local virtual environment and installs the dependencies specified by the project.

---

## Reproducing paper results

The `ACS/` folder contains the necessary data and scripts to reproduce the results of the two prediction tasks on the American Community Survey (ACS) dataset.

Perform the following steps from the root of the repository.

### 1. Move into the ACS folder

```bash
cd ACS
```

### 2. Run an experiment

Run the experiment script with the desired experiment name and classifier:

```bash
uv run python run_experiment.py --exp <experiment name> --classifier <classifier>
```

Available experiment names and classifiers:

```text
Experiment names
ACSIncome_no_race, ACSPubcov_no_sex

Classifiers:
linear, tree, rf
```

### 3. Outputs

Results are saved under:

```text
ACS/results/<experiment_name>/
```

---

## References

```bibtex
@inproceedings{
bharti2025multiaccuracy,
title={Multiaccuracy and Multicalibration via Proxy Groups},
author={Beepul Bharti and Mary Versa Clemens-Sewall and Paul Yi and Jeremias Sulam},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=sGny74zx2V}
}
```
