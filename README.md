# Methane Leak Detection with Machine Learning

This repository contains our student research project for detecting possible methane leaks from satellite imagery and related geospatial data.

## Project goals

- Build a reproducible data-preparation pipeline.
- Establish a simple baseline model before experimenting with advanced deep learning.
- Compare model predictions with labelled methane plume or non-plume examples.
- Record experiments, limitations, and decisions so another student can reproduce the work.

## Team workflow

1. Never work directly on `main`.
2. Create a focused branch such as `member1/data-preprocessing` or `member3/baseline-model`.
3. Make small commits that explain one change.
4. Push the branch and open a pull request into `main`.
5. Ask at least one teammate to review the pull request.
6. Merge only after the automated test check passes and the branch is up to date.

Full instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Local setup

### 1. Create a Python environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CPU-only PyTorch installation, use the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) and select the appropriate operating system and CPU option if required.

### 3. Run automated tests

From the repository root, run:

```bash
pytest -q
```

### 4. Run the toy training smoke test

```bash
python -m src.training.train_toy
```

This runs a small segmentation model on synthetic data to verify that the training pipeline works. It is a technical smoke test, not a real methane-detection experiment.

Each run creates a directory under `results/` containing:

- `config.yaml` — the configuration used for the run.
- `seed.txt` — the random seed.
- `checkpoint.pt` — the saved model checkpoint.

To verify checkpoint loading and prediction, run:

```bash
python -m src.training.check_checkpoint
```

The checkpoint verification script uses a specific run path. Update the path in `src/training/check_checkpoint.py` if that run directory no longer exists.

## Repository map

- `data/` — local data instructions.
- `manifests/` — dataset inventory and manifests.
- `configs/` — experiment settings.
- `notebooks/` — dataset inspection and exploration.
- `src/` — implementation modules.
- `results/` — local training-run artifacts.
- `reports/` — handovers, methodology notes, and decisions.
- `tests/` — automated checks.
- `.github/` — collaboration automation.

## Data and research integrity

Do not commit raw satellite data, credentials, API keys, model checkpoints, or large generated files. Read [data/README.md](data/README.md) before adding data.

This is the project foundation. Dataset choice, licences, labels, preprocessing choices, and model results must be verified by the team before being described as final.

The current toy training pipeline uses synthetic data and does not establish real-world methane-detection accuracy or generalization to satellite imagery.