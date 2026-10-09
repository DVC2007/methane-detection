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

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

Do not commit raw satellite data, credentials, API keys, model checkpoints, or large generated files. Read [data/README.md](data/README.md) before adding data.

## Repository map

`data/` local data instructions; `configs/` experiment settings; `src/` implementation modules; `reports/` handovers and decisions; `tests/` automated checks; `.github/` collaboration automation.

This is the project foundation. Dataset choice, labels, preprocessing choices, and model results must be verified by the team before being described as final.
