
# TABA – Smart Agriculture Sustainability Evaluation

This repository contains code and experiments for the National College of Ireland H9ETS TABA project:
**"Evaluating Sustainability Strategies in Computer Vision for Smart Agriculture"**.

## Objectives
- Compare baseline deep learning models to energy-efficient architectures.
- Measure performance, environmental impact (energy, CO₂), and fairness.
- Provide reproducible experiments for the TABA final report.

## Folder Structure
- `data/` – Dataset (PlantVillage subset or crop dataset).
- `runs/` – Experiment outputs (metrics, emissions, logs).
- `src/` – Load, training, evaluation, and utility scripts.
- `results/` – Final tables and plots for report.

## Quick Start
```bash

# 1) Create a Python virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 4) Load data
python -m src.load_data --config src/config.yaml

# 4) Train baseline + sustainable models
python -m src.train --model resnet18 --config src/config.yaml
python -m src.train --model mobilenetv3_small_100 --config src/config.yaml

# 5) Evaluate + measure latency
python -m src.evaluate --run_dir runs/<timestamp>_resnet18
python -m src.evaluate --run_dir runs/<timestamp>_mobilenetv3_small_100
```

## Data Options
- **ImageFolder (recommended):** place data like `data/cifar10/train/<class>/*.jpg`, `val/`, `test/`.
- **CIFAR-10 (fallback):** set `data.mode: cifar10` in `src/config.yaml` to run end-to-end quickly.
