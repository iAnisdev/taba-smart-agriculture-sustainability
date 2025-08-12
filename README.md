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
- `src/` – Training, evaluation, and utility scripts.
- `results/` – Final tables and plots for report.

## Quick Start
```bash
# Create environment
conda create -n taba python=3.10 -y
conda activate taba
```

# Install dependencies
```bash
pip install -r requirements.txt
```

# Train baseline model

```bash
python src/train.py --model resnet18 --config src/config.yaml
```

# Train sustainable model

```bash
python src/train.py --model mobilenetv3_small_100 --config src/config.yaml
```

# Evaluate

```bash
python src/evaluate.py --model_path runs/<folder>/model.pt
```
