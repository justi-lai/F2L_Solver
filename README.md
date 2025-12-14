# Rubik's Cube Mamba Solver

A high-performance Rubik's Cube solving model using the **Mamba** (State Space Model) architecture. This model is designed to imitate the **CFOP** (Cross, F2L, OLL, PLL) method, predicting moves based on the current 3D state of the cube.

## Overview

Unlike traditional solvers that search for the absolute shortest path (God's Number), this model imitates human-like algorithmic solving.
- **Architecture**: Mamba (Linear-Time Sequence Modeling) + Virtual Cube Embedding (20 pieces × 24 positions).
- **Performance**: 99.6% Accuracy on next-token prediction.
- **Inference**: Uses **Beam Search** to consistently solve scrambled cubes.

## Models

We have trained two distinct variations of the model, each serving a different purpose. (Note: Data used for training is ~170GB and not hosted on GitHub).

### Run 1: The "Optimal" Imitator
- **Training Data**: Generated with optimization enabled (`optimize=True`).
- **Characteristics**: 
    - Finds efficient solutions.
    - **No Cube Rotations** (`x`, `y`, `z` are removed).
    - Solves purely through face turns.
- **Best For**: Speed and move efficiency.

### Run 2: The "Realistic" Imitator (Fine-Tune Ready)
- **Training Data**: Generated with optimization disabled (`optimize=False`).
- **Characteristics**:
    - Includes **Cube Rotations** (`x`, `y`, `z`).
    - Mimics how humans actually hold and rotate the cube during F2L/OLL/PLL.
- **Best For**: **Fine-tuning** on real human solves (e.g., from `reco_scraper`).

## Usage

### 1. Data Generation
Generate your own training data using the provided pipeline. This allows for massive datasets (1M+ samples) via sharding.

```bash
# Generate 1 Million samples for Run 2 (Unoptimized / Rotations)
python3 data_generation/generate_dataset.py \
    --num_samples 1000000 \
    --output_path data/dataset_1m_r \
    --workers 16

# Generate for Run 1 (Optimized / No Rotations)
python3 data_generation/generate_dataset.py \
    --num_samples 1000000 \
    --output_path data/dataset_1m_opt \
    --optimize
```

### 2. Training
Train the Mamba model. By default, this uses a vocab size of 38 (includes rotations).

```bash
python3 models/train.py \
    --data_path data/dataset_1m_r \
    --weight_decay 0.1 \
    --epochs 50 \
    --batch_size 64 \
    --save_dir checkpoints/run_2
```

### 3. Inference & Testing
Evaluate the model using interactive beam search. This generates a random scramble and attempts to solve it live.

```bash
# Interactive Mode with Beam Search (Recommended Width: 5-10)
python3 models/test_model.py \
    --checkpoint checkpoints/run_2/best_model.pt \
    --interactive \
    --beam_width 10
```

## Future Work: Fine-Tuning
The `reco_scraper` directory contains tools to scrape and process reconstructed solves (~9,000 solves) from top speedcubers. The goal is to fine-tune the **Run 2** model on this human data to create a style-imitating AI.

## Project Structure
- `models/`: Mamba architecture (`general_model.py`), training (`train.py`), and testing scripts.
- `data_generation/`: Virtual Cube logic and parallel generation pipeline.
- `PyCube-Solver/`: Underlying engine for cube logic and solver validation.
- `reco_scraper/`: Scraper for CubeSolves.com (WIP).
