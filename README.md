# Coupled FSI Training

This repository contains the implementation of a Attention-based Time-conditioned Heterogeneous gRaph Neural OperatOr for flow-structure iNteraction(AeTHERON) simulations. The project focuses on preprocessing simulation data, training a neural FSI model, and verifying its performance.

## Project Structure
- `src/`: Source code
  - `model/`: Neural FSI model and operators
  - `dataloader/`: Data loading and preprocessing scripts
  - `training/`: Training and evaluation scripts
- `data/`: Simulation metadata and datasets
- `logs/`: Training logs and model checkpoints
- `utils/`: Utility scripts and best model weights
- `notebooks/`: Jupyter notebooks for exploration

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/coupled-training.git
   cd NeuralOpt_FSI/
2. Setup Environment
   conda env create -f ./utils/NeuralFSIEnv.yml
3. Run model of distributed cluster
    sbatch job
