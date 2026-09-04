# Regime-Aware KePIN

This directory contains the codebase, experiments, and results for the **Regime-Aware Koopman-embedded Prognostic Inference Network (Regime-KePIN)**.

## Overview
The original KePIN architecture employs a single, global Koopman operator to capture latent degradation dynamics. This project validates a research hypothesis: *Can we improve Remaining Useful Life (RUL) prediction by learning latent degradation regimes dynamically (unsupervised) and assigning a distinct Koopman operator to each regime?*

We introduced:
1. **RegimeKoopmanOperator**: An ensemble of $R$ SVD-based Koopman operators.
2. **Regime Inference Head**: An MLP that outputs dynamic transition probabilities ($\pi_t$) for each regime at every timestep.
3. **Regime Smoothness Loss**: An optional constraint penalizing rapid jumps between regimes ($\sum \|\pi_t - \pi_{t-1}\|^2$).

## Project Structure
- `regime_kepin_model.py`: Core architecture extensions (RegimeKoopmanOperator, RegimeKePINModel) and updated loss functions.
- `run_regime_experiments.py`: Training script for CMAPSS FD001 across multiple ablation configurations.
- `analyze_regimes.py`: Visualization tool to plot $\pi_t$ regime probabilities over the engine's lifetime.
- `submit_training.slurm`: SLURM batch script for running the experimental suite on A100 GPUs.
- `experiments_regime/`: Contains trained model weights, evaluation metrics, and Numpy arrays of predictions for all seeds.

## Usage

### 1. Training
To reproduce the experiments on a SLURM cluster:
```bash
sbatch submit_training.slurm
```
This runs the full suite (Baseline, 3-Koopman Static, Proposed, No-Smoothness, No-Physics) across 3 random seeds. 

### 2. Analysis
Generate regime transition plots and entropy statistics:
```bash
python analyze_regimes.py --res_dir experiments_regime/proposed_seed42 --out_dir plots_proposed
```

## Results & Conclusion
The completely unsupervised regime formulation **failed** to outperform the static 3-Koopman baseline.

| Model | RMSE (Mean) | MAE (Mean) |
|-------|-------------|------------|
| 3-Koopman (Static Baseline) | **18.93** | **13.98** |
| Proposed (Regime-KePIN) | 19.38 | 13.94 |
| Proposed (No Physics) | 19.30 | 14.19 |
| Proposed (No Smooth) | 19.35 | 13.78 |

**Why it failed:**
- When penalized with temporal smoothness, the network suffers from **Regime Collapse**, ignoring the ensemble and routing ~96% of data through a single operator.
- When smoothness is removed, it suffers from **Regime Chaos**, jumping randomly between operators at every timestep rather than modeling contiguous degradation phases.

Adding raw parallel capacity (the static baseline) is currently more robust than unsupervised dynamic switching for RUL prediction on CMAPSS.
