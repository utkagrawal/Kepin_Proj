#!/bin/bash
set -e

# Load conda environment
source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin

# We will run the optimized CMAPSS training specifically for FD002
# with 30 epochs and save it to experiments_result/phase11/fd002_orth_reg

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Starting Orthogonality Regularization Retraining (30 epochs)"
python kepin_cmapss_optimized.py --dataset CMAPSS_FD002 --epochs 30 --n_runs 1 --output_dir experiments_result/phase11/fd002_orth_reg
echo "Retraining completed successfully!"
