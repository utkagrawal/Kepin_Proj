#!/bin/bash
#SBATCH --partition=gpu-A100
#SBATCH --job-name=kepin_A100_half2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=kepin_output_a100_half2.log
#SBATCH --error=kepin_error_a100_half2.log

eval "$(conda shell.bash hook)"
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

# Run FD002 (250 epochs to perfectly align the Cosine Annealing cycle)
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 1 \
    --output_dir experiments_result_a100_half2 \
    --epochs 250 --seed 42 --condition_dim 3

# Run FD004 (250 epochs to perfectly align the Cosine Annealing cycle)
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 3 \
    --output_dir experiments_result_a100_half2 \
    --epochs 250 --seed 42 --condition_dim 3

cp -r experiments_result_a100_half2 ~/Kepin_code/
