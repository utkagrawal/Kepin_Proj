#!/bin/bash
#SBATCH --partition=gpu-P100
#SBATCH --job-name=kepin_half1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=kepin_output_half1.log
#SBATCH --error=kepin_error_half1.log

eval "$(conda shell.bash hook)"
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

# Run FD001
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 0 \
    --output_dir experiments_result_half1 \
    --epochs 200 --seed 42 --condition_dim 3

# Run FD003
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 2 \
    --output_dir experiments_result_half1 \
    --epochs 200 --seed 42 --condition_dim 3

cp -r experiments_result_half1 ~/Kepin_code/
