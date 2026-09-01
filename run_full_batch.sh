#!/bin/bash
#SBATCH --partition=gpu-P100
#SBATCH --job-name=kepin_full_batch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=kepin_output_full_batch.log
#SBATCH --error=kepin_error_full_batch.log

eval "$(conda shell.bash hook)"
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

echo "Starting FD002..."
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 1 \
    --output_dir experiments_result_full_batch \
    --epochs 250 --seed 42

echo "Starting FD003..."
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 2 \
    --output_dir experiments_result_full_batch \
    --epochs 250 --seed 42

echo "Starting FD004..."
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 3 \
    --output_dir experiments_result_full_batch \
    --epochs 250 --seed 42

cp -r experiments_result_full_batch ~/Kepin_code/
echo "All done! Results saved to ~/Kepin_code/experiments_result_full_batch"
