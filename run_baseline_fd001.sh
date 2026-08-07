#!/bin/bash
#SBATCH --partition=gpu-P100
#SBATCH --job-name=kepin_baseline_fd001
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=kepin_baseline_fd001.log
#SBATCH --error=kepin_baseline_fd001.err

eval "$(conda shell.bash hook)"
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

echo "=== STEP 4: Baseline regression — FD001, condition_dim=0 ==="
echo "Expected RMSE ~12.92 (paper). Any major deviation = broken baseline."
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 0 \
    --output_dir experiments_result_baseline_fd001 \
    --epochs 250 --seed 42

cp -r experiments_result_baseline_fd001 ~/Kepin_code/
echo "Baseline regression complete."
