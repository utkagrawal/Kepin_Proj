#!/bin/bash
#SBATCH --partition=gpu-P100
#SBATCH --job-name=kepin_baseline_fd001
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=kepin_baseline_fd001.log
#SBATCH --error=kepin_baseline_fd001.err

eval "$(conda shell.bash hook)"
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

echo "=== Baseline regression — FD001, condition_dim=0 ==="
echo "Expected RMSE ~12.92 (paper). Any major deviation = broken baseline."
python -u kepin_cmapss_optimized.py \
    --dataset CMAPSS_FD001 \
    --output_dir experiments_result_baseline_fd001 \
    --condition_dim 0

cp -r experiments_result_baseline_fd001 ~/Kepin_code/
echo "Baseline regression complete."
