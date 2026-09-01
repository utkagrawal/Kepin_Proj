#!/bin/bash
#SBATCH --partition=gpu-P100
#SBATCH --job-name=kepin_conditioned
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=kepin_conditioned.log
#SBATCH --error=kepin_conditioned.err

eval "$(conda shell.bash hook)"
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

echo "=== Conditioned model — FD002, FD004, condition_dim=3 ==="
python -u kepin_cmapss_optimized.py \
    --dataset CMAPSS_FD002 \
    --output_dir experiments_result_conditioned \
    --condition_dim 3

python -u kepin_cmapss_optimized.py \
    --dataset CMAPSS_FD004 \
    --output_dir experiments_result_conditioned \
    --condition_dim 3

cp -r experiments_result_conditioned ~/Kepin_code/
echo "Conditioned runs complete."
