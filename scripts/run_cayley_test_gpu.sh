#!/bin/bash
# run_cayley_test_gpu.sh
#SBATCH --partition=gpu-P100
#SBATCH --nodelist=gpu-P100-01
#SBATCH --job-name=cayley_reg
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=cayley_test_gpu.log

source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

mkdir -p ~/Kepin_code/experiments_result/phase11/fd002_cayley_exact

# Running for 250 epochs to compare against the 14.56 baseline
python -u kepin_cmapss_optimized.py \
    --dataset CMAPSS_FD002 \
    --condition_dim 3 \
    --conditioning raw3 \
    --condition_net_type mlp \
    --epochs 250 \
    --n_runs 1 \
    --output_dir experiments_result/phase11/fd002_cayley_exact

cp -r experiments_result/phase11/fd002_cayley_exact ~/Kepin_code/experiments_result/phase11/
echo "Cayley test on GPU finished"
