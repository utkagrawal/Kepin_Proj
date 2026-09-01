#!/bin/bash
# run_phase10_fd002_kan_fast.sh
#SBATCH --partition=gpu-P100
#SBATCH --nodelist=gpu-P100-02
#SBATCH --job-name=fd002_kan_f
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=phase10_fd002_kan_fast.log

source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

mkdir -p ~/Kepin_code/experiments_result/phase10/fd002_kan_fast

python -u kepin_cmapss_optimized.py \
    --dataset CMAPSS_FD002 \
    --condition_dim 3 \
    --conditioning raw3 \
    --condition_net_type kan \
    --epochs 10 \
    --n_runs 1 \
    --output_dir experiments_result/phase10/fd002_kan_fast

cp -r experiments_result/phase10/fd002_kan_fast ~/Kepin_code/experiments_result/phase10/
echo "FD002 Fast KAN finished"
