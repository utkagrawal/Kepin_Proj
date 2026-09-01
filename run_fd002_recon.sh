#!/bin/bash
#SBATCH --partition=gpu-P100
#SBATCH --nodelist=gpu-P100-01
#SBATCH --job-name=kepin_fd002_recon
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=kepin_fd002_recon.log
#SBATCH --error=kepin_fd002_recon.err

source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

echo "=== Conditioned model — FD002, condition_dim=3 ==="
python -u kepin_cmapss_optimized.py \
    --dataset CMAPSS_FD002 \
    --output_dir experiments_result_fd002_recon \
    --condition_dim 3

cp -r experiments_result_fd002_recon ~/Kepin_code/
echo "FD002 recon run complete."
