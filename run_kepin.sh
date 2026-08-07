#!/bin/bash
#SBATCH --partition=gpu-P100
#SBATCH --job-name=kepin_train
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-P100-02
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=kepin_output.log
#SBATCH --error=kepin_error.log

# 1. Load the conda environment (if conda is available by default)
# You are already in '(base)' in your terminal, so you likely just need:
eval "$(conda shell.bash hook)"
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

# 2. Safely move to the compute node's high-speed temporary directory (REQUIRED BY YOUR CLUSTER)
cd "$TMPDIR" || exit 1

# 3. Copy your project code from your home directory to this temporary directory
cp -r ~/Kepin_code .
cd Kepin_code

# 4. Run the code
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --output_dir experiments_result \
    --epochs 50 --seed 42 --condition_dim 3

# 5. Copy the results back to your home directory so you don't lose them!
# (The experiments_result folder is where the code saves weights and metrics)
cp -r experiments_result ~/Kepin_code/
