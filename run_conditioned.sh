#!/bin/bash
#SBATCH --partition=gpu-P100
#SBATCH --job-name=kepin_conditioned
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=kepin_conditioned.log
#SBATCH --error=kepin_conditioned.err

eval "$(conda shell.bash hook)"
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

echo "=== STEP 5: Conditioned Koopman — FD002, condition_dim=3 ==="
echo "K(mu) = U * diag(sigmoid(s + delta_s(mu))) * V^T"
echo "Condition columns: setting1, setting2, setting3 (indices 0,1,2 in FD002 feature_cols)"
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 1 \
    --output_dir experiments_result_conditioned \
    --epochs 250 --seed 42 --condition_dim 3

echo ""
echo "=== STEP 5: Conditioned Koopman — FD004, condition_dim=3 ==="
python -u scripts/train.py \
    --config configs/datasets_kepin_config.json \
    --dataset_idx 3 \
    --output_dir experiments_result_conditioned \
    --epochs 250 --seed 42 --condition_dim 3

cp -r experiments_result_conditioned ~/Kepin_code/
echo "Conditioned training complete."
