#!/bin/bash
#SBATCH --job-name=kepin_cylinder
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/kepin_cylinder_%j.out
#SBATCH --error=slurm/logs/kepin_cylinder_%j.err
#SBATCH --mem=64G

cd "$TMPDIR" || exit 1

source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin

python /userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/train_multidomain_conditioned.py --dataset Cylinder_Wake --verbose 1 --n_runs 3 --output_dir /userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/results_cylinder_wake_conditioned_${SLURM_JOB_ID}
