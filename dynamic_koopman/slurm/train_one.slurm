#!/bin/bash
# One training run. All arguments are passed to train.py, e.g.
#   sbatch -J FD001_eig_static_clean_r0 slurm/train_one.slurm --dataset FD001 --koopman eig --mode static --protocol clean --run_id 0
# A100 preferred; SLURM falls back to P100 when no A100 can start the job first.
#SBATCH --partition=gpu-A100,gpu-P100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman/slurm/logs/%x_%j.out
#SBATCH --error=/userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman/slurm/logs/%x_%j.err

echo "Job $SLURM_JOB_ID on $(hostname) [$SLURM_JOB_PARTITION] started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
# TF finds the GPU only with the pip-installed CUDA libraries on the path
NV=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
for lib in cudnn cublas cufft cusolver cusparse curand nccl nvjitlink cuda_runtime; do
    export LD_LIBRARY_PATH=$NV/$lib/lib:$LD_LIBRARY_PATH
done
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NV/cuda_nvcc"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

cd "$TMPDIR" || exit 1
python -c "import tensorflow as tf, sys; g = tf.config.list_physical_devices('GPU'); print('TF GPUs:', g); sys.exit(0 if g else 3)" || { echo "No GPU visible to TensorFlow - aborting"; exit 3; }
python /userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman/train.py "$@"
CODE=$?
echo "Finished $(date) with exit code $CODE"
exit $CODE
