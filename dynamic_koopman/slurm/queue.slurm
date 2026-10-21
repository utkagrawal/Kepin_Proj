#!/bin/bash
# Runs the shared task queue on one node: WORKERS_PER_GPU workers on each of its GPUs.
#   sbatch -J dk_queue1 slurm/queue.slurm tasks/phase1.txt
# The mtech QOS allows 2 jobs per user, so submit this twice (same task file).
# A100 preferred; SLURM falls back to P100 when no A100 can start the job first.
#SBATCH --partition=gpu-A100,gpu-P100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=/userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman/slurm/logs/%x_%j.out
#SBATCH --error=/userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman/slurm/logs/%x_%j.err

TASKS=${1:?usage: sbatch slurm/queue.slurm <task file>}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
DK=/userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman

echo "Job $SLURM_JOB_ID on $(hostname) [$SLURM_JOB_PARTITION] started $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

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
export TF_FORCE_GPU_ALLOW_GROWTH=true

cd "$TMPDIR" || exit 1
python -c "import tensorflow as tf, sys; g = tf.config.list_physical_devices('GPU'); print('TF GPUs:', g); sys.exit(0 if g else 3)" || { echo "No GPU visible to TensorFlow - aborting"; exit 3; }

# Use ONLY the GPUs SLURM allocated to this job. nvidia-smi lists every GPU on the node,
# including GPUs allocated to other users' jobs, so it must not be used to pick devices.
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:?no GPU allocated}}"
echo "Allocated GPUs: ${GPUS[*]}"
for g in "${GPUS[@]}"; do
    for ((w = 0; w < WORKERS_PER_GPU; w++)); do
        CUDA_VISIBLE_DEVICES=$g python $DK/queue_worker.py --tasks $DK/$TASKS \
            --worker "job${SLURM_JOB_ID}-gpu${g}-w${w}" &
        sleep 20   # stagger start-up so workers claim different tasks cleanly
    done
done
wait
echo "Finished $(date)"
