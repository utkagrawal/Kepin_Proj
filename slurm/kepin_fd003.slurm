#!/bin/bash
#SBATCH --job-name=kepin_fd003
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --output=/userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/slurm/kepin_fd003_%j.out
#SBATCH --error=/userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/slurm/kepin_fd003_%j.err

cd "$TMPDIR" || exit 1

# ──────────────────────────────────────────────────────────────────────────────
# KePIN FD003 Replication — SLURM Job
# Target: ensemble RMSE ≈ 11.42 on C-MAPSS FD003
# Author config: 3 runs × 300 epochs, batch=128, lr=8e-4
# ──────────────────────────────────────────────────────────────────────────────

echo "========================================================"
echo " KePIN FD003 SLURM Job"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $SLURMD_NODENAME"
echo " Started  : $(date)"
echo "========================================================"

# Activate conda environment with TF 2.16.1 + Keras 3.3.3
source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin

# Add pip-installed CUDA libraries to LD_LIBRARY_PATH for TF 2.16
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/curand/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
# TF 2.16 often requires NVVM too
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvcc"

# Verify GPU
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Verify TF sees the GPU
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TF version: {tf.__version__}')
print(f'GPUs visible to TF: {len(gpus)}')
for g in gpus: print(f'  {g}')
"

# Output directory with job ID for uniqueness
OUTPUT_DIR="/userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/results_fd003_${SLURM_JOB_ID}"

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python /userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/train_fd003.py \
    --output_dir "$OUTPUT_DIR" \
    --verbose 1

EXIT_CODE=$?

echo ""
echo "========================================================"
echo " Job finished: $(date)"
echo " Exit code  : $EXIT_CODE"
echo " Results in : $OUTPUT_DIR"
echo "========================================================"

# Print final RMSE from results JSON if available
if [ -f "$OUTPUT_DIR/fd003_results.json" ]; then
    echo ""
    echo "Results summary:"
    python -c "
import json
with open('$OUTPUT_DIR/fd003_results.json') as f:
    results = json.load(f)
for r in results:
    run = r.get('run_id', '?')
    rmse = r.get('rmse', 'N/A')
    mae  = r.get('mae', 'N/A')
    print(f'  {str(run):>10}: RMSE={rmse:.4f}, MAE={mae:.4f}')
"
fi

exit $EXIT_CODE
