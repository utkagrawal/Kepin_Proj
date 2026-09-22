#!/bin/bash
#SBATCH --job-name=fd004_cond
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/slurm/fd004_cond_%j.out
#SBATCH --error=/userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/slurm/fd004_cond_%j.err

cd "$TMPDIR" || exit 1

echo "========================================================"
echo " KePIN FD004 Conditioned SLURM Job"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $SLURMD_NODENAME"
echo " Started  : $(date)"
echo "========================================================"

source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/curand/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvcc"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

OUTPUT_DIR="/userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/results_fd004_conditioned_3057"

echo "Output directory: $OUTPUT_DIR"

python /userhome/mtech/a.utkarsh/RCKePIN/regime_conditioned/train_fd004_conditioned.py \
    --output_dir "$OUTPUT_DIR" \
    --signal_type a \
    --start_run 1 \
    --verbose 1

EXIT_CODE=$?

echo "========================================================"
echo " Job finished: $(date)"
echo " Exit code  : $EXIT_CODE"
echo "========================================================"

if [ -f "$OUTPUT_DIR/fd004_conditioned_results.json" ]; then
    echo "Results summary:"
    python -c "
import json
with open('$OUTPUT_DIR/fd004_conditioned_results.json') as f:
    results = json.load(f)
for r in results:
    run = r.get('run_id', '?')
    rmse = r.get('rmse', 'N/A')
    mae  = r.get('mae', 'N/A')
    print(f'  {str(run):>10}: RMSE={rmse:.4f}, MAE={mae:.4f}')
"
fi

exit $EXIT_CODE
