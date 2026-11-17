#!/bin/bash
# Start extra queue workers inside an already running job (srun --overlap), one per allocated GPU.
#   srun --jobid=<id> --overlap slurm/extra_workers.sh tasks/<file>.txt
TASKS=${1:?usage: extra_workers.sh <task file>}
DK=/userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman
source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
NV=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
for lib in cudnn cublas cufft cusolver cusparse curand nccl nvjitlink cuda_runtime; do
    export LD_LIBRARY_PATH=$NV/$lib/lib:$LD_LIBRARY_PATH
done
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NV/cuda_nvcc" TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 TF_FORCE_GPU_ALLOW_GROWTH=true
export TMPDIR=/tmp/$USER/extra_$SLURM_JOB_ID; mkdir -p $TMPDIR; cd $TMPDIR
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:?no GPU}}"
echo "extra workers in job $SLURM_JOB_ID on GPUs ${GPUS[*]} at $(date)"
for g in "${GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=$g python $DK/queue_worker.py --tasks $DK/$TASKS --worker "job${SLURM_JOB_ID}-gpu${g}-x" &
    sleep 20
done
wait
