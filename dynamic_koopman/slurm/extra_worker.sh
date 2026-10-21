#!/bin/bash
# Start one more queue worker on GPU $1 inside an existing job allocation, e.g.
# CAUTION: $1 must be a GPU allocated to THAT job (see its "Allocated GPUs" log line).
#   srun --jobid=<job> --overlap -n1 bash slurm/extra_worker.sh 1 &
source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
NV=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
for lib in cudnn cublas cufft cusolver cusparse curand nccl nvjitlink cuda_runtime; do
    export LD_LIBRARY_PATH=$NV/$lib/lib:$LD_LIBRARY_PATH
done
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=$1
cd /tmp || exit 1
python /userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman/queue_worker.py \
    --tasks /userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman/tasks/fd002.txt --worker "job${SLURM_JOB_ID}-gpu$1-extra"
