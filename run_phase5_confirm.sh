#!/bin/bash
# run_phase5_confirm.sh
# Runs phase 5 confirmation on FD002 (3-run ensemble, 250 epochs)
#SBATCH --partition=gpu-P100
#SBATCH --nodelist=gpu-P100-02
#SBATCH --job-name=phase5_confirm
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=phase5_confirm.log
#SBATCH --error=phase5_confirm.err

# 1. Setup conda and environment
source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

# 2. Setup output directories
PHASE5_DIR="experiments_result/ablation_conditioning/phase5_confirm"
mkdir -p "$PHASE5_DIR"
RESULTS_CSV="$PHASE5_DIR/results_table.csv"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "variant,run_idx,host,rmse,score" > "$RESULTS_CSV"
fi

HOST="gpu-P100-02"
var="raw2_bc"
echo "Running Phase 5 confirmation for variant: $var"

OUT_DIR="$PHASE5_DIR/FD002_raw2_bc"
mkdir -p "$OUT_DIR"

# Run the optimized training script (defaults to n_runs=3, epochs=250)
python -u kepin_cmapss_optimized.py \
    --dataset CMAPSS_FD002 \
    --output_dir "$OUT_DIR" \
    --condition_dim 3 \
    --conditioning "$var" \
    > "$OUT_DIR/run.log" 2>&1
    
JSON_FILE="$OUT_DIR/optimized_cmapss_results.json"
if [ -f "$JSON_FILE" ]; then
    RMSE=$(python -c "import json; print(json.load(open('$JSON_FILE'))[-1].get('rmse', 'N/A'))")
    
    # Parse score
    SCORE=$(python -c "
import sys, os
lpath = '$OUT_DIR/run.log'
if os.path.exists(lpath):
    for line in reversed(open(lpath).readlines()):
        if 'Score:' in line:
            print(line.split('Score:')[-1].split(',')[0].strip())
            sys.exit(0)
print('N/A')
")
else
    RMSE="ERROR"
    SCORE="ERROR"
fi

echo "raw2_bc_ensemble,0,$HOST,$RMSE,$SCORE" >> "$RESULTS_CSV"
echo "Finished $var: RMSE=$RMSE, Score=$SCORE"

cp -r "$PHASE5_DIR" ~/Kepin_code/experiments_result/ablation_conditioning/
echo "Phase 5 confirmation completed."
