#!/bin/bash
# run_phase6_static_baseline.sh
# Runs phase 6: Static Baseline on FD002 with FIXED Koopman SVD module.
#SBATCH --partition=gpu-A100
#SBATCH --nodelist=gpu-A100-01
#SBATCH --job-name=phase6_static
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=phase6_static.log
#SBATCH --error=phase6_static.err

# 1. Setup conda and environment
source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

# 2. Setup output directories
PHASE6_DIR="experiments_result/ablation_conditioning/phase6_static"
mkdir -p "$PHASE6_DIR"
RESULTS_CSV="$PHASE6_DIR/results_table.csv"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "variant,run_idx,host,rmse,score" > "$RESULTS_CSV"
fi

HOST="gpu-A100-01"
var="static_fixed"
echo "Running Phase 6 static baseline (FIXED SVD): $var"

OUT_DIR="$PHASE6_DIR/FD002_static_fixed"
mkdir -p "$OUT_DIR"

# Run the optimized training script (defaults to n_runs=3, epochs=250)
python -u kepin_cmapss_optimized.py \
    --dataset CMAPSS_FD002 \
    --output_dir "$OUT_DIR" \
    --condition_dim 0 \
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

echo "static_fixed,0,$HOST,$RMSE,$SCORE" >> "$RESULTS_CSV"
echo "Finished $var: RMSE=$RMSE, Score=$SCORE"

cp -r "$PHASE6_DIR" ~/Kepin_code/experiments_result/ablation_conditioning/
echo "Phase 6 static baseline completed."
