#!/bin/bash
# run_phase4_screening.sh
# Runs phase 4 ablation screening on FD002
#SBATCH --partition=gpu-P100
#SBATCH --nodelist=gpu-P100-01
#SBATCH --job-name=phase4_screen
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=phase4_screen.log
#SBATCH --error=phase4_screen.err

# 1. Setup conda and environment
source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

# 2. Setup output directories
PHASE4_DIR="experiments_result/ablation_conditioning/phase4"
mkdir -p "$PHASE4_DIR"
RESULTS_CSV="$PHASE4_DIR/results_table.csv"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "variant,run_idx,host,rmse,score" > "$RESULTS_CSV"
fi

HOST="gpu-P100-01"

VARIANTS=("sensor_topk" "hybrid")
DIMS=(5 2)

for i in "${!VARIANTS[@]}"; do
    var="${VARIANTS[$i]}"
    dim="${DIMS[$i]}"
    
    variant_name="${var}"
    if [ "$var" = "sensor_topk" ]; then
        variant_name="${var}_k${dim}"
    fi

    echo "Running variant: $variant_name"
    OUT_DIR="$PHASE4_DIR/FD002_$variant_name"
    mkdir -p "$OUT_DIR"
    
    python -u kepin_cmapss_optimized.py \
        --dataset CMAPSS_FD002 \
        --output_dir "$OUT_DIR" \
        --condition_dim $dim \
        --conditioning "$var" \
        --epochs 50 \
        --patience 10 \
        --n_runs 1 \
        > "$OUT_DIR/run.log" 2>&1
        
    JSON_FILE="$OUT_DIR/optimized_cmapss_results.json"
    if [ -f "$JSON_FILE" ]; then
        RMSE=$(python -c "import json; print(json.load(open('$JSON_FILE'))[0].get('rmse', 'N/A'))")
        
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
    
    echo "$variant_name,0,$HOST,$RMSE,$SCORE" >> "$RESULTS_CSV"
    echo "Finished $variant_name: RMSE=$RMSE, Score=$SCORE"
done

cp -r "$PHASE4_DIR" ~/Kepin_code/experiments_result/ablation_conditioning/
echo "Phase 4 initial variants completed."
