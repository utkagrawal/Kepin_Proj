#!/bin/bash
# run_phase9_latent_inject.sh
# Runs phase 9: Latent Injection conditioning on FD002 with STATIC Koopman block.
# Strategy: inject regime (Alt/Mach/TRA) into the encoder latent space rather than the Koopman matrices.
#SBATCH --partition=gpu-P100
#SBATCH --nodelist=gpu-P100-02
#SBATCH --job-name=phase9_latent
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=phase9_latent.log
#SBATCH --error=phase9_latent.err

# 1. Setup conda and environment
source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

# 2. Setup output directories
PHASE9_DIR="experiments_result/ablation_conditioning/phase9_latent_inject"
mkdir -p "$PHASE9_DIR"
RESULTS_CSV="$PHASE9_DIR/results_table.csv"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "variant,run_idx,host,rmse,score" > "$RESULTS_CSV"
fi

HOST="gpu-P100-02"
var="latent_inject"
echo "Running Phase 9 Latent Injection: $var"

OUT_DIR="$PHASE9_DIR/FD002_latent_inject_$var"
mkdir -p "$OUT_DIR"

# Run the optimized training script with latent_inject strategy
python -u kepin_cmapss_optimized.py \
    --dataset CMAPSS_FD002 \
    --output_dir "$OUT_DIR" \
    --condition_dim 3 \
    --conditioning "latent_inject" \
    > "$OUT_DIR/run.log" 2>&1

JSON_FILE="$OUT_DIR/optimized_cmapss_results.json"
if [ -f "$JSON_FILE" ]; then
    RMSE=$(python -c "
import json, statistics
results = json.load(open('$JSON_FILE'))
rmses = [r['rmse'] for r in results if 'rmse' in r]
print(statistics.mean(rmses))
")
    SCORE=$(python -c "
import sys, os
lpath = '$OUT_DIR/run.log'
scores = []
if os.path.exists(lpath):
    for line in open(lpath).readlines():
        if 'Score:' in line:
            try: scores.append(float(line.split('Score:')[-1].split(',')[0].strip()))
            except: pass
print(round(sum(scores)/len(scores), 4) if scores else 'N/A')
")
else
    RMSE="ERROR"
    SCORE="ERROR"
fi

echo "latent_inject_$var,0,$HOST,$RMSE,$SCORE" >> "$RESULTS_CSV"
echo "Finished $var: RMSE=$RMSE, Score=$SCORE"

# Copy results back to home
cp -r "$PHASE9_DIR" ~/Kepin_code/experiments_result/ablation_conditioning/
echo "Phase 9 latent injection completed."
