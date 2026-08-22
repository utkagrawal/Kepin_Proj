#!/bin/bash
# run_phase3_screening.sh
# Runs phase 3 ablation screening on FD002 for 4 conditioning variants
#SBATCH --partition=gpu-P100
#SBATCH --nodelist=gpu-P100-01
#SBATCH --job-name=phase3_screen
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=phase3_screen.log
#SBATCH --error=phase3_screen.err

# 1. Setup conda and environment
source /userhome/mtech/a.utkarsh/miniconda3/etc/profile.d/conda.sh
conda activate kepin
export LD_LIBRARY_PATH=$(python -c 'import os, glob, site; print(":".join(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))))'):$LD_LIBRARY_PATH

cd "$TMPDIR" || exit 1
cp -r ~/Kepin_code .
cd Kepin_code

# 2. Setup output directories
PHASE3_DIR="experiments_result/ablation_conditioning/phase3"
mkdir -p "$PHASE3_DIR"
RESULTS_CSV="$PHASE3_DIR/results_table.csv"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "variant,run_idx,host,rmse,score" > "$RESULTS_CSV"
fi

HOST="gpu-P100-01"

VARIANTS=("raw2_ab" "raw2_ac" "raw2_bc" "regime_embed")

for var in "${VARIANTS[@]}"; do
    echo "Running variant: $var"
    OUT_DIR="$PHASE3_DIR/FD002_$var"
    mkdir -p "$OUT_DIR"
    
    # Run the optimized training script
    python -u kepin_cmapss_optimized.py \
        --dataset CMAPSS_FD002 \
        --output_dir "$OUT_DIR" \
        --condition_dim 3 \
        --conditioning "$var" \
        --epochs 50 \
        --patience 10 \
        --n_runs 1 \
        > "$OUT_DIR/run.log" 2>&1
        
    # Extract RMSE and Score from the JSON result file
    # The script saves results in OUT_DIR/CMAPSS_FD002/optimized_cmapss_results.json
    JSON_FILE="$OUT_DIR/CMAPSS_FD002/optimized_cmapss_results.json"
    if [ -f "$JSON_FILE" ]; then
        RMSE=$(python -c "import json; print(json.load(open('$JSON_FILE'))[0].get('rmse', 'N/A'))")
        SCORE=$(python -c "import json; print(json.load(open('$JSON_FILE'))[0].get('score', 'N/A'))")
    else
        RMSE="ERROR"
        SCORE="ERROR"
    fi
    
    echo "$var,0,$HOST,$RMSE,$SCORE" >> "$RESULTS_CSV"
    echo "Finished $var: RMSE=$RMSE, Score=$SCORE"
done

cp -r "$PHASE3_DIR" ~/Kepin_code/experiments_result/ablation_conditioning/
echo "All screening variants completed."
