#!/bin/bash
#SBATCH --job-name=kepin_regime
#SBATCH --partition=gpu-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/userhome/mtech/a.utkarsh/RCKePIN/kepin_regime/kepin_train_%j.out

cd "$TMPDIR" || exit 1

cp -r /userhome/mtech/a.utkarsh/RCKePIN/kepin_regime/* .
python run_regime_experiments.py --output experiments_regime
cp -r experiments_regime /userhome/mtech/a.utkarsh/RCKePIN/kepin_regime/
